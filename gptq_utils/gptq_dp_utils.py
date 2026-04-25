# coding=utf-8
"""GPTQ/GPTAQ (+ optional ReQuant Phase-2) with data-parallel Hessian / dXXT.

Each rank runs calibration on its shard of sample indices; local ``H`` / ``dXXT``
are weight-merged across ranks (all-reduce, or reduce-to-rank0 with
``--dp_fasterquant_master_only``). Set ``use_requant=True`` to swap ``GPTAQ``
for ``GPTQReQuant`` (Phase-1 P-matrix + Phase-2 coordinate descent).

Flags: ``--dp_fasterquant_master_only`` (rank0-only fasterquant + broadcast),
``--dp_shard_inps`` (scatter ``inps`` per-rank to save GPU memory).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from gptq_utils.gptq_utils import GPTQ
from gptq_utils.gptaq_utils import GPTAQ
from gptq_utils.dp_common import (
    capture_calibration_state,
    local_sample_indices,
    propagate_layer_inputs,
    run_layer_sample,
)
from requant import FPInputsCache, GPTQReQuant
from utils import memory_utils, model_utils, quant_utils


def _merge_hessian_stats(
    H_loc: torch.Tensor,
    dXXT_loc: torch.Tensor,
    n_loc: int,
    device: torch.device,
    rank: int,
    group: Optional[Any] = None,
    master_only: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """Weighted merge of per-rank H/dXXT."""
    reducer = (
        (lambda t: dist.reduce(t, dst=0, op=dist.ReduceOp.SUM, group=group))
        if master_only else
        (lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group))
    )

    n_buf = torch.tensor([float(n_loc)], device=device, dtype=torch.float64)
    reducer(n_buf)
    n_total = float(n_buf.item()) if (not master_only or rank == 0) else 0.0

    w = float(n_loc)
    h_acc = H_loc * w
    d_acc = dXXT_loc * w if dXXT_loc is not None else torch.zeros_like(H_loc)
    reducer(h_acc)
    reducer(d_acc)

    if master_only and rank != 0:
        return None, None, 0.0
    if n_total < 1.0:
        raise RuntimeError(
            "gptq_dp: total calibration token count N is zero; "
            "check nsamples and dataloader."
        )
    return h_acc / n_total, d_acc / n_total, n_total


@torch.no_grad()
def gptq_fwrd_data_parallel(
    args,
    analyzer: model_utils.ModelAnalyzer,
    dataloader,
    dev: torch.device,
    group: Optional[Any] = None,
    use_requant: bool = False,
):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "gptq_fwrd_data_parallel requires torch.distributed; launch with torchrun."
        )

    master_only_flag = bool(getattr(args, "dp_fasterquant_master_only", False))
    if master_only_flag and getattr(args, "export_to_et", False):
        raise ValueError(
            "dp_fasterquant_master_only is not supported with export_to_et."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    master_only = bool(master_only_flag and world_size > 1)
    use_phase1_dxxt = args.w_method == "gptaq"
    collect_dxxt = use_phase1_dxxt or use_requant
    if args.w_method not in ("gptq", "gptaq"):
        raise ValueError(
            f"gptq_fwrd_data_parallel expects w_method in {{gptq,gptaq}}; got {args.w_method}"
        )

    local_js = local_sample_indices(rank, world_size, args.nsamples)
    shard_inps = bool(getattr(args, "dp_shard_inps", False) and world_size > 1)

    logging.info(
        "-----GPTQ DP rank=%d world=%d local=%d/%d master_only=%s shard=%s "
        "method=%s use_requant=%s collect_dxxt=%s phase1_dxxt=%s-----",
        rank, world_size, len(local_js), args.nsamples,
        master_only, shard_inps, args.w_method, use_requant,
        collect_dxxt, use_phase1_dxxt,
    )

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    state = capture_calibration_state(
        args, model, layers, dataloader, dev, rank, world_size, group, clone_fp=True
    )
    local_js = state.local_indices
    shard_inps = state.shard_inputs

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    quantizers: dict = {}
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential) if collect_dxxt else None

    if use_requant:
        requant_beta = getattr(args, "requant_beta", None)
        if requant_beta is None:
            requant_beta = 0.3
        requant_kwargs = dict(
            export_to_et=getattr(args, "export_to_et", False),
            requant_sweeps=getattr(args, "requant_phase2_sweeps", 3),
            requant_candidates=getattr(args, "requant_phase2_candidates", 2),
            requant_anchor_lambda=getattr(args, "requant_anchor_lambda", 0.0),
            requant_beta=requant_beta,
            requant_min_gain_eps=args.requant_min_gain_eps,
            requant_early_stop_consecutive_cols=getattr(
                args, "requant_early_stop_consecutive_cols", 0
            ),
        )
    else:
        requant_kwargs = {}

    def _run(idx, src, layer, store=True):
        run_layer_sample(
            layer,
            src,
            idx,
            dev,
            state.attention_mask,
            state.position_ids,
            state.position_embeddings,
            store=store,
        )

    pbar = tqdm(
        range(len(layers)), ncols=120,
        desc=f"Quantizing Layers ({args.w_method.upper()} DP)", disable=(rank != 0),
    )
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        if collect_dxxt:
            assert fp_inputs_cache is not None
            bits_config = quant_utils.disable_act_quant(layer)
            fp_inputs_cache.add_hook(full)
            if shard_inps:
                assert state.fp_inps_shard is not None
                for k in range(len(local_js)):
                    _run(k, state.fp_inps_shard, layer)
            else:
                assert state.fp_inps is not None
                for j in local_js:
                    _run(j, state.fp_inps, layer)
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)

        gptq: Optional[dict] = None
        for names in sequential:
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

            gptq = {}
            for name in subset:
                if "lm_head" in name:
                    continue
                if use_requant:
                    gptq[name] = GPTQReQuant(
                        subset[name],
                        use_dxxt=collect_dxxt,
                        use_phase1_dxxt=use_phase1_dxxt,
                    )
                elif use_phase1_dxxt:
                    gptq[name] = GPTAQ(subset[name])
                else:
                    gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    args.w_bits, perchannel=True, sym=not args.w_asym, mse=args.w_clip,
                )
                if collect_dxxt:
                    assert fp_inputs_cache is not None
                    gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]
            if not gptq:
                continue

            first_name = next(iter(gptq))

            def _add_batch(name):
                def _hook(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return _hook

            hook_module = getattr(subset[first_name], "module", subset[first_name])
            handle = hook_module.register_forward_hook(_add_batch(first_name))
            if shard_inps:
                assert state.inps_shard is not None
                for k in range(len(local_js)):
                    _run(k, state.inps_shard, layer, store=False)
            else:
                assert state.inps is not None
                for j in local_js:
                    _run(j, state.inps, layer, store=False)
            handle.remove()

            if world_size == 1:
                H_glob = gptq[first_name].H
                dXXT_glob = gptq[first_name].dXXT if collect_dxxt else None
                n_total = float(gptq[first_name].nsamples)
            else:
                H_glob, dXXT_glob, n_total = _merge_hessian_stats(
                    gptq[first_name].H, getattr(gptq[first_name], "dXXT", None),
                    int(gptq[first_name].nsamples),
                    dev, rank, group, master_only=master_only,
                )

            if rank == 0:
                logging.debug(
                    "layer %d group %s: N_local=%d N_global=%.0f",
                    i, names, gptq[first_name].nsamples, n_total,
                )

            q_names = list(gptq.keys())
            if master_only and rank != 0:
                for name in q_names:
                    gptq[name].free()
            else:
                gptq[first_name].H = H_glob
                if hasattr(gptq[first_name], "dXXT"):
                    gptq[first_name].dXXT = dXXT_glob
                for name in q_names:
                    if name != first_name:
                        gptq[name].H = gptq[first_name].H
                        if hasattr(gptq[name], "dXXT"):
                            gptq[name].dXXT = gptq[first_name].dXXT

            for name in q_names:
                pbar.set_postfix(module=f"layers.{i}." + name)
                if rank == 0 or not master_only:
                    fasterquant_kwargs = dict(
                        percdamp=args.percdamp,
                        groupsize=args.w_groupsize,
                        actorder=args.act_order,
                        static_groups=args.act_order,
                    )
                    if use_phase1_dxxt or use_requant:
                        fasterquant_kwargs["alpha"] = args.alpha
                    fasterquant_kwargs.update(requant_kwargs)
                    gptq[name].fasterquant(**fasterquant_kwargs)
                    quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
                    gptq[name].free()
                if master_only and world_size > 1:
                    dist.broadcast(subset[name].weight.data, src=0, group=group)

        propagate_layer_inputs(args, layer, state, dev, rank, world_size, group)

        if fp_inputs_cache is not None:
            fp_inputs_cache.clear_cache()
        layers[i] = layer.to(orig_device)
        del layer
        if gptq is not None:
            del gptq
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----%s DP Done (rank=%d)-----\n", args.w_method.upper(), rank)
    return quantizers
