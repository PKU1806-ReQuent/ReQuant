# coding=utf-8
"""
GPTAQ-Refined with data-parallel Hessian / dXXT accumulation (multi-process, multi-GPU).

Each rank runs calibration forwards only on its shard of sample indices; local H, dXXT
match the single-rank online ``add_batch`` recurrence on that shard.  Globally:

    H*     = (1/N) * sum_r  n_r * H_r
    dXXT*  = (1/N) * sum_r  n_r * dXXT_r

where n_r is ``GPTAQRefined.nsamples`` (per-rank ``add_batch`` 调用次数) on rank r
and N = sum_r n_r.  Under exact arithmetic this matches one sequential pass; see
``scripts/compare_h_dxxt_dp.py``.

After the initial hook pass, **``inps`` is broadcast from rank 0** so every rank matches
the same calibration activations before layer 0. After each layer, **``inps`` are updated
on rank 0 only and broadcast** so all ranks share the same inputs for the next layer
(avoids multi-GPU nondeterminism splitting ``inps`` across ranks and drifting away from
single-GPU behaviour).  FP activations ``fp_inps`` / ``fp_inps_shard`` follow the same
layer-wise progression as ``gptaq_dp_utils`` (no per-layer forced resync from ``inps``).

This module is standalone; it imports ``GPTAQRefined`` / ``FPInputsCache`` from
``gptaq_refined_utils`` (Phase-1 matches ``GPTAQ.fasterquant``; Phase-2 ``_refine``: vectorised coordinates;
accept when ``best_gain < -refine_min_gain_eps``, optional consecutive-column early stop).
DP only changes how ``H`` / ``dXXT``
are accumulated (sharded forwards + all-reduce) before calling ``fasterquant``.

With ``--dp_fasterquant_master_only``, merged ``H``/``dXXT`` are reduced to rank 0 only (workers drop
local Hessians), ``fasterquant`` runs only on rank 0, and quantized weights are ``broadcast`` to
workers—same math as all-reduce + per-rank ``fasterquant``, but workers avoid holding the full
global Hessian and the associated factorization/refinement peak memory.

With ``--dp_shard_inps`` (multi-rank only), rank 0 builds the full ``inps`` on **CPU**, then
``scatter_object_list`` sends each rank only its sample shard; forwards use ``inps_shard[k]`` for
``j = local_js[k]``. After each layer, every rank updates its own shard with the **same** quantized
weights—mathematically the same as rank0 updating the full tensor and broadcasting rows, without
each GPU storing all ``nsamples`` rows at once.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from gptq_utils.gptaq_refined_utils import FPInputsCache, GPTAQRefined
from utils import memory_utils, model_utils, quant_utils


def _move_attn_struct_to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensor leaves to ``device`` (for mask / position_ids / embeddings)."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_move_attn_struct_to_device(t, device) for t in x)
    if isinstance(x, list):
        return [_move_attn_struct_to_device(t, device) for t in x]
    return x


def _local_sample_indices(rank: int, world_size: int, nsamples: int) -> List[int]:
    """Split ``range(nsamples)`` across ranks (contiguous shards)."""
    base, rem = divmod(nsamples, world_size)
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return list(range(start, start + count))


def _merge_hessian_stats(
    H_loc: torch.Tensor,
    dXXT_loc: torch.Tensor,
    n_loc: int,
    device: torch.device,
    group: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    All-reduce merged H and dXXT.  Ranks with n_loc == 0 should pass zero tensors
    for H_loc / dXXT_loc (same shape as on other ranks).
    """
    n_t = torch.tensor([float(n_loc)], device=device, dtype=torch.float64)
    dist.all_reduce(n_t, op=dist.ReduceOp.SUM, group=group)
    n_total = float(n_t.item())
    if n_total < 1.0:
        raise RuntimeError(
            "gptaq_refined_dp: total calibration token count N is zero; "
            "check nsamples and dataloader."
        )

    scale = float(n_loc)
    h_acc = H_loc * scale
    dist.all_reduce(h_acc, op=dist.ReduceOp.SUM, group=group)
    h_glob = h_acc / n_total

    d_acc = dXXT_loc * scale
    dist.all_reduce(d_acc, op=dist.ReduceOp.SUM, group=group)
    d_glob = d_acc / n_total

    return h_glob, d_glob, n_total


def _merge_hessian_stats_to_rank0(
    H_loc: torch.Tensor,
    dXXT_loc: torch.Tensor,
    n_loc: int,
    device: torch.device,
    rank: int,
    group: Optional[Any] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """
    Same weighted merge as ``_merge_hessian_stats``, but sum only on rank 0 via ``reduce``.
    Non-root ranks must not use the returned tensors (they get ``None``).
    """
    n_buf = torch.tensor([float(n_loc)], device=device, dtype=torch.float64)
    dist.reduce(n_buf, dst=0, op=dist.ReduceOp.SUM, group=group)
    n_total = float(n_buf.item()) if rank == 0 else 0.0

    scale = float(n_loc)
    h_acc = H_loc * scale
    dist.reduce(h_acc, dst=0, op=dist.ReduceOp.SUM, group=group)
    d_acc = dXXT_loc * scale
    dist.reduce(d_acc, dst=0, op=dist.ReduceOp.SUM, group=group)

    if rank == 0:
        if n_total < 1.0:
            raise RuntimeError(
                "gptaq_refined_dp: total calibration token count N is zero; "
                "check nsamples and dataloader."
            )
        return h_acc / n_total, d_acc / n_total, n_total
    return None, None, 0.0


@torch.no_grad()
def gptq_refined_fwrd_data_parallel(
    args,
    analyzer: model_utils.ModelAnalyzer,
    dataloader,
    dev: torch.device,
    group: Optional[Any] = None,
):
    """
    Same outer flow as ``gptaq_refined_fwrd``, but ranks shard calibration indices
    for FP + quant hooks, then merge H/dXXT before ``fasterquant`` on every rank.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "gptaq_refined_fwrd_data_parallel requires torch.distributed; "
            "launch with torchrun (see scripts/gptq_refined_dp.sh)."
        )

    if getattr(args, "dp_fasterquant_master_only", False) and getattr(
        args, "export_to_et", False
    ):
        raise ValueError(
            "dp_fasterquant_master_only is not supported with export_to_et "
            "(extra buffers on rank 0 would not exist on workers)."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_js = _local_sample_indices(rank, world_size, args.nsamples)

    master_only = bool(
        getattr(args, "dp_fasterquant_master_only", False) and world_size > 1
    )
    shard_inps = bool(getattr(args, "dp_shard_inps", False) and world_size > 1)
    logging.info(
        "-----GPTAQ-Refined DP (H/dXXT sharded) rank=%d world=%d local_samples=%d/%d "
        "fasterquant_master_only=%s dp_shard_inps=%s-----",
        rank,
        world_size,
        len(local_js),
        args.nsamples,
        master_only,
        shard_inps,
    )

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    seqlen = model.seqlen

    inps_shard: Optional[torch.Tensor] = None
    fp_inps_shard: Optional[torch.Tensor] = None
    inps: Optional[torch.Tensor] = None
    fp_inps: Optional[torch.Tensor] = None

    if shard_inps:
        # 仅 rank0 在 CPU 上建完整 inps；scatter 后每 rank 只保留约 1/world_size 行（可再配合 offload）
        cache = {"i": 0, "attention_mask": None}
        inps_full_cpu: Optional[torch.Tensor] = None
        if rank == 0:
            inps_full_cpu = torch.zeros(
                (args.nsamples, seqlen, hidden), dtype=dtype, device="cpu"
            )

        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type

            def forward(self, inp, **kwargs):
                assert inps_full_cpu is not None
                inps_full_cpu[cache["i"]] = inp.detach().cpu()
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                cache["position_embeddings"] = kwargs["position_embeddings"]
                raise ValueError

        if rank == 0:
            layers[0] = Catcher(layers[0])
            for batch in dataloader:
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass
            layers[0] = layers[0].module
        dist.barrier(group=group)

        def _to_cpu_meta(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu()
            return x

        if rank == 0:
            meta = [
                _to_cpu_meta(cache["attention_mask"]),
                _to_cpu_meta(cache["position_ids"]),
                _to_cpu_meta(cache["position_embeddings"]),
            ]
        else:
            meta = [None, None, None]
        dist.broadcast_object_list(meta, src=0, group=group)
        attention_mask, position_ids, position_embeddings = meta[0], meta[1], meta[2]
        attention_mask = _move_attn_struct_to_device(attention_mask, dev)
        position_ids = _move_attn_struct_to_device(position_ids, dev)
        position_embeddings = _move_attn_struct_to_device(position_embeddings, dev)

        if rank == 0:
            assert inps_full_cpu is not None
            scatter_objs = [
                inps_full_cpu[
                    _local_sample_indices(r, world_size, args.nsamples), :, :
                ].contiguous()
                for r in range(world_size)
            ]
        else:
            scatter_objs = None
        recv_list: List[Any] = [None]
        dist.scatter_object_list(recv_list, scatter_objs, src=0, group=group)
        shard_storage = torch.device("cpu") if args.offload_inps else dev
        inps_shard = recv_list[0].to(shard_storage)
        fp_inps_shard = inps_shard.clone()
        del recv_list
        if rank == 0:
            del inps_full_cpu, cache
        memory_utils.cleanup_memory(False)
    else:
        inps = torch.zeros(
            (args.nsamples, seqlen, hidden),
            dtype=dtype,
            device=dev,
        )
        cache = {"i": 0, "attention_mask": None}

        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                cache["position_embeddings"] = kwargs["position_embeddings"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]

        if world_size > 1:
            dist.broadcast(inps, src=0, group=group)

        fp_inps = inps.clone()

        if args.offload_inps:
            inps = inps.cpu()
            fp_inps = fp_inps.cpu()

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()

    fp_inputs_cache = FPInputsCache(sequential)

    refine_sweeps = getattr(args, "refine_sweeps", 3)
    refine_candidates = getattr(args, "refine_candidates", 2)
    refine_anchor_lambda = getattr(args, "refine_anchor_lambda", 0.0)
    refine_beta = getattr(args, "refine_beta", 0.25)
    refine_min_gain_eps = args.refine_min_gain_eps
    refine_early_stop_consecutive_cols = getattr(
        args, "refine_early_stop_consecutive_cols", 0
    )

    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (DP)",
        disable=(rank != 0),
    )
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)
        if shard_inps:
            for k in range(len(local_js)):
                fp_inps_shard[k] = layer(
                    fp_inps_shard[k].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0].to(fp_inps_shard.device)
        else:
            for j in local_js:
                fp_inps[j] = layer(
                    fp_inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0].to(fp_inps.device)
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        gptq = None
        for names in sequential:
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

            gptq = {}
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not args.w_asym
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                gptq[name] = GPTAQRefined(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            if not gptq:
                continue

            # 与 gptaq_refined_utils.gptq_refined_fwrd 一致：hook 挂在 subset 中第一个名字上
            first_module_name = list(subset.keys())[0]
            if first_module_name not in gptq:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handle = subset[first_module_name].register_forward_hook(
                add_batch(first_module_name)
            )
            if shard_inps:
                for k in range(len(local_js)):
                    _ = layer(
                        inps_shard[k].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
            else:
                for j in local_js:
                    _ = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
            handle.remove()

            n_loc = int(gptq[first_module_name].nsamples)
            H_loc = gptq[first_module_name].H
            dXXT_loc = gptq[first_module_name].dXXT

            q_names = [n for n in subset if n in gptq]

            if master_only:
                H_glob, dXXT_glob, n_total = _merge_hessian_stats_to_rank0(
                    H_loc, dXXT_loc, n_loc, dev, rank, group
                )
            else:
                H_glob, dXXT_glob, n_total = _merge_hessian_stats(
                    H_loc, dXXT_loc, n_loc, dev, group
                )

            if rank == 0:
                logging.debug(
                    "layer %d group %s: N_local=%d N_global=%.0f",
                    i,
                    str(names),
                    n_loc,
                    n_total,
                )

            # 与单卡 gptq_refined_fwrd 一致：全局 H/dXXT 挂在 first 上，其余模块共享同一 tensor
            if not master_only:
                gptq[first_module_name].H = H_glob
                gptq[first_module_name].dXXT = dXXT_glob
                for name in subset:
                    if name != first_module_name and name in gptq:
                        gptq[name].H = gptq[first_module_name].H
                        gptq[name].dXXT = gptq[first_module_name].dXXT
            elif rank == 0:
                gptq[first_module_name].H = H_glob
                gptq[first_module_name].dXXT = dXXT_glob
                for name in subset:
                    if name != first_module_name and name in gptq:
                        gptq[name].H = gptq[first_module_name].H
                        gptq[name].dXXT = gptq[first_module_name].dXXT
            else:
                for name in q_names:
                    gptq[name].free()

            for name in q_names:
                pbar.set_postfix(module=f"layers.{i}." + name)
                if rank == 0 or not master_only:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.w_groupsize,
                        actorder=args.act_order,
                        static_groups=args.act_order,
                        export_to_et=getattr(args, "export_to_et", False),
                        alpha=args.alpha,
                        refine_sweeps=refine_sweeps,
                        refine_candidates=refine_candidates,
                        refine_anchor_lambda=refine_anchor_lambda,
                        refine_beta=refine_beta,
                        refine_min_gain_eps=refine_min_gain_eps,
                        refine_early_stop_consecutive_cols=refine_early_stop_consecutive_cols,
                    )
                    if (not master_only) or rank == 0:
                        quantizers["model.layers.%d.%s" % (i, name)] = (
                            gptq[name].quantizer
                        )
                    gptq[name].free()
                if master_only and world_size > 1:
                    dist.broadcast(subset[name].weight.data, src=0, group=group)

        # 非分片：rank0 更新整表再广播，与各 rank 自算分片在浮点上可能略有不同
        if shard_inps:
            for k in range(len(local_js)):
                inps_shard[k] = layer(
                    inps_shard[k].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0].to(inps_shard.device)
        else:
            for j in range(args.nsamples):
                if rank == 0:
                    inps[j] = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(inps.device)
                if world_size > 1:
                    buf = (
                        inps[j].to(dev)
                        if rank == 0
                        else torch.empty(
                            inps[j].shape,
                            dtype=inps.dtype,
                            device=dev,
                        )
                    )
                    dist.broadcast(buf, src=0, group=group)
                    inps[j] = buf.to(inps.device)

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
    logging.info("-----GPTAQ-Refined DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers


def resolve_local_cuda_device() -> torch.device:
    """``cuda:LOCAL_RANK`` when ``LOCAL_RANK`` is set (torchrun); else ``cuda:0``."""
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{lr}")
