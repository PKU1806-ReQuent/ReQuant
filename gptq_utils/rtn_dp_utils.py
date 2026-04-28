# coding=utf-8
"""RTN (+ optional ReQuant) with the shared data-parallel PTQ flow."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from gptq_utils.dp_common import (
    capture_calibration_state,
    propagate_layer_inputs,
    run_layer_sample,
)
from requant import FPInputsCache, requant_layer
from utils import dist_utils, memory_utils, model_utils, quant_utils


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def _collect_fp_weights(layer_idx: int, full: dict, sequential: Sequence[Sequence[str]]):
    fp_weights = {}
    for names in sequential:
        for name in names:
            module = full.get(name, full.get(name + ".module", None))
            if module is None or "lm_head" in name:
                continue
            fp_weights[f"model.layers.{layer_idx}.{name}"] = (
                _unwrap(module).weight.data.float().detach().cpu().clone()
            )
    return fp_weights


def _quantize_layer_weights(layer_idx: int, full: dict, sequential: Sequence[Sequence[str]], args):
    quantizers = {}
    for names in sequential:
        for name in names:
            module = full.get(name, full.get(name + ".module", None))
            if module is None or "lm_head" in name:
                continue
            module = _unwrap(module)
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits,
                perchannel=True,
                sym=not args.w_asym,
                mse=args.w_clip,
                weight_groupsize=args.w_groupsize,
            )
            weight = module.weight.data
            quantizer.find_params(weight)
            q_weight, _, _ = quantizer.fake_quantize(weight)
            module.weight.data = q_weight.to(module.weight.data.dtype)
            quantizers[f"model.layers.{layer_idx}.{name}"] = quantizer.cpu()
    return quantizers


@torch.no_grad()
def rtn_fwrd_data_parallel(
    args,
    analyzer: model_utils.ModelAnalyzer,
    dataloader,
    dev: torch.device,
    group: Optional[Any] = None,
):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "rtn_dp: rtn_fwrd_data_parallel requires torch.distributed; launch with torchrun."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    use_requant = getattr(args, "requant_sweeps", 0) > 0
    logging.info(
        "-----RTN DP rank=%d world=%d requant=%s dp_shard_inps=%s-----",
        rank,
        world_size,
        use_requant,
        bool(getattr(args, "dp_shard_inps", False) and world_size > 1),
    )

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    state = None
    if use_requant:
        if dataloader is None:
            raise ValueError("rtn DP requires calibration dataloader when requant_sweeps > 0")
        for module in analyzer.get_pre_block_modules():
            module.to(dev)
        layers[0] = layers[0].to(dev)
        state = capture_calibration_state(
            args, model, layers, dataloader, dev, rank, world_size, group, clone_fp=True
        )
        layers[0] = layers[0].to(orig_device)
        memory_utils.cleanup_memory(False)

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential) if use_requant else None

    for i in tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (RTN DP)",
        disable=(rank != 0),
    ):
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        fp_weights = None
        if use_requant:
            assert state is not None and fp_inputs_cache is not None
            bits_config = quant_utils.disable_act_quant(layer)
            fp_inputs_cache.add_hook(full)
            if state.shard_inputs:
                assert state.fp_inps_shard is not None
                fp_src = state.fp_inps_shard
                fp_indices = range(len(state.local_indices))
            else:
                assert state.fp_inps is not None
                fp_src = state.fp_inps
                fp_indices = state.local_indices
            for idx in fp_indices:
                run_layer_sample(
                    layer,
                    fp_src,
                    idx,
                    dev,
                    state.attention_mask,
                    state.position_ids,
                    state.position_embeddings,
                    store=True,
                )
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)
            fp_weights = _collect_fp_weights(i, full, sequential)

        quantizers.update(_quantize_layer_weights(i, full, sequential, args))

        if use_requant:
            assert state is not None and fp_inputs_cache is not None
            if state.shard_inputs:
                assert state.inps_shard is not None
                rq_inps = state.inps_shard
                rq_indices = range(len(state.local_indices))
            else:
                assert state.inps is not None
                rq_inps = state.inps
                rq_indices = state.local_indices
            requant_layer(
                layer,
                full,
                sequential,
                i,
                quantizers,
                fp_weights,
                rq_inps,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
                args,
                dev,
                input_indices=rq_indices,
                group=group,
                fp_inputs_cache=fp_inputs_cache.fp_cache,
            )
            fp_inputs_cache.clear_cache()
            del fp_weights
            memory_utils.cleanup_memory(False)

        if world_size > 1:
            dist_utils.broadcast_parameters(layer, src=0, group=group)

        if use_requant:
            assert state is not None
            propagate_layer_inputs(args, layer, state, dev, rank, world_size, group)

        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory()

    if use_requant:
        for module in analyzer.get_pre_block_modules():
            module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----RTN DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers
