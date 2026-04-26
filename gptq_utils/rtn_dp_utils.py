# coding=utf-8
"""RTN with data-parallel calibration input propagation and optional ReQuant."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from gptq_utils.awq_dp_utils import _quantize_layer_weights, _unwrap_linear_module
from gptq_utils.dp_common import (
    capture_calibration_state,
    propagate_layer_inputs,
    run_layer_sample,
)
from requant import FPInputsCache, requant_layer
from utils import dist_utils, memory_utils, model_utils, quant_utils


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
            "rtn_dp: rtn_fwrd_data_parallel requires torch.distributed; "
            "launch with torchrun (see ptq_dp.py)."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_inps = bool(getattr(args, "dp_shard_inps", False) and world_size > 1)
    use_requant = getattr(args, "requant_sweeps", 0) > 0
    logging.info(
        "-----RTN DP rank=%d world=%d requant=%s dp_shard_inps=%s-----",
        rank,
        world_size,
        use_requant,
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

    state = capture_calibration_state(
        args,
        model,
        layers,
        dataloader,
        dev,
        rank,
        world_size,
        group,
        clone_fp=use_requant,
    )

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential) if use_requant else None
    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (RTN DP)",
        disable=(rank != 0),
    )

    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        if use_requant:
            bits_config = quant_utils.disable_act_quant(layer)
            assert fp_inputs_cache is not None
            fp_inputs_cache.add_hook(full)
            if state.shard_inputs:
                assert state.fp_inps_shard is not None
                for k in range(len(state.local_indices)):
                    run_layer_sample(
                        layer,
                        state.fp_inps_shard,
                        k,
                        dev,
                        state.attention_mask,
                        state.position_ids,
                        state.position_embeddings,
                    )
            else:
                assert state.fp_inps is not None
                for j in range(args.nsamples):
                    run_layer_sample(
                        layer,
                        state.fp_inps,
                        j,
                        dev,
                        state.attention_mask,
                        state.position_ids,
                        state.position_embeddings,
                    )
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)

            fp_weights = {}
            for names in sequential:
                for name in names:
                    module = full.get(name, full.get(name + ".module", None))
                    if module is not None and "lm_head" not in name:
                        fp_weights[f"model.layers.{i}.{name}"] = (
                            _unwrap_linear_module(module).weight.data.float().clone()
                        )
        else:
            fp_weights = None

        quantizers.update(_quantize_layer_weights(i, full, sequential, args, rank))

        if use_requant:
            if state.shard_inputs:
                assert state.inps_shard is not None
                rq_inps = state.inps_shard
                rq_indices = range(len(state.local_indices))
            else:
                assert state.inps is not None
                rq_inps = state.inps
                rq_indices = range(args.nsamples)
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

        if world_size > 1:
            dist_utils.broadcast_parameters(layer, src=0, group=group)

        propagate_layer_inputs(args, layer, state, dev, rank, world_size, group)
        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----RTN DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers
