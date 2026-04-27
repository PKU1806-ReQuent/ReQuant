# coding=utf-8
"""
RTN with distributed layer-parallel quantization (multi-process, multi-GPU).

RTN does not need calibration data, so DP mode shards layers across ranks:
- each layer is owned by one rank (round-robin),
- owner rank quantizes that layer,
- quantized layer weights/buffers are broadcast to all ranks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Sequence

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


def _quantize_layer_weights(
    layer_idx: int,
    full: dict[str, torch.nn.Module],
    sequential_names: Sequence[Sequence[str]],
    args,
) -> dict[str, quant_utils.WeightQuantizer]:
    quantizers = {}
    for names in sequential_names:
        for name in names:
            module = full.get(name, full.get(name + ".module", None))
            if module is None or "lm_head" in name:
                continue
            module = getattr(module, "module", module)
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits,
                perchannel=True,
                sym=not args.w_asym,
                mse=args.w_clip,
                weight_groupsize=args.w_groupsize,
            )
            weight = module.weight.data.float()
            quantizer.find_params(weight)
            q_weight, _, _ = quantizer.fake_quantize(weight)
            module.weight.data = q_weight.to(module.weight.data.dtype)
            quantizers[f"model.layers.{layer_idx}.{name}"] = quantizer.cpu()
    return quantizers


def _broadcast_layer(layer: torch.nn.Module, src: int, group: Any = None) -> None:
    # Some params/buffers may be non-contiguous after rotation/wrapping; NCCL
    # broadcast requires contiguous tensors, so broadcast a contiguous view.
    rank = dist.get_rank(group=group)
    for p in layer.parameters():
        buf = p.data if p.data.is_contiguous() else p.data.contiguous()
        dist.broadcast(buf, src=src, group=group)
        if (not p.data.is_contiguous()) or (rank != src):
            p.data.copy_(buf)
    for b in layer.buffers():
        if torch.is_tensor(b):
            buf = b.data if b.data.is_contiguous() else b.data.contiguous()
            dist.broadcast(buf, src=src, group=group)
            if (not b.data.is_contiguous()) or (rank != src):
                b.data.copy_(buf)


@torch.no_grad()
def rtn_fwrd_data_parallel(
    args,
    analyzer: model_utils.ModelAnalyzer,
    dev: torch.device,
    dataloader=None,
    group: Any = None,
):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "rtn_dp: rtn_fwrd_data_parallel requires torch.distributed; launch with torchrun."
        )

    model = analyzer.model
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    use_requant = getattr(args, "requant_sweeps", 0) > 0

    logging.info(
        "-----RTN DP%s rank=%d world=%d-----",
        " + ReQuant" if use_requant else " (layer-parallel)",
        rank,
        world_size,
    )

    if use_requant:
        if dataloader is None:
            raise ValueError("rtn_dp requires calibration dataloader when requant_sweeps > 0")

        use_cache = model.config.use_cache
        model.config.use_cache = False
        for module in analyzer.get_pre_block_modules():
            module.to(dev)
        layers[0] = layers[0].to(dev)

        state = capture_calibration_state(
            args, model, layers, dataloader, dev, rank, world_size, group,
            clone_fp=True,
        )
        local_js = state.local_indices
        shard_inps = state.shard_inputs

        layers[0] = layers[0].to(orig_device)
        memory_utils.cleanup_memory(False)

        quantizers: Dict[str, object] = {}
        sequential = analyzer.get_sequential_quantizable_module_names()
        fp_inputs_cache = FPInputsCache(sequential)
        pbar = tqdm(
            range(len(layers)),
            ncols=120,
            desc="Quantizing Layers (RTN DP + ReQuant)",
            disable=(rank != 0),
        )
        for i in pbar:
            layer = layers[i].to(dev)
            full = analyzer.get_quantizable_modules(layer)
            bits_config = quant_utils.disable_act_quant(layer)

            fp_weights = {}
            for names in sequential:
                for name in names:
                    module = full.get(name, full.get(name + ".module", None))
                    if module is not None and "lm_head" not in name:
                        module = getattr(module, "module", module)
                        fp_weights[f"model.layers.{i}.{name}"] = (
                            module.weight.data.float().clone()
                        )

            fp_inputs_cache.add_hook(full)
            if shard_inps:
                assert state.fp_inps_shard is not None
                for k in range(len(local_js)):
                    run_layer_sample(
                        layer,
                        state.fp_inps_shard,
                        k,
                        dev,
                        state.attention_mask,
                        state.position_ids,
                        state.position_embeddings,
                        store=True,
                    )
            else:
                assert state.fp_inps is not None
                for j in local_js:
                    run_layer_sample(
                        layer,
                        state.fp_inps,
                        j,
                        dev,
                        state.attention_mask,
                        state.position_ids,
                        state.position_embeddings,
                        store=True,
                    )
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)

            quantizers.update(_quantize_layer_weights(i, full, sequential, args))

            if shard_inps:
                assert state.inps_shard is not None
                rq_inps = state.inps_shard
                rq_indices = range(len(local_js))
            else:
                assert state.inps is not None
                rq_inps = state.inps
                rq_indices = local_js
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
            memory_utils.cleanup_memory(False)

        for module in analyzer.get_pre_block_modules():
            module.to(orig_device)
        model.config.use_cache = use_cache
        memory_utils.cleanup_memory(verbos=True)
        logging.info("-----RTN DP + ReQuant Done (rank=%d)-----\n", rank)
        return quantizers

    local_quantizers: Dict[str, object] = {}
    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (RTN DP)",
        disable=(rank != 0),
    )
    for i in pbar:
        owner = i % world_size
        layer = layers[i].to(dev)

        if rank == owner:
            subset = analyzer.get_quantizable_modules(layer)
            for name in subset:
                layer_weight_bits = args.w_bits
                w_groupsize = args.w_groupsize
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=not args.w_asym,
                    mse=args.w_clip,
                    weight_groupsize=w_groupsize,
                )
                W = subset[name].weight.data
                quantizer.find_params(W)
                q, _, _ = quantizer.fake_quantize(W)
                subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
                local_quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()

        _broadcast_layer(layer, src=owner, group=group)

        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory(verbos=False)

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_quantizers, gathered, dst=0)

    quantizers = {}
    if rank == 0 and gathered is not None:
        for part in gathered:
            if part:
                quantizers.update(part)

    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----RTN DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers


def resolve_local_cuda_device() -> torch.device:
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{lr}")
