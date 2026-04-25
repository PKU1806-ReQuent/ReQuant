# coding=utf-8
"""
AWQ with data-parallel activation-stat collection (multi-process, multi-GPU).

The implementation follows the same outer PTQ flow as the existing GPTAQ DP path:
1. Capture / shard first-layer inputs across ranks.
2. Quantize decoder layers sequentially.
3. For each layer, collect per-channel activation importance for the AWQ scaling groups
   (`qkv` and `up/gate`) over local samples, then all-reduce to global statistics.
4. Apply equivalent scaling to the preceding norm and grouped linear weights.
5. Quantize the layer weights on every rank, then broadcast parameters from rank 0.

This keeps the layerwise sequential PTQ contract intact while making the AWQ search
see the full calibration set across GPUs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from gptq_utils.dp_common import (
    capture_calibration_state,
    local_sample_indices,
    propagate_layer_inputs,
    run_layer_sample,
)
from requant import FPInputsCache, requant_layer
from utils import dist_utils, memory_utils, model_utils, quant_utils


def _unwrap_linear_module(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def _quantize_tensor_with_args(weight: torch.Tensor, args) -> torch.Tensor:
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.w_bits,
        perchannel=True,
        sym=not args.w_asym,
        mse=args.w_clip,
        weight_groupsize=args.w_groupsize,
    )
    quantizer.find_params(weight)
    q_weight, _, _ = quantizer.fake_quantize(weight)
    return q_weight


def _search_awq_scale(
    modules: Sequence[torch.nn.Module],
    act_mean: torch.Tensor,
    args,
) -> torch.Tensor:
    act_mean = act_mean.float().clamp_min(float(args.awq_scale_eps))
    merged_weight = torch.cat(
        [_unwrap_linear_module(module).weight.data.float().abs() for module in modules], dim=0
    )
    weight_mean = merged_weight.mean(dim=0).clamp_min(float(args.awq_scale_eps))

    if args.awq_grid <= 1:
        alpha_candidates = [float(args.awq_max_alpha)]
    else:
        alpha_candidates = torch.linspace(
            float(args.awq_min_alpha),
            float(args.awq_max_alpha),
            steps=int(args.awq_grid),
            device=act_mean.device,
            dtype=torch.float32,
        ).tolist()

    best_scale = torch.ones_like(act_mean)
    best_score = None
    act_importance = act_mean.unsqueeze(0)
    eps = float(args.awq_scale_eps)
    for alpha in alpha_candidates:
        scale = act_mean.pow(alpha) / weight_mean.pow(1.0 - alpha)
        scale = scale.clamp_min(eps)
        scale = scale / scale.mean().clamp_min(eps)

        score = 0.0
        scale_row = scale.unsqueeze(0)
        for module in modules:
            weight = _unwrap_linear_module(module).weight.data.float()
            scaled_weight = weight * scale_row
            q_weight = _quantize_tensor_with_args(scaled_weight, args)
            diff = (q_weight - scaled_weight).pow(2)
            score += float((diff * act_importance).mean().item())

        if best_score is None or score < best_score:
            best_score = score
            best_scale = scale.clone()

    return best_scale


def _apply_awq_scale(
    norm_module: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    scale: torch.Tensor,
) -> None:
    scale = scale.to(device=next(norm_module.parameters()).device, dtype=next(norm_module.parameters()).dtype)
    if hasattr(norm_module, "weight") and norm_module.weight is not None:
        norm_module.weight.data.div_(scale)
    if hasattr(norm_module, "bias") and norm_module.bias is not None:
        norm_module.bias.data.div_(scale)

    scale_row = scale.unsqueeze(0)
    for module in modules:
        linear = _unwrap_linear_module(module)
        linear.weight.data.mul_(scale_row.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _collect_group_activation_mean(
    layer: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: torch.Tensor,
    input_indices: Sequence[int],
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    group: Optional[Any] = None,
) -> torch.Tensor:
    stats = {"sum_abs": None, "count": 0}

    def hook_fn(_, inp, __):
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        sum_abs = x.float().abs().sum(dim=0)
        if stats["sum_abs"] is None:
            stats["sum_abs"] = sum_abs
        else:
            stats["sum_abs"] += sum_abs
        stats["count"] += x.shape[0]

    handle = hook_module.register_forward_hook(hook_fn)
    try:
        for idx in input_indices:
            _ = layer(
                inputs[idx].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
    finally:
        handle.remove()

    if stats["sum_abs"] is None:
        hidden = _unwrap_linear_module(hook_module).weight.shape[1]
        stats["sum_abs"] = torch.zeros(hidden, device=dev, dtype=torch.float32)

    sum_abs = stats["sum_abs"].to(dev, dtype=torch.float32)
    count = torch.tensor([float(stats["count"])], device=dev, dtype=torch.float64)
    dist.all_reduce(sum_abs, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    if count.item() <= 0:
        raise RuntimeError("awq_dp: no activation samples were collected for an AWQ group.")
    return sum_abs / count.to(dtype=sum_abs.dtype)


def _quantize_layer_weights(
    layer_idx: int,
    full: dict[str, torch.nn.Module],
    sequential_names: Sequence[Sequence[str]],
    args,
    rank: int,
) -> dict[str, quant_utils.WeightQuantizer]:
    quantizers = {}
    for names in sequential_names:
        for name in names:
            module = full.get(name, full.get(name + ".module", None))
            if module is None or "lm_head" in name:
                continue
            weight = module.weight.data.float()
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits,
                perchannel=True,
                sym=not args.w_asym,
                mse=args.w_clip,
                weight_groupsize=args.w_groupsize,
            )
            quantizer.find_params(weight)
            q_weight, _, _ = quantizer.fake_quantize(weight)
            module.weight.data = q_weight.to(module.weight.data.dtype)
            quantizers[f"model.layers.{layer_idx}.{name}"] = quantizer.cpu()
    return quantizers


@torch.no_grad()
def awq_fwrd_data_parallel(
    args,
    analyzer: model_utils.ModelAnalyzer,
    dataloader,
    dev: torch.device,
    group: Optional[Any] = None,
):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "awq_dp: awq_fwrd_data_parallel requires torch.distributed; "
            "launch with torchrun (see ptq_dp.py)."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_js = local_sample_indices(rank, world_size, args.nsamples)
    shard_inps = bool(getattr(args, "dp_shard_inps", False) and world_size > 1)
    logging.info(
        "-----AWQ DP rank=%d world=%d local_samples=%d/%d dp_shard_inps=%s-----",
        rank,
        world_size,
        len(local_js),
        args.nsamples,
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

    use_requant = getattr(args, "requant_sweeps", 0) > 0
    state = capture_calibration_state(
        args, model, layers, dataloader, dev, rank, world_size, group,
        clone_fp=use_requant,
    )
    local_js = state.local_indices
    shard_inps = state.shard_inputs

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    quantizers = {}
    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (AWQ DP)",
        disable=(rank != 0),
    )
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential) if use_requant else None
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)
        bits_config = quant_utils.disable_act_quant(layer)

        awq_norms = analyzer.get_layernorms(layer)
        awq_input_groups = analyzer.get_perlayer_input_modules(layer)
        for norm_module, modules in zip(awq_norms, awq_input_groups):
            if not modules:
                continue
            hook_module = modules[0]
            if shard_inps:
                group_inputs = state.inps_shard
                group_indices = range(len(local_js))
            else:
                group_inputs = state.inps
                group_indices = local_js
            assert group_inputs is not None
            act_mean = _collect_group_activation_mean(
                layer,
                hook_module,
                group_inputs,
                group_indices,
                dev,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
                group=group,
            )
            scale = _search_awq_scale(modules, act_mean, args)
            _apply_awq_scale(norm_module, modules, scale)

        fp_weights = {}
        for names in sequential:
            for name in names:
                module = full.get(name, full.get(name + ".module", None))
                if module is not None and "lm_head" not in name:
                    fp_weights[f"model.layers.{i}.{name}"] = (
                        _unwrap_linear_module(module).weight.data.float().clone()
                    )
        if use_requant:
            assert fp_inputs_cache is not None
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
        quantizers.update(_quantize_layer_weights(i, full, sequential, args, rank))
        if use_requant:
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
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----AWQ DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers
