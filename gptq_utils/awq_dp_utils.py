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


def _quantize_tensor_with_args(weight: torch.Tensor, args, *, mse: bool = False) -> torch.Tensor:
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.w_bits,
        perchannel=True,
        sym=not args.w_asym,
        mse=mse,
        weight_groupsize=args.w_groupsize,
    )
    quantizer.find_params(weight)
    q_weight, _, _ = quantizer.fake_quantize(weight)
    return q_weight


def _module_output(module: torch.nn.Module, x: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    out = module(x, **kwargs) if kwargs else module(x)
    return out[0] if isinstance(out, tuple) else out


def _capture_module_inputs(
    layer: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: torch.Tensor,
    input_indices: Sequence[int],
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook_fn(_, inp, __):
        captured.append(inp[0].detach().cpu())

    handle = hook_module.register_forward_hook(hook_fn)
    try:
        for idx in input_indices:
            run_layer_sample(
                layer,
                inputs,
                idx,
                dev,
                attention_mask,
                position_ids,
                position_embeddings,
                store=False,
            )
    finally:
        handle.remove()
    return captured


def _activation_scale_from_inputs(
    input_features: Sequence[torch.Tensor],
    dev: torch.device,
    group: Optional[Any],
    hidden_dim: int,
) -> torch.Tensor:
    sum_abs = torch.zeros(hidden_dim, device=dev, dtype=torch.float32)
    count = 0
    for feat in input_features:
        x = feat.to(dev)
        x = x.reshape(-1, x.shape[-1]).float()
        cur = x.abs().sum(dim=0)
        sum_abs += cur
        count += x.shape[0]

    count_tensor = torch.tensor([float(count)], device=dev, dtype=torch.float64)
    dist.all_reduce(sum_abs, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, group=group)
    if count_tensor.item() <= 0:
        raise RuntimeError("awq_dp: no activation samples were collected for scale search.")
    return sum_abs / count_tensor.to(dtype=sum_abs.dtype)


def _awq_alpha_candidates(args) -> list[float]:
    grid = max(int(args.awq_grid), 1)
    min_alpha = float(args.awq_min_alpha)
    max_alpha = float(args.awq_max_alpha)
    if grid == 1:
        return [max_alpha]
    # Match llm-awq's default grid: 0/20, 1/20, ..., 19/20.
    return [min_alpha + (max_alpha - min_alpha) * i / grid for i in range(grid)]


def _make_awq_scale(act_scale: torch.Tensor, alpha: float, eps: float) -> torch.Tensor:
    scale = act_scale.float().clamp_min(eps).pow(alpha).clamp_min(eps)
    return scale / (scale.max() * scale.min()).sqrt().clamp_min(eps)


def _set_scaled_quant_weight(module: torch.nn.Module, scale: torch.Tensor, args) -> None:
    linear = _unwrap_linear_module(module)
    original = linear.weight.data.float()
    scale_row = scale.to(device=linear.weight.device, dtype=original.dtype).unsqueeze(0)
    q_weight = _quantize_tensor_with_args(original * scale_row, args, mse=False)
    linear.weight.data = (q_weight / scale_row).to(linear.weight.dtype)


def _restore_weights(weight_cache: Sequence[tuple[torch.nn.Module, torch.Tensor]]) -> None:
    for linear, weight in weight_cache:
        linear.weight.data = weight


def _search_awq_scale(
    modules: Sequence[torch.nn.Module],
    inspect_module: torch.nn.Module,
    inspect_kwargs: dict[str, Any],
    input_features: Sequence[torch.Tensor],
    args,
    dev: torch.device,
    group: Optional[Any] = None,
) -> torch.Tensor:
    hidden_dim = _unwrap_linear_module(modules[0]).weight.shape[1]
    act_scale = _activation_scale_from_inputs(input_features, dev, group, hidden_dim)
    eps = float(args.awq_scale_eps)
    alpha_candidates = _awq_alpha_candidates(args)

    org_outputs = [
        _module_output(inspect_module, feat.to(dev), inspect_kwargs).detach().cpu()
        for feat in input_features
    ]

    weight_cache = [
        (_unwrap_linear_module(module), _unwrap_linear_module(module).weight.data.clone())
        for module in modules
    ]
    best_scale = torch.ones_like(act_scale)
    best_error = None
    try:
        for alpha in alpha_candidates:
            scale = _make_awq_scale(act_scale, alpha, eps)
            for module in modules:
                _set_scaled_quant_weight(module, scale, args)

            score = torch.zeros(2, device=dev, dtype=torch.float64)
            for feat, org_out in zip(input_features, org_outputs):
                out = _module_output(inspect_module, feat.to(dev), inspect_kwargs)
                diff = (out.float() - org_out.to(dev).float()).pow(2)
                score[0] += diff.sum(dtype=torch.float64)
                score[1] += diff.numel()
            dist.all_reduce(score, op=dist.ReduceOp.SUM, group=group)
            error = (score[0] / score[1].clamp_min(1)).item()
            if best_error is None or error < best_error:
                best_error = error
                best_scale = scale.detach().clone()
            _restore_weights(weight_cache)
    finally:
        _restore_weights(weight_cache)

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


def _apply_awq_fc_fc_scale(
    prev_module: torch.nn.Module,
    next_module: torch.nn.Module,
    scale: torch.Tensor,
) -> None:
    prev_linear = _unwrap_linear_module(prev_module)
    next_linear = _unwrap_linear_module(next_module)
    scale_prev = scale.to(device=prev_linear.weight.device, dtype=prev_linear.weight.dtype)
    prev_linear.weight.data[-scale_prev.numel():].div_(scale_prev.view(-1, 1))
    if prev_linear.bias is not None:
        prev_linear.bias.data[-scale_prev.numel():].div_(scale_prev)

    scale_next = scale.to(device=next_linear.weight.device, dtype=next_linear.weight.dtype)
    next_linear.weight.data.mul_(scale_next.view(1, -1))


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


def _linear_input_features(
    input_features: Sequence[torch.Tensor],
    dev: torch.device,
    max_tokens: int,
    hidden_dim: int,
) -> torch.Tensor:
    if not input_features:
        return torch.zeros((0, hidden_dim), device=dev, dtype=torch.float32)
    flat = torch.cat([feat.reshape(-1, feat.shape[-1]).to(dev) for feat in input_features], dim=0)
    if flat.shape[0] > max_tokens:
        step = max(flat.shape[0] // max_tokens, 1)
        flat = flat[::step][:max_tokens]
    return flat.float()


def _auto_clip_linear(
    module: torch.nn.Module,
    input_features: Sequence[torch.Tensor],
    args,
    dev: torch.device,
    group: Optional[Any],
) -> torch.Tensor:
    linear = _unwrap_linear_module(module)
    weight = linear.weight.data.float().to(dev)
    inputs = _linear_input_features(
        input_features,
        dev,
        max_tokens=int(getattr(args, "awq_clip_tokens", 512)),
        hidden_dim=weight.shape[1],
    )
    oc_batch_size = 256 if weight.shape[0] % 256 == 0 else 64
    if weight.shape[0] % oc_batch_size != 0:
        oc_batch_size = weight.shape[0]

    best_chunks = []
    n_grid = int(getattr(args, "awq_clip_grid", 20))
    max_shrink = float(getattr(args, "awq_clip_max_shrink", 0.5))
    for start in range(0, weight.shape[0], oc_batch_size):
        w = weight[start:start + oc_batch_size]
        org_max = w.abs().amax(dim=1, keepdim=True).clamp_min(float(args.awq_scale_eps))
        best_max = org_max.clone()
        best_err = torch.full((w.shape[0],), float("inf"), device=dev)
        org_out = inputs @ w.t()

        for i in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i / n_grid)
            cur_w = torch.clamp(w, -max_val, max_val)
            q_w = _quantize_tensor_with_args(cur_w, args, mse=False)
            cur_out = inputs @ q_w.t()
            err_stat = torch.zeros((2, w.shape[0]), device=dev, dtype=torch.float64)
            if inputs.numel() > 0:
                err_stat[0] = (cur_out - org_out).float().pow(2).sum(dim=0).double()
                err_stat[1].fill_(inputs.shape[0])
            dist.all_reduce(err_stat, op=dist.ReduceOp.SUM, group=group)
            err = err_stat[0] / err_stat[1].clamp_min(1)
            better = err < best_err
            if torch.any(better):
                best_err[better] = err[better]
                best_max[better] = max_val[better]
        best_chunks.append(best_max)

    best_max = torch.cat(best_chunks, dim=0)
    return best_max.to(device=linear.weight.device, dtype=linear.weight.dtype)


def _apply_awq_clip(
    module: torch.nn.Module,
    max_val: torch.Tensor,
) -> None:
    linear = _unwrap_linear_module(module)
    linear.weight.data = torch.clamp(linear.weight.data, -max_val, max_val)


def _auto_clip_layer_weights(
    layer: torch.nn.Module,
    full: dict[str, torch.nn.Module],
    inputs: torch.Tensor,
    input_indices: Sequence[int],
    args,
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    group: Optional[Any],
) -> None:
    if not args.w_clip:
        return
    for name, module in full.items():
        if any(skip in name for skip in ("q_proj", "k_proj", "query", "key", "Wqkv")):
            continue
        input_features = _capture_module_inputs(
            layer,
            module,
            inputs,
            input_indices,
            dev,
            attention_mask,
            position_ids,
            position_embeddings,
        )
        max_val = _auto_clip_linear(module, input_features, args, dev, group)
        _apply_awq_clip(module, max_val)
        del input_features
        memory_utils.cleanup_memory(False)


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
                mse=False,
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
        if shard_inps:
            assert state.inps_shard is not None
            group_inputs = state.inps_shard
            all_group_indices = list(range(len(local_js)))
        else:
            assert state.inps is not None
            group_inputs = state.inps
            all_group_indices = local_js
        search_limit = int(getattr(args, "awq_search_samples", 128))
        group_indices = [
            idx for idx, global_idx in zip(all_group_indices, local_js)
            if global_idx < search_limit
        ]

        awq_norms = analyzer.get_layernorms(layer)
        awq_input_groups = analyzer.get_perlayer_input_modules(layer)
        for task_idx, (norm_module, modules) in enumerate(zip(awq_norms, awq_input_groups)):
            if not modules:
                continue
            hook_module = modules[0]
            input_features = _capture_module_inputs(
                layer,
                hook_module,
                group_inputs,
                group_indices,
                dev,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
            )
            inspect_module = layer.self_attn if task_idx == 0 else layer.mlp
            inspect_kwargs = (
                {
                    "attention_mask": state.attention_mask,
                    "position_ids": state.position_ids,
                    "position_embeddings": state.position_embeddings,
                }
                if inspect_module is layer.self_attn
                else {}
            )
            scale = _search_awq_scale(
                modules,
                inspect_module,
                inspect_kwargs,
                input_features,
                args,
                dev,
                group,
            )
            _apply_awq_scale(norm_module, modules, scale)
            del input_features

        v_proj = getattr(layer.self_attn, "v_proj", None)
        o_proj = getattr(layer.self_attn, "o_proj", None)
        if v_proj is not None and o_proj is not None:
            v_linear = _unwrap_linear_module(v_proj)
            o_linear = _unwrap_linear_module(o_proj)
            if v_linear.weight.shape == o_linear.weight.shape:
                input_features = _capture_module_inputs(
                    layer,
                    o_proj,
                    group_inputs,
                    group_indices,
                    dev,
                    state.attention_mask,
                    state.position_ids,
                    state.position_embeddings,
                )
                scale = _search_awq_scale(
                    [o_proj],
                    o_proj,
                    {},
                    input_features,
                    args,
                    dev,
                    group,
                )
                _apply_awq_fc_fc_scale(v_proj, o_proj, scale)
                del input_features

        if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj") and hasattr(layer.mlp, "down_proj"):
            up_proj = layer.mlp.up_proj
            down_proj = layer.mlp.down_proj
            input_features = _capture_module_inputs(
                layer,
                down_proj,
                group_inputs,
                group_indices,
                dev,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
            )
            scale = _search_awq_scale(
                [down_proj],
                down_proj,
                {},
                input_features,
                args,
                dev,
                group,
            )
            _apply_awq_fc_fc_scale(up_proj, down_proj, scale)
            del input_features

        _auto_clip_layer_weights(
            layer,
            full,
            group_inputs,
            group_indices,
            args,
            dev,
            state.attention_mask,
            state.position_ids,
            state.position_embeddings,
            group,
        )
        if world_size > 1:
            dist_utils.broadcast_parameters(layer, src=0, group=group)

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
