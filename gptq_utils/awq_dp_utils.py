# coding=utf-8
"""AWQ (activation-aware weight quantization) with data-parallel calibration.

This module is aligned with the MIT reference implementation
(``mit-han-lab/llm-awq``: ``auto_scale.py`` + ``auto_clip.py``):

* Per-linear input features are collected with forward hooks over the local FP
  forward pass (one pass per rank), then used for three things:

  1. ``_search_awq_scale_block``: block-forward MSE scale search. The candidate
     scales follow the MIT formula
     ``scale = x_max.pow(ratio).clamp(1e-4)`` / ``(max*min).sqrt()`` and the
     objective is ``(block(x) - block_quant(x)).pow(2).mean()`` - exactly the
     same as ``_search_module_scale`` in ``auto_scale.py``. In the DP version
     the squared error and element count are summed across ranks, and every
     rank picks the globally best ratio.
  2. ``_apply_awq_scale`` / ``_apply_awq_fc_fc_scale``: applies the found
     scales to either a preceding ``LayerNorm``/``RMSNorm`` + grouped FCs, or a
     preceding FC + a following FC (``up_proj -> down_proj``), mirroring
     ``scale_ln_fcs`` / ``scale_fc_fc`` in the reference implementation. The
     stored per-linear inputs are divided by the scale so that auto-clip sees
     the post-scale activations.
  3. ``_auto_clip_layer``: per-row (and per-group) clip-max search over
     ``range(int(max_shrink * n_grid))`` shrinkage steps, aligned with
     ``auto_clip_layer``. DP ranks sum the per-row squared error and then every
     rank selects the same best clip-max.

After scale/clip are applied, the existing ReQuant + weight-quant +
broadcast + propagate pipeline runs unchanged.

Differences vs. the reference that are intentional:

* On Llama-3 / Qwen-3 GQA, ``v_proj.out != o_proj.in``, so the optional
  ``v_proj -> o_proj`` scale group is skipped (MIT's ``pre_quant`` does the
  same via its ``if v.weight.shape == o.weight.shape`` guard).
* The ``up_proj -> down_proj`` FC-FC scale group is gated by the flag
  ``awq_fc_fc_scale`` because combining it with Hadamard rotation is not
  supported in this repo's rotate pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from gptq_utils.dp_common import (
    capture_calibration_state,
    local_sample_indices,
    propagate_layer_inputs,
    run_layer_sample,
)
from requant import FPInputsCache, requant_layer
from utils import dist_utils, memory_utils, model_utils, quant_utils


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _get_submodule(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    cur: nn.Module = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _build_quantize_fn(args) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an MIT-style ``pseudo_quantize_tensor`` equivalent.

    Strictly matches ``auto_scale.py``:

        def w_quantize_func(p):
            return pseudo_quantize_tensor(p, n_bit=w_bit, **q_config).detach()

    where ``pseudo_quantize_tensor`` is pure per-channel (or per-group) min-max
    quantization with a zero point iff asymmetric. We therefore use the repo's
    ``WeightQuantizer`` with ``mse=False`` so the search is not polluted by an
    inner clip-ratio search - that is the job of ``auto_clip`` below.
    """

    def _quant(w: torch.Tensor) -> torch.Tensor:
        q = quant_utils.WeightQuantizer()
        q.configure(
            args.w_bits,
            perchannel=True,
            sym=not args.w_asym,
            mse=False,
            weight_groupsize=args.w_groupsize,
        )
        w32 = w.float()
        q.find_params(w32)
        q_w, _, _ = q.fake_quantize(w32)
        return q_w.to(dtype=w.dtype)

    return _quant


# ---------------------------------------------------------------------------
# Scale search (MIT-style block-forward MSE, data-parallel)
# ---------------------------------------------------------------------------


def _get_act_scale(x: torch.Tensor) -> torch.Tensor:
    # MIT: x.abs().view(-1, ci).mean(0). Sum-based so DP all_reduce is trivial.
    return x.abs().view(-1, x.shape[-1]).sum(dim=0)


def _global_x_max(
    x_local: torch.Tensor,
    group: Optional[Any],
    dev: torch.device,
) -> torch.Tensor:
    ci = x_local.shape[-1]
    local_sum = _get_act_scale(x_local.to(dev, dtype=torch.float32))
    local_cnt = torch.tensor(
        [float(x_local.reshape(-1, ci).shape[0])], dtype=torch.float64, device=dev
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM, group=group)
    return local_sum / local_cnt.to(dtype=local_sum.dtype).clamp_min(1.0)


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


@torch.no_grad()
def _search_awq_scale_block(
    block: nn.Module,
    linears2scale: Sequence[nn.Module],
    x_local: torch.Tensor,
    block_kwargs: Dict[str, Any],
    args,
    dev: torch.device,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """MIT ``_search_module_scale`` faithfully, with DP loss reduction.

    Strict port of ``auto_scale.py::_search_module_scale``:

        x = x.to(next(block.parameters()).device)
        org_out = block(x, **kwargs)
        x_max = get_act_scale(x)                 # mean(|x|) over (n, seq) tokens
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid            # {0, 1/20, ..., 19/20}
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            loss = (org_out - out).pow(2).mean()
            ...
            block.load_state_dict(org_sd)

    In the DP variant the per-grid squared error and element count are summed
    across ranks, so every rank selects the same globally-best ratio.
    """

    linears2scale = [_unwrap(m) for m in linears2scale]
    q_func = _build_quantize_fn(args)

    # MIT: ``x = x.to(next(block.parameters()).device)`` - no dtype cast,
    # ``x`` was cached in the model's native dtype by the forward hook.
    x_dev = x_local.to(dev)

    org_out = block(x_dev, **block_kwargs)
    if isinstance(org_out, tuple):
        org_out = org_out[0]
    org_out = org_out.detach()

    # MIT computes x_max in the model's native dtype (``get_act_scale`` is
    # dtype-preserving). We use fp32 only to keep the all_reduce numerics
    # stable, then cast back so all downstream scale arithmetic matches MIT.
    x_max = _global_x_max(x_local, group, dev).to(dtype=x_dev.dtype)

    # MIT-style: snapshot the entire block state_dict and restore it at the end
    # of every grid point via ``block.load_state_dict(org_sd)`` so the search
    # never mutates any internal buffer permanently (beyond the inner loop).
    org_sd = {k: v.detach().clone() for k, v in block.state_dict().items()}

    n_grid = max(int(args.awq_grid), 1)
    min_a = float(args.awq_min_alpha)
    max_a = float(args.awq_max_alpha)

    best_ratio = -1.0
    best_scales: Optional[torch.Tensor] = None
    best_loss: Optional[float] = None
    history: List[float] = []

    for i in range(n_grid):
        # MIT: ratio = i / n_grid, i in [0, n_grid) -> {0, 0.05, ..., 0.95}.
        # We keep the configurable min/max form but default to min=0, max=1.
        ratio = min_a + (max_a - min_a) * (i / n_grid)
        scales = x_max.pow(ratio).clamp(min=float(args.awq_scale_eps)).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        for fc in linears2scale:
            s = scales.to(device=fc.weight.device)
            fc.weight.mul_(s.view(1, -1))
            fc.weight.data = q_func(fc.weight.data) / s.view(1, -1)

        out = block(x_dev, **block_kwargs)
        if isinstance(out, tuple):
            out = out[0]

        local_sq = (org_out - out).float().pow(2).sum()
        local_nel = torch.tensor(
            [float(out.numel())], dtype=torch.float64, device=dev
        )
        buf = torch.stack([local_sq.to(torch.float64), local_nel.squeeze()])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
        global_loss = (buf[0] / buf[1].clamp_min(1.0)).item()
        history.append(global_loss)

        if best_loss is None or global_loss < best_loss:
            best_loss = global_loss
            best_ratio = ratio
            best_scales = scales.clone()

        # MIT: ``block.load_state_dict(org_sd)`` - fully restore every buffer /
        # parameter that the inner loop may have touched.
        block.load_state_dict(org_sd)

    if best_scales is None:
        raise RuntimeError(f"awq_dp: scale search produced no candidate (history={history})")

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()


# ---------------------------------------------------------------------------
# Scale application (aligned with scale_ln_fcs / scale_fc_fc)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _apply_scale_ln_fcs(
    ln: nn.Module,
    fcs: Sequence[nn.Module],
    scales: torch.Tensor,
    input_feats: Sequence[Optional[torch.Tensor]] = (),
) -> None:
    scales = scales.to(device=ln.weight.device, dtype=ln.weight.dtype)
    ln.weight.data.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data.div_(scales)
    for fc in fcs:
        lin = _unwrap(fc)
        s = scales.to(device=lin.weight.device, dtype=lin.weight.dtype)
        lin.weight.data.mul_(s.view(1, -1))
    for inp in input_feats:
        if inp is None:
            continue
        s = scales.to(device=inp.device, dtype=inp.dtype)
        inp.div_(s.view(1, -1))


@torch.no_grad()
def _apply_scale_fc_fc(
    fc1: nn.Module,
    fc2: nn.Module,
    scales: torch.Tensor,
    input_feats: Sequence[Optional[torch.Tensor]] = (),
) -> None:
    fc1 = _unwrap(fc1)
    fc2 = _unwrap(fc2)
    s1 = scales.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    fc1.weight.data[-s1.size(0) :].div_(s1.view(-1, 1))
    if getattr(fc1, "bias", None) is not None:
        fc1.bias.data.div_(s1.view(-1))
    s2 = scales.to(device=fc2.weight.device, dtype=fc2.weight.dtype)
    fc2.weight.data.mul_(s2.view(1, -1))
    for inp in input_feats:
        if inp is None:
            continue
        s = scales.to(device=inp.device, dtype=inp.dtype)
        inp.div_(s.view(1, -1))


# ---------------------------------------------------------------------------
# Auto-clip (MIT auto_clip_layer with DP all_reduce on err)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _auto_clip_layer(
    weight: torch.Tensor,
    input_feat_local: torch.Tensor,
    args,
    dev: torch.device,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """Per-row best clip-max, aligned with MIT ``auto_clip.py::auto_clip_layer``.

    Strict port:

        group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)
        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
        assert w.shape[0] % oc_batch_size == 0
        for i_b in ...:
            org_max_val = w.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            org_out = (input_feat * w).sum(dim=-1)
            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                cur_w = torch.clamp(w, -max_val, max_val)
                q_w = pseudo_quantize_tensor(cur_w, n_bit, **q_config)
                cur_out = (input_feat * q_w).sum(dim=-1)
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                best_max_val[err < min_errs] = max_val[err < min_errs]

    For DP, the per-row squared error is summed across ranks at each shrink
    step, so every rank picks the same globally-best ``max_val``. Dtypes
    follow MIT (model native, typically fp16/bf16) - no float32 upcast.
    """

    assert weight.dim() == 2
    group_size = int(args.w_groupsize) if args.w_groupsize > 0 else weight.shape[1]
    n_grid = max(int(args.awq_clip_n_grid), 1)
    max_shrink = float(args.awq_clip_max_shrink)
    n_sample_token = max(int(args.awq_clip_n_sample_token), 1)
    q_func = _build_quantize_fn(args)

    # Match MIT: keep the model's native dtype, no fp32 upcast.
    x = input_feat_local.to(dev).view(-1, input_feat_local.shape[-1])
    x = x.reshape(1, x.shape[0], -1, group_size)
    if x.shape[1] > n_sample_token:
        step = x.shape[1] // n_sample_token
        x = x[:, 0::step]

    w_all = weight.to(dev).reshape(weight.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w_all.shape[0] % 256 == 0 else 64
    assert w_all.shape[0] % oc_batch_size == 0, (
        f"awq auto_clip: weight.shape[0]={w_all.shape[0]} is not divisible by "
        f"oc_batch_size={oc_batch_size} (same constraint as MIT auto_clip_layer)."
    )
    best_max_list: List[torch.Tensor] = []

    for i_b in range(w_all.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_max_val = w.abs().amax(dim=-1, keepdim=True)
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        org_out = (x * w).sum(dim=-1)  # [oc_b, n_token, n_group]

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = q_func(cur_w.reshape(cur_w.shape[0], -1)).reshape(cur_w.shape)
            cur_out = (x * q_w).sum(dim=-1)

            # MIT uses .mean(dim=1); with DP we use sum(dim=1) + all_reduce(SUM)
            # so every rank selects the same globally-best max_val (mean and
            # sum give the same ordering since n_token is constant).
            err = (cur_out - org_out).pow(2).sum(dim=1, keepdim=True).view(min_errs.shape)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(err, op=dist.ReduceOp.SUM, group=group)

            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]

            del cur_w, cur_out, q_w

        best_max_list.append(best_max_val)

    best_max_val = torch.cat(best_max_list, dim=0)
    return best_max_val.squeeze(1)  # [co, n_group]


@torch.no_grad()
def _apply_clip(weight_module: nn.Module, max_val: torch.Tensor) -> None:
    """Strict port of MIT ``apply_clip`` (auto_clip.py)."""
    lin = _unwrap(weight_module)
    max_val = max_val.to(device=lin.weight.device, dtype=lin.weight.dtype)
    org_shape = lin.weight.shape
    # max_val comes from _auto_clip_layer with shape [co, n_group, 1]; the
    # weight is reshaped to [co, n_group, group_size] and clamp broadcasts on
    # the last dim - exactly as in MIT's apply_clip.
    lin.weight.data = lin.weight.data.reshape(*max_val.shape[:2], -1)
    lin.weight.data = torch.clamp(lin.weight.data, -max_val, max_val)
    lin.weight.data = lin.weight.data.reshape(org_shape)


# ---------------------------------------------------------------------------
# Per-linear input collection (local FP forward)
# ---------------------------------------------------------------------------


def _register_linear_input_hook(
    module: nn.Module, storage: Dict[str, List[torch.Tensor]], name: str
):
    def _hook(_m, inp, _out):
        # Keep the original [batch, seq, hidden] shape so LlamaAttention /
        # Qwen2Attention can consume it in _search_awq_scale_block. Keep the
        # model's native dtype (usually fp16/bf16) to halve CPU memory; upstream
        # consumers promote to fp32 as needed.
        storage[name].append(inp[0].detach().to("cpu", non_blocking=True))

    return module.register_forward_hook(_hook)


def _collect_linear_inputs(
    layer: nn.Module,
    hook_names: Sequence[str],
    inputs_tensor: torch.Tensor,
    indices: Iterable[int],
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
) -> Dict[str, torch.Tensor]:
    storage: Dict[str, List[torch.Tensor]] = {n: [] for n in hook_names}
    handles = []
    for name in hook_names:
        sub = _get_submodule(layer, name)
        if sub is None:
            continue
        handles.append(_register_linear_input_hook(sub, storage, name))
    try:
        for idx in indices:
            _ = run_layer_sample(
                layer,
                inputs_tensor,
                idx,
                dev,
                attention_mask,
                position_ids,
                position_embeddings,
                store=False,
            )
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, torch.Tensor] = {}
    for name, lst in storage.items():
        if not lst:
            continue
        out[name] = torch.cat(lst, dim=0)
    return out


# ---------------------------------------------------------------------------
# Weight quantization (unchanged from previous implementation)
# ---------------------------------------------------------------------------


def _quantize_layer_weights(
    layer_idx: int,
    full: Dict[str, nn.Module],
    sequential_names: Sequence[Sequence[str]],
    args,
) -> Dict[str, quant_utils.WeightQuantizer]:
    quantizers: Dict[str, quant_utils.WeightQuantizer] = {}
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


# ---------------------------------------------------------------------------
# Per-layer AWQ (scale + clip) driver
# ---------------------------------------------------------------------------


def _select_linear_hook_names(analyzer: model_utils.ModelAnalyzer) -> List[str]:
    if analyzer._is_qwen_dense() or analyzer._is_llama():
        return [
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ]
    raise NotImplementedError("AWQ DP currently supports Llama / Qwen-dense only.")


def _awq_scale_and_clip_layer(
    layer: nn.Module,
    layer_idx: int,
    analyzer: model_utils.ModelAnalyzer,
    input_feat: Dict[str, torch.Tensor],
    block_kwargs: Dict[str, Any],
    args,
    dev: torch.device,
    group: Optional[Any],
) -> None:
    """Search + apply AWQ scales and (optional) auto-clip on one decoder layer."""

    q_proj = _get_submodule(layer, "self_attn.q_proj")
    k_proj = _get_submodule(layer, "self_attn.k_proj")
    v_proj = _get_submodule(layer, "self_attn.v_proj")
    o_proj = _get_submodule(layer, "self_attn.o_proj")
    gate_proj = _get_submodule(layer, "mlp.gate_proj")
    up_proj = _get_submodule(layer, "mlp.up_proj")
    down_proj = _get_submodule(layer, "mlp.down_proj")
    input_ln = _get_submodule(layer, "input_layernorm")
    post_ln = _get_submodule(layer, "post_attention_layernorm")
    self_attn = _get_submodule(layer, "self_attn")
    mlp = _get_submodule(layer, "mlp")

    # ---- Group 1: input_layernorm -> [q, k, v], inspect=self_attn ----
    if q_proj is not None and "self_attn.q_proj" in input_feat:
        qkv = [m for m in (q_proj, k_proj, v_proj) if m is not None]
        scale = _search_awq_scale_block(
            self_attn, qkv, input_feat["self_attn.q_proj"], block_kwargs, args, dev, group=group
        )
        _apply_scale_ln_fcs(
            input_ln, qkv, scale, input_feats=[input_feat.get("self_attn.q_proj")]
        )

    # ---- Group 2 (optional): v_proj -> o_proj (skipped on GQA shape mismatch) ----
    if (
        o_proj is not None
        and v_proj is not None
        and v_proj.weight.shape == o_proj.weight.shape
        and "self_attn.o_proj" in input_feat
    ):
        scale = _search_awq_scale_block(
            o_proj, [o_proj], input_feat["self_attn.o_proj"], {}, args, dev, group=group
        )
        _apply_scale_fc_fc(
            v_proj, o_proj, scale, input_feats=[input_feat.get("self_attn.o_proj")]
        )

    # ---- Group 3: post_attention_layernorm -> [gate, up], inspect=mlp ----
    if gate_proj is not None and "mlp.gate_proj" in input_feat:
        fc_pack = [m for m in (gate_proj, up_proj) if m is not None]
        scale = _search_awq_scale_block(
            mlp, fc_pack, input_feat["mlp.gate_proj"], {}, args, dev, group=group
        )
        _apply_scale_ln_fcs(
            post_ln, fc_pack, scale, input_feats=[input_feat.get("mlp.gate_proj")]
        )

    # ---- Group 4 (optional): up_proj -> down_proj ----
    if (
        bool(getattr(args, "awq_fc_fc_scale", 0))
        and up_proj is not None
        and down_proj is not None
        and "mlp.down_proj" in input_feat
    ):
        scale = _search_awq_scale_block(
            down_proj, [down_proj], input_feat["mlp.down_proj"], {}, args, dev, group=group
        )
        _apply_scale_fc_fc(
            up_proj, down_proj, scale, input_feats=[input_feat.get("mlp.down_proj")]
        )

    # ---- Auto clip (applied to the scaled weights / inputs) ----
    # Aligned with MIT ``auto_clip_block``: clip every Linear except ``q_*`` /
    # ``k_*`` (which are hard to clip precisely due to the QK bmm). ``v_proj``
    # is included; it shares the input tensor with ``q_proj`` because both come
    # from ``input_layernorm.output`` and the Group-1 scale application already
    # divided that shared input by ``s`` in-place.
    if bool(getattr(args, "awq_auto_clip", 1)):
        clip_targets = {
            "self_attn.v_proj": v_proj,
            "self_attn.o_proj": o_proj,
            "mlp.gate_proj": gate_proj,
            "mlp.up_proj": up_proj,
            "mlp.down_proj": down_proj,
        }
        # v_proj / q_proj share input_layernorm output; gate_proj / up_proj
        # share post_attention_layernorm output. Reuse the already-scaled
        # input_feat tensors to avoid an extra FP forward pass.
        clip_inputs = {
            "self_attn.v_proj": input_feat.get("self_attn.q_proj"),
            "self_attn.o_proj": input_feat.get("self_attn.o_proj"),
            "mlp.gate_proj": input_feat.get("mlp.gate_proj"),
            "mlp.up_proj": input_feat.get("mlp.gate_proj"),
            "mlp.down_proj": input_feat.get("mlp.down_proj"),
        }
        for name, module in clip_targets.items():
            if module is None:
                continue
            inp = clip_inputs.get(name)
            if inp is None or inp.numel() == 0:
                continue
            lin = _unwrap(module)
            max_val = _auto_clip_layer(
                lin.weight.data, inp, args, dev, group=group
            )
            _apply_clip(module, max_val)
            del max_val


# ---------------------------------------------------------------------------
# Main DP driver
# ---------------------------------------------------------------------------


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

    # ---- Safety: auto_clip + w_clip double-search is NOT what MIT does ----
    # MIT llm-awq's pseudo_quantize_tensor is pure min-max (no mse clip search).
    # Clip is fully delegated to auto_clip. If the user happens to pass --w_clip
    # together with awq_auto_clip=1 we force w_clip off for this run so the
    # final weight-quant step does not shrink clip again on top of auto_clip.
    if bool(getattr(args, "awq_auto_clip", 1)) and bool(getattr(args, "w_clip", False)):
        if rank == 0:
            logging.warning(
                "awq_dp: --w_clip and awq_auto_clip=1 conflict (double clip search). "
                "Forcing args.w_clip=False so auto_clip fully drives the clip decision "
                "(MIT-aligned behaviour)."
            )
        args.w_clip = False

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

    block_kwargs_for_attn: Dict[str, Any] = {
        "attention_mask": state.attention_mask,
        "position_ids": state.position_ids,
        "position_embeddings": state.position_embeddings,
    }

    hook_names = _select_linear_hook_names(analyzer)

    quantizers: Dict[str, quant_utils.WeightQuantizer] = {}
    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (AWQ DP)",
        disable=(rank != 0),
    )
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential) if use_requant else None

    # ------------------------------------------------------------------
    # Decoupled AWQ calibration sample count (matches MIT llm-awq default).
    # ``args.nsamples`` feeds GPTQ/GPTAQ Hessian + ReQuant; AWQ's scale
    # search and auto_clip use a (usually smaller) independent subset taken
    # from the already-captured first-block-input buffer.
    # ------------------------------------------------------------------
    awq_nsamples_cfg = int(getattr(args, "awq_nsamples", args.nsamples))
    awq_nsamples = max(1, min(awq_nsamples_cfg, int(args.nsamples)))
    if rank == 0 and awq_nsamples != awq_nsamples_cfg:
        logging.info(
            "-----AWQ DP: clamping awq_nsamples from %d to %d (bounded by nsamples)-----",
            awq_nsamples_cfg,
            awq_nsamples,
        )
    # Per-rank local indices for the AWQ subset (global split with the same
    # ``local_sample_indices`` policy used for ``args.nsamples`` so that when
    # ``awq_nsamples == args.nsamples`` we fall back to the original behaviour).
    awq_local_js = local_sample_indices(rank, world_size, awq_nsamples)
    if rank == 0:
        logging.info(
            "-----AWQ DP: awq_nsamples=%d (requant/Hessian nsamples=%d), per-rank local=%d-----",
            awq_nsamples,
            args.nsamples,
            len(awq_local_js),
        )

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

        # Pick source / indices for AWQ local FP forward:
        #   * shard_inps=False: state.inps has all args.nsamples globally, so
        #     take the AWQ-specific global indices awq_local_js directly.
        #   * shard_inps=True : state.inps_shard only has this rank's slice of
        #     args.nsamples (len == len(local_js)); take its first len(awq_local_js)
        #     entries, which equals the rank's AWQ sub-slice because
        #     ``local_sample_indices`` assigns contiguous ranges per rank.
        if shard_inps:
            awq_cur_inputs = state.inps_shard
            awq_cur_indices = list(range(min(len(awq_local_js), len(local_js))))
        else:
            awq_cur_inputs = state.inps
            awq_cur_indices = awq_local_js
        assert awq_cur_inputs is not None

        # --- Step A: local FP forward -> per-linear inputs (model native dtype) ---
        input_feat = _collect_linear_inputs(
            layer,
            hook_names,
            awq_cur_inputs,
            awq_cur_indices,
            dev,
            state.attention_mask,
            state.position_ids,
            state.position_embeddings,
        )

        # --- Step B / C: AWQ scale search + apply + auto_clip ---
        _awq_scale_and_clip_layer(
            layer,
            i,
            analyzer,
            input_feat,
            block_kwargs_for_attn,
            args,
            dev,
            group,
        )
        del input_feat
        memory_utils.cleanup_memory(False)

        # --- Save fp_weights (post scale/clip) for ReQuant ---
        fp_weights: Dict[str, torch.Tensor] = {}
        for names in sequential:
            for name in names:
                module = full.get(name, full.get(name + ".module", None))
                if module is not None and "lm_head" not in name:
                    fp_weights[f"model.layers.{i}.{name}"] = (
                        _unwrap(module).weight.data.float().clone()
                    )

        # --- Step D: FP forward with FPInputsCache hooks (for ReQuant) ---
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

        # --- Step E: enable A-quant, quantize weights, ReQuant, broadcast ---
        quant_utils.enable_act_quant(layer, bits_config)
        quantizers.update(_quantize_layer_weights(i, full, sequential, args))

        if use_requant:
            if shard_inps:
                assert state.inps_shard is not None
                rq_inps = state.inps_shard
                rq_indices = list(range(len(local_js)))
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
