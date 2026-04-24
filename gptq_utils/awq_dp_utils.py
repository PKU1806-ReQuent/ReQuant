# coding=utf-8
"""AWQ with data-parallel calibration (multi-process, multi-GPU).

Ported to match the reference implementation at
``mit-han-lab/llm-awq/awq/quantize/{auto_scale,auto_clip,pre_quant}.py`` while
preserving the layer-wise sequential PTQ contract used by the other DP paths
in this repo.

Per decoder layer, all ranks run in lockstep:

1. A single forward on each rank's local activation shard caches every
   Linear's input on CPU (``input_feat[name]``). This mirrors llm-awq's
   ``cache_input_hook`` and is the sole data source for both the scale
   search and the optional weight-clip search.
2. Iterate the AWQ scaling groups. For Llama/Qwen-dense layers there are
   four groups, matching ``auto_scale.auto_scale_block``:

     - ``input_layernorm -> [q_proj, k_proj, v_proj]``  (LN->FCs)
     - ``v_proj        -> [o_proj]``                   (FC->FC, when v/o
       weight shapes match; skipped for GQA)
     - ``post_attention_layernorm -> [gate_proj, up_proj]`` (LN->FCs)
     - ``up_proj       -> [down_proj]``                (FC->FC)

   For every group we:

     a) all_reduce the local ``sum(|x|)`` and token count to get the global
        per-channel activation scale ``x_max``;
     b) grid search ``alpha`` in ``[awq_min_alpha, awq_max_alpha]``. The
        candidate is ``s = x_max.pow(alpha)`` normalised by
        ``sqrt(s.max() * s.min())`` (llm-awq form). The score is the
        **block-output MSE** of ``module2inspect`` with ``numerator`` and
        ``denominator`` all_reduced across ranks, so every rank elects the
        same ``best_alpha``;
     c) apply the winning scale to the previous op (LN or FC) and the
        grouped linears, then divide the cached activations of the scaled
        linears by ``s`` (so the clip step sees post-scale inputs).

3. If ``args.w_clip`` is set, run per-linear ``auto_clip`` (DP-aware): grid
   search ``max_val`` minimising the per-``(oc, group)`` token-MSE using the
   cached activations; q/k projections are skipped (llm-awq convention).

4. Fake-quantize layer weights deterministically on every rank (same cached
   activations + same weights => same scale/zero). Rank 0 then broadcasts
   the parameters for numerical safety.

5. Propagate hidden states sample-by-sample to produce the next layer's
   input, same as before.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from gptq_utils.gptaq_dp_utils import _local_sample_indices, _move_attn_struct_to_device
from utils import dist_utils, memory_utils, model_utils, quant_utils


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _unwrap_linear(module: nn.Module) -> nn.Module:
    """Return the inner ``nn.Linear`` when ``module`` is wrapped by
    ``ActQuantWrapper`` (which exposes the real Linear as ``.module``)."""
    return getattr(module, "module", module)


def _get_child(root: nn.Module, dotted: str) -> nn.Module:
    out = root
    for p in dotted.split("."):
        out = getattr(out, p)
    return out


def _configured_weight_quantizer(args) -> quant_utils.WeightQuantizer:
    """WeightQuantizer configured for AWQ's fake-quantize calls.

    ``mse=False`` because (a) the search loop must be cheap, and (b) when
    ``args.w_clip`` is set the dedicated ``auto_clip`` phase already performs
    a stronger output-error clipping search.
    """
    q = quant_utils.WeightQuantizer()
    q.configure(
        args.w_bits,
        perchannel=True,
        sym=not args.w_asym,
        mse=False,
        weight_groupsize=args.w_groupsize,
    )
    return q


# ---------------------------------------------------------------------------
# AWQ scaling group specification
# ---------------------------------------------------------------------------


def _awq_groups_for_layer(
    analyzer: model_utils.ModelAnalyzer, layer: nn.Module
) -> List[Dict[str, Any]]:
    """Return ordered AWQ scaling group specs.

    Each spec has:
        prev_op:         dotted name relative to ``layer``
        prev_kind:       ``"ln"`` | ``"fc"``
        linear_names:    dotted names of the linears to scale (input side)
        inspect:         dotted name of the module fed to the output-MSE
                         objective (``module2inspect`` in llm-awq)
        hook:            dotted name of the linear whose hooked input serves
                         as the module-under-inspect's input
        use_layer_kwargs: whether to forward ``layer_kwargs`` (attention
                          mask/position ids/embeddings) to ``inspect``.
    """
    if not (analyzer._is_qwen_dense() or analyzer._is_llama()):
        raise NotImplementedError(
            "awq_dp: AWQ scaling groups are only defined for Llama/Qwen-dense "
            "decoder layers; extend _awq_groups_for_layer for this arch."
        )

    groups: List[Dict[str, Any]] = [
        dict(
            prev_op="input_layernorm", prev_kind="ln",
            linear_names=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            inspect="self_attn", hook="self_attn.q_proj",
            use_layer_kwargs=True,
        ),
    ]

    # v_proj -> o_proj only when weight shapes match (skipped for GQA).
    v_lin = _unwrap_linear(_get_child(layer, "self_attn.v_proj"))
    o_lin = _unwrap_linear(_get_child(layer, "self_attn.o_proj"))
    if v_lin.weight.shape == o_lin.weight.shape:
        groups.append(
            dict(
                prev_op="self_attn.v_proj", prev_kind="fc",
                linear_names=["self_attn.o_proj"],
                inspect="self_attn.o_proj", hook="self_attn.o_proj",
                use_layer_kwargs=False,
            )
        )

    groups.extend([
        dict(
            prev_op="post_attention_layernorm", prev_kind="ln",
            linear_names=["mlp.gate_proj", "mlp.up_proj"],
            inspect="mlp", hook="mlp.gate_proj",
            use_layer_kwargs=False,
        ),
        dict(
            prev_op="mlp.up_proj", prev_kind="fc",
            linear_names=["mlp.down_proj"],
            inspect="mlp.down_proj", hook="mlp.down_proj",
            use_layer_kwargs=False,
        ),
    ])
    return groups


# ---------------------------------------------------------------------------
# Per-linear activation cache (single layer forward with hooks)
# ---------------------------------------------------------------------------


def _cache_linear_inputs(
    layer: nn.Module,
    linear_rel_names: Sequence[str],
    inps_source: torch.Tensor,
    local_indices: Sequence[int],
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
) -> Dict[str, torch.Tensor]:
    """Run layer forward on local samples, capture each Linear's input.

    Returns ``{rel_name: tensor [N_loc, seq, ci] on CPU}``. Dtype matches the
    activation dtype (typically bf16/fp16), i.e. no upcast, to keep CPU
    memory bounded.
    """
    feat: Dict[str, List[torch.Tensor]] = {n: [] for n in linear_rel_names}
    handles = []
    for n in linear_rel_names:
        lin = _unwrap_linear(_get_child(layer, n))

        def _hook(_m, inp, _out, name=n):
            x = inp[0].detach()
            # Expect [B=1, S, C]; if different, normalise to 3D.
            if x.dim() == 2:
                x = x.unsqueeze(0)
            feat[name].append(x.cpu())

        handles.append(lin.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            for idx in local_indices:
                _ = layer(
                    inps_source[idx].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, torch.Tensor] = {}
    for n in linear_rel_names:
        if not feat[n]:
            # No local samples: reserve an empty tensor so downstream code
            # can still operate uniformly.
            lin = _unwrap_linear(_get_child(layer, n))
            ci = lin.weight.shape[1]
            out[n] = torch.zeros((0, 0, ci), dtype=lin.weight.dtype, device="cpu")
        else:
            out[n] = torch.cat(feat[n], dim=0)  # [N_loc, S, C]
    return out


# ---------------------------------------------------------------------------
# Per-group scale search (block-output MSE, DP-aware)
# ---------------------------------------------------------------------------


def _all_reduce_sum(t: torch.Tensor, group: Optional[Any]) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t


def _search_group_scale_dp(
    layer: nn.Module,
    group: Dict[str, Any],
    input_feat_3d: torch.Tensor,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    args,
    dev: torch.device,
    dp_group: Optional[Any] = None,
) -> torch.Tensor:
    """Grid search the AWQ scale for a single group.

    ``input_feat_3d`` is ``[N_loc, S, ci]`` on CPU: the cached input to
    ``group["hook"]``. Returns the winning ``scales`` tensor of shape
    ``[ci]`` on ``dev`` (identical across ranks).
    """
    module2inspect = _get_child(layer, group["inspect"])
    linears2scale = [
        _unwrap_linear(_get_child(layer, n)) for n in group["linear_names"]
    ]
    block_kwargs: Dict[str, Any] = {}
    if group["use_layer_kwargs"]:
        block_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    n_loc = input_feat_3d.shape[0]
    ci = input_feat_3d.shape[-1] if input_feat_3d.numel() > 0 else linears2scale[0].weight.shape[1]

    # ---- global per-channel activation scale ---------------------------
    # Do not materialise ``[N*S, ci]`` on GPU at once (OOM on 8B + long ctx).
    if input_feat_3d.numel() > 0:
        local_sum = torch.zeros(ci, device=dev, dtype=torch.float32)
        token_cnt = 0
        for i in range(n_loc):
            chunk = input_feat_3d[i].to(dev, non_blocking=True)
            flat = chunk.reshape(-1, ci).float()
            local_sum += flat.abs().sum(dim=0)
            token_cnt += flat.shape[0]
        local_cnt = torch.tensor(
            [float(token_cnt)], device=dev, dtype=torch.float64
        )
    else:
        local_sum = torch.zeros(ci, device=dev, dtype=torch.float32)
        local_cnt = torch.zeros(1, device=dev, dtype=torch.float64)
    _all_reduce_sum(local_sum, dp_group)
    _all_reduce_sum(local_cnt, dp_group)
    if float(local_cnt.item()) <= 0:
        raise RuntimeError(
            "awq_dp: activation cache is empty across all ranks for group "
            f"prev_op={group['prev_op']!r}."
        )
    x_max = (local_sum / local_cnt.to(local_sum.dtype)).clamp_min(
        float(args.awq_scale_eps)
    )

    # ---- reference outputs (no scaling applied) ------------------------
    inspect_params_iter = iter(module2inspect.parameters())
    inspect_dev = dev
    try:
        inspect_dev = next(inspect_params_iter).device
    except StopIteration:
        pass
    inspect_dtype = linears2scale[0].weight.dtype

    org_outs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(n_loc):
            x = input_feat_3d[i].unsqueeze(0).to(device=inspect_dev, dtype=inspect_dtype)
            out = module2inspect(x, **block_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            org_outs.append(out.detach().float().cpu())

    # ---- alpha grid ----------------------------------------------------
    if args.awq_grid <= 1:
        alphas = [float(args.awq_max_alpha)]
    else:
        alphas = torch.linspace(
            float(args.awq_min_alpha),
            float(args.awq_max_alpha),
            steps=int(args.awq_grid),
        ).tolist()

    eps = float(args.awq_scale_eps)
    saved_weights = [(fc, fc.weight.data.detach().clone()) for fc in linears2scale]
    best_score: Optional[float] = None
    best_scales: Optional[torch.Tensor] = None

    try:
        for alpha in alphas:
            scales = x_max.pow(alpha).clamp_min(eps)
            denom = (scales.max() * scales.min()).sqrt().clamp_min(eps)
            scales = (scales / denom).float()

            # Apply candidate scaling to each linear (quantize then unscale).
            scale_row = scales.view(1, -1)
            for fc, orig_w in saved_weights:
                orig_dev_w = orig_w.to(dev).float()
                scaled = orig_dev_w * scale_row
                wq = _configured_weight_quantizer(args)
                wq.find_params(scaled)
                q_scaled, _, _ = wq.fake_quantize(scaled)
                q_w = q_scaled.float() / scale_row
                fc.weight.data.copy_(q_w.to(fc.weight.data.dtype))

            # Forward on local samples and accumulate errors.
            local_err = torch.zeros((), device=dev, dtype=torch.float64)
            local_cnt_f = torch.zeros((), device=dev, dtype=torch.float64)
            with torch.no_grad():
                for i in range(n_loc):
                    x = input_feat_3d[i].unsqueeze(0).to(
                        device=inspect_dev, dtype=inspect_dtype
                    )
                    out = module2inspect(x, **block_kwargs)
                    if isinstance(out, tuple):
                        out = out[0]
                    org = org_outs[i].to(device=dev)
                    diff = (out.float().to(dev) - org).pow(2)
                    local_err += diff.sum().double()
                    local_cnt_f += float(diff.numel())
            _all_reduce_sum(local_err, dp_group)
            _all_reduce_sum(local_cnt_f, dp_group)
            score = float((local_err / local_cnt_f.clamp_min(1.0)).item())

            # Restore original weights.
            for fc, orig_w in saved_weights:
                fc.weight.data.copy_(orig_w)

            if (best_score is None) or (score < best_score):
                best_score = score
                best_scales = scales.detach().clone()
    finally:
        # Defensive restore in case of exception.
        for fc, orig_w in saved_weights:
            if not torch.equal(fc.weight.data, orig_w):
                fc.weight.data.copy_(orig_w)

    assert best_scales is not None
    assert not torch.isnan(best_scales).any(), "awq_dp: NaN in best_scales"
    del org_outs
    memory_utils.cleanup_memory(False)
    return best_scales


# ---------------------------------------------------------------------------
# Apply the searched scale
# ---------------------------------------------------------------------------


def _apply_scale_ln_fcs(
    ln: nn.Module, fcs: Sequence[nn.Module], scales: torch.Tensor
) -> None:
    s = scales.to(device=ln.weight.device, dtype=ln.weight.dtype)
    ln.weight.data.div_(s)
    if getattr(ln, "bias", None) is not None:
        ln.bias.data.div_(s)
    for fc in fcs:
        s_fc = s.to(device=fc.weight.device, dtype=fc.weight.dtype)
        fc.weight.data.mul_(s_fc.view(1, -1))


def _apply_scale_fc_fc(
    fc1: nn.Linear, fcs2: Sequence[nn.Linear], scales: torch.Tensor
) -> None:
    """Row-scale the tail of ``fc1`` and column-scale each ``fc2``.

    Mirrors ``llm-awq/auto_scale.scale_fc_fc``: divides the last ``|s|`` rows
    of ``fc1.weight`` (and the matching slice of its bias), then multiplies
    every ``fc2``'s columns by ``s``. This supports GQA layouts where
    ``fc1.out_features`` can exceed ``scales.size(0)``.
    """
    assert len(fcs2) >= 1
    s = scales.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    sN = s.size(0)
    fc1.weight.data[-sN:].div_(s.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.data[-sN:].div_(s.view(-1))
    for fc2 in fcs2:
        s2 = scales.to(device=fc2.weight.device, dtype=fc2.weight.dtype)
        fc2.weight.data.mul_(s2.view(1, -1))


def _apply_group_scale(
    layer: nn.Module,
    group: Dict[str, Any],
    scales: torch.Tensor,
    input_feat_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    prev_op = _get_child(layer, group["prev_op"])
    if group["prev_kind"] == "fc":
        prev_op = _unwrap_linear(prev_op)
    linears = [_unwrap_linear(_get_child(layer, n)) for n in group["linear_names"]]

    if group["prev_kind"] == "ln":
        _apply_scale_ln_fcs(prev_op, linears, scales)
    elif group["prev_kind"] == "fc":
        _apply_scale_fc_fc(prev_op, linears, scales)
    else:
        raise NotImplementedError(group["prev_kind"])

    if input_feat_dict is not None:
        scales_cpu = scales.detach().cpu().float().clamp_min(1e-8)
        for n in group["linear_names"]:
            feat = input_feat_dict.get(n)
            if feat is None or feat.numel() == 0:
                continue
            inv = (1.0 / scales_cpu).to(feat.dtype)
            feat.mul_(inv.view(1, 1, -1))


# ---------------------------------------------------------------------------
# Per-linear auto-clip (DP-aware)
# ---------------------------------------------------------------------------


def _auto_clip_linear_dp(
    lin: nn.Linear,
    input_feat_3d: torch.Tensor,
    args,
    dev: torch.device,
    dp_group: Optional[Any] = None,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
) -> None:
    """Clamp ``lin.weight`` to the best per-(oc, group) ``max_val``.

    Ported from ``llm-awq/auto_clip.auto_clip_layer``; the error is summed
    across local tokens on each rank and all_reduced before comparison so
    every rank picks the same clip.
    """
    if input_feat_3d.numel() == 0:
        # Still need to participate in all_reduces to keep ranks in sync.
        n_local_tok = 0
    else:
        n_local_tok = input_feat_3d.shape[0] * input_feat_3d.shape[1]

    w_full = lin.weight.data.detach().to(dev).float()
    assert w_full.dim() == 2
    co, ci = w_full.shape
    group_size = args.w_groupsize if args.w_groupsize > 0 else ci
    if ci % group_size != 0:
        # Fall back: no group reshape possible; skip clipping gracefully.
        logging.warning(
            "awq_dp: auto_clip skipped (ci=%d not divisible by group_size=%d)",
            ci, group_size,
        )
        return
    n_group = ci // group_size

    # Decide OC batch size to cap peak memory, matching llm-awq heuristic.
    if co % 256 == 0:
        oc_bs = 256
    elif co % 64 == 0:
        oc_bs = 64
    else:
        oc_bs = co
    if co % oc_bs != 0:
        logging.warning(
            "awq_dp: auto_clip skipped (co=%d not divisible by oc_bs=%d)",
            co, oc_bs,
        )
        return

    # Build input tensor (sub-sampled to at most n_sample_token).
    # Keep reshape + stride on CPU, then one small transfer to GPU (avoids OOM).
    if n_local_tok > 0:
        x_flat_cpu = input_feat_3d.reshape(-1, ci)
        if x_flat_cpu.shape[0] > n_sample_token:
            stride = max(1, x_flat_cpu.shape[0] // n_sample_token)
            x_flat_cpu = x_flat_cpu[::stride][:n_sample_token]
        x_flat = x_flat_cpu.to(dev).float()
        x_tok = x_flat.shape[0]
        x_grp = x_flat.reshape(1, x_tok, n_group, group_size)
    else:
        x_tok = 0
        x_grp = torch.zeros(1, 0, n_group, group_size, device=dev, dtype=torch.float32)
    n_tok_tensor = torch.tensor([float(x_tok)], device=dev, dtype=torch.float64)
    n_tok_global = n_tok_tensor.clone()
    _all_reduce_sum(n_tok_global, dp_group)
    if float(n_tok_global.item()) <= 0:
        logging.warning("awq_dp: auto_clip skipped (no tokens across ranks)")
        return

    w_grp = w_full.reshape(co, 1, n_group, group_size)

    best_max_val_all = []
    for i_b in range(co // oc_bs):
        w_b = w_grp[i_b * oc_bs : (i_b + 1) * oc_bs]  # [oc_bs, 1, ng, gs]
        org_max_val = w_b.abs().amax(dim=-1, keepdim=True)  # [oc_bs, 1, ng, 1]
        best_max_val = org_max_val.clone()
        best_err = torch.full_like(org_max_val, float("inf"))

        if x_tok > 0:
            org_out = (x_grp * w_b).sum(dim=-1)  # [oc_bs, tok, ng]
        else:
            org_out = None

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1.0 - i_s / n_grid)
            cur_w = torch.clamp(w_b, -max_val, max_val)

            flat_w = cur_w.reshape(oc_bs, ci).contiguous()
            wq = _configured_weight_quantizer(args)
            wq.find_params(flat_w)
            q_flat, _, _ = wq.fake_quantize(flat_w)
            q_w = q_flat.float().reshape(oc_bs, 1, n_group, group_size)

            if org_out is not None:
                cur_out = (x_grp * q_w).sum(dim=-1)  # [oc_bs, tok, ng]
                local_err = (cur_out - org_out).pow(2).sum(dim=1)  # [oc_bs, ng]
            else:
                local_err = torch.zeros(oc_bs, n_group, device=dev, dtype=torch.float32)

            err_buf = local_err.double().contiguous()
            _all_reduce_sum(err_buf, dp_group)
            global_mse = (err_buf / n_tok_global.clamp_min(1.0)).to(best_err.dtype)
            global_mse = global_mse.view(oc_bs, 1, n_group, 1)

            mask = global_mse < best_err
            if torch.any(mask):
                best_err = torch.where(mask, global_mse, best_err)
                best_max_val = torch.where(mask, max_val, best_max_val)

            del cur_w, flat_w, q_flat, q_w
        best_max_val_all.append(best_max_val)

    best_max_val_full = torch.cat(best_max_val_all, dim=0).squeeze(1)  # [co, ng, 1]

    # Clamp in place on the real weight, preserving dtype.
    orig_shape = lin.weight.data.shape
    work = lin.weight.data.to(dev).float().reshape(co, n_group, group_size)
    work.clamp_(-best_max_val_full.to(dev), best_max_val_full.to(dev))
    lin.weight.data = work.reshape(orig_shape).to(lin.weight.data.dtype).to(lin.weight.device)


def _run_auto_clip(
    analyzer: model_utils.ModelAnalyzer,
    layer: nn.Module,
    input_feat_dict: Dict[str, torch.Tensor],
    args,
    dev: torch.device,
    dp_group: Optional[Any] = None,
) -> None:
    """Apply ``_auto_clip_linear_dp`` to each non-q/k linear in the layer."""
    for rel_name in list(input_feat_dict.keys()):
        # llm-awq skips q/k because the qk bmm makes clipping unstable.
        if any(tag in rel_name for tag in (".q_proj", ".k_proj")):
            continue
        lin = _unwrap_linear(_get_child(layer, rel_name))
        feat = input_feat_dict.get(rel_name)
        if feat is None:
            continue
        _auto_clip_linear_dp(lin, feat, args, dev, dp_group=dp_group)


# ---------------------------------------------------------------------------
# Final weight quantization (matches the AWQ flow: mse=False + clip-already-applied)
# ---------------------------------------------------------------------------


def _quantize_layer_weights(
    layer_idx: int,
    full: Dict[str, nn.Module],
    sequential_names: Sequence[Sequence[str]],
    args,
    rank: int,
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
                mse=False,  # clipping already done by auto_clip when requested
                weight_groupsize=args.w_groupsize,
            )
            quantizer.find_params(weight)
            q_weight, _, _ = quantizer.fake_quantize(weight)
            module.weight.data = q_weight.to(module.weight.data.dtype)
            if rank == 0:
                quantizers[f"model.layers.{layer_idx}.{name}"] = quantizer.cpu()
    return quantizers


# ---------------------------------------------------------------------------
# Main entry
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
            "launch with torchrun (see ptq_awq_dp.py)."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_js = _local_sample_indices(rank, world_size, args.nsamples)
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

    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    seqlen = model.seqlen

    inps_shard: Optional[torch.Tensor] = None
    inps: Optional[torch.Tensor] = None

    # ---- capture first-layer inputs (same branching as before) ---------
    if shard_inps:
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
                inps_full_cpu[_local_sample_indices(r, world_size, args.nsamples), :, :].contiguous()
                for r in range(world_size)
            ]
        else:
            scatter_objs = None
        recv_list: List[Any] = [None]
        dist.scatter_object_list(recv_list, scatter_objs, src=0, group=group)
        shard_storage = torch.device("cpu") if args.offload_inps else dev
        inps_shard = recv_list[0].to(shard_storage)
        del recv_list
        if rank == 0:
            del inps_full_cpu, cache
        memory_utils.cleanup_memory(False)
    else:
        inps = torch.zeros(
            (args.nsamples, seqlen, hidden), dtype=dtype, device=dev
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
            inps = inps.contiguous()
            dist.broadcast(inps, src=0, group=group)
        if args.offload_inps:
            inps = inps.cpu()

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    # ---- layer-wise AWQ quantization -----------------------------------
    quantizers: Dict[str, quant_utils.WeightQuantizer] = {}
    pbar = tqdm(
        range(len(layers)),
        ncols=120,
        desc="Quantizing Layers (AWQ DP)",
        disable=(rank != 0),
    )
    sequential = analyzer.get_sequential_quantizable_module_names()

    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)
        bits_config = quant_utils.disable_act_quant(layer)

        groups = _awq_groups_for_layer(analyzer, layer)
        # Collect all linears that either feed a group-hook or are clip
        # targets later; deduplicate while preserving iteration order.
        clip_targets_rel = [name.replace(".module", "") for name in full.keys()]
        clip_targets_rel = sorted(set(clip_targets_rel))
        hook_targets = {g["hook"] for g in groups}
        rel_to_cache = sorted(set(list(hook_targets) + clip_targets_rel))

        # Source tensor & indices for this rank.
        if shard_inps:
            assert inps_shard is not None
            src_tensor = inps_shard
            local_indices = list(range(len(local_js)))
        else:
            assert inps is not None
            src_tensor = inps
            local_indices = local_js

        input_feats = _cache_linear_inputs(
            layer,
            rel_to_cache,
            src_tensor,
            local_indices,
            dev,
            attention_mask,
            position_ids,
            position_embeddings,
        )

        # AWQ scale groups (output-MSE grid search + apply).
        for g in groups:
            hook_feat = input_feats.get(g["hook"])
            if hook_feat is None:
                continue
            scales = _search_group_scale_dp(
                layer, g, hook_feat,
                attention_mask, position_ids, position_embeddings,
                args, dev, dp_group=group,
            )
            _apply_group_scale(layer, g, scales, input_feat_dict=input_feats)

        # Optional per-linear auto-clip (matches llm-awq when mse_range=True).
        if getattr(args, "w_clip", False):
            _run_auto_clip(analyzer, layer, input_feats, args, dev, dp_group=group)

        # Free cached activations before the quant + propagation passes.
        del input_feats
        memory_utils.cleanup_memory(False)

        quant_utils.enable_act_quant(layer, bits_config)
        quantizers.update(_quantize_layer_weights(i, full, sequential, args, rank))
        if world_size > 1:
            dist_utils.broadcast_parameters(layer, src=0, group=group)

        # Propagate hidden states to build next layer's input.
        if shard_inps:
            assert inps_shard is not None
            for k in range(len(local_js)):
                inps_shard[k] = layer(
                    inps_shard[k].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0].to(inps_shard.device)
        else:
            assert inps is not None
            for j in range(args.nsamples):
                if rank == 0:
                    out_j = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(inps.device)
                    inps[j] = out_j.contiguous()
                if world_size > 1:
                    buf = (
                        inps[j].to(dev).contiguous()
                        if rank == 0
                        else torch.empty(
                            inps[j].shape,
                            dtype=inps.dtype,
                            device=dev,
                        )
                    )
                    dist.broadcast(buf, src=0, group=group)
                    inps[j] = buf.contiguous().to(inps.device)

        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----AWQ DP Quantization Done (rank=%d)-----\n", rank)
    return quantizers


def resolve_local_cuda_device() -> torch.device:
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{lr}")
