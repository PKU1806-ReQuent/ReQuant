# coding=utf-8
"""
GPTAQ + Post-Quantization Discrete Coordinate Refinement  (Scheme C)

Two-phase quantization:
  Phase 1 – Same column-wise GPTAQ as ``gptaq_utils.GPTAQ.fasterquant`` (``quantize``,
            ``self.dXXT`` dead-column handling, ``P`` then ``del self.dXXT``).
  Phase 2 – Per-row coordinate descent in the discrete grid, with
            activation-error correction via dXXT (optional; ``refine_sweeps``).

Phase 1 (GPTAQ) compensation:
    P = α · triu(dXXT · Hinv^T, diag=1) · Hinv
    W[:, i:] -= err·Hinv[i,i:] − w·P[i,i:]
    W[:, i2:] -= Err·Hinv[i1:i2,i2:] − W1·P[i1:i2,i2:]

Phase 2 refinement objective per row:
    L_i = e^T H e  +  2 (dXXT^T w)^T e
    g̃ = H e + dXXT^T · w
    ΔL = −2 Δq_j g̃_j  +  Δq_j² H_{jj}
    g̃ ← g̃ − Δq_j H[:,j]

Dequant on the integer grid: symmetric Q = s·z; asymmetric Q = s·(z − z0)
with z in [−(maxq+1), maxq] or [0, maxq] respectively (see WeightQuantizer).

When fp inputs are unavailable, dXXT ≡ 0 and both phases degrade to
standard GPTQ + Scheme-A refinement.
"""

import copy
import functools
import logging
import math
import pprint
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import quant_utils, memory_utils, model_utils


# ---------------------------------------------------------------------------
#  FP input cache (reused from GPTAQ, kept self-contained)
# ---------------------------------------------------------------------------

class FPInputsCache:
    """Collect full-precision sub-layer inputs for computing dXXT."""

    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = []
        for names in sequential:
            self.names += names
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name] += [inp.t()]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        memory_utils.cleanup_memory()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []
        memory_utils.cleanup_memory()


# ---------------------------------------------------------------------------
#  GPTAQRefined  –  GPTAQ + discrete coordinate refinement
# ---------------------------------------------------------------------------

class GPTAQRefined:
    """GPTAQ Phase-1 quantisation (with P-matrix) + Phase-2 coordinate refinement."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []

    # ---- Hessian / dXXT accumulation ------------------------------------

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # Token-level sample count: for [B, T, d], count B*T (not just B).
        tmp = inp.shape[0] * inp.shape[1] if len(inp.shape) == 3 else inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        if self.fp_inp:
            dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
            self.dXXT += dX.matmul(inp.t())
            del self.fp_inp[0]

    # ---- Phase-2: vectorised coordinate descent -------------------------

    def _refine(self, W_orig, Q, W_int, Scale, W_zero, H_orig, dXXT,
                alpha, min_int, max_int, num_candidates, num_sweeps,
                lambda_anchor=0.0,
                min_gain_eps=0.0,
                early_stop_consecutive_cols=0):
        """
        Batch coordinate descent over all rows simultaneously.

        For each column *j* in every sweep we:
          1. Evaluate 2·num_candidates neighbouring grid points per row.
          2. Accept the change that maximally decreases L_i.
          3. Update the running gradient  g̃ ← g̃ − Δq_j H[:,j].

        If min_gain_eps > 0, accept only when best_gain < -min_gain_eps (ignore tiny/noisy gains).
        If early_stop_consecutive_cols > 0, stop refinement after that many consecutive
        columns with zero accepted updates.
        """
        _, d = Q.shape
        device = Q.device

        E = W_orig - Q
        G = E @ H_orig                               # g̃ = H e

        has_correction = dXXT is not None and alpha > 0
        if has_correction:
            G += alpha * (W_orig @ dXXT)              # g̃ += α dXXT^T w (w fixed)

        H_diag = torch.diag(H_orig)                  # [d]
        rdtype = G.dtype
        ks_list = [
            k for k in range(-num_candidates, num_candidates + 1) if k != 0
        ]
        if not ks_list:
            return Q, W_int, 0
        ks = torch.tensor(ks_list, device=device, dtype=rdtype).view(-1, 1)
        pos_inf = torch.tensor(float("inf"), device=device, dtype=rdtype)
        neg_eps = torch.tensor(-float(min_gain_eps), device=device, dtype=rdtype)
        total_improved = 0
        # Anchor to GPTAQ Phase-1 solution (Q at refinement start).
        Q_anchor = Q.clone()

        for sweep in range(num_sweeps):
            col_no_update_streak = 0
            sweep_improved = 0
            for j in range(d):
                s_j = Scale[:, j]                     # [n_row]
                z_j = W_int[:, j]                     # [n_row]
                g_j = G[:, j]                         # [n_row]
                h_jj = H_diag[j]                      # scalar
                h_row = H_orig[j, :]
                if lambda_anchor > 0:
                    anchor_diff = Q[:, j] - Q_anchor[:, j]
                    g_eff = g_j - lambda_anchor * anchor_diff
                    h_eff = h_jj + lambda_anchor
                else:
                    g_eff = g_j
                    h_eff = h_jj

                dq_all = ks * s_j.unsqueeze(0)
                g_exp = g_eff.unsqueeze(0)
                delta_L = -2.0 * dq_all * g_exp + (dq_all * dq_all) * h_eff
                z_new = z_j.unsqueeze(0) + ks
                valid = (z_new >= min_int) & (z_new <= max_int)
                delta_L = torch.where(valid, delta_L, pos_inf)
                best_gain, best_k_idx = delta_L.min(dim=0)
                # ks has shape [K, 1]; select per-row candidate from dim-0.
                best_k = ks[best_k_idx, 0]

                improve = best_gain < neg_eps
                n_updated = int(improve.sum().item())
                if n_updated > 0:
                    sweep_improved += n_updated
                    int_step = best_k.to(W_int.dtype) * improve.to(W_int.dtype)
                    W_int[:, j] += int_step
                    # Rebuild from integer grid to keep Q exactly on quantization lattice.
                    z_eff = W_int[:, j].to(s_j.dtype) - W_zero[:, j].to(s_j.dtype)
                    Q[:, j] = z_eff * s_j
                    dq_real = int_step.to(s_j.dtype) * s_j
                    G.sub_(torch.outer(dq_real, h_row))
                    col_no_update_streak = 0
                else:
                    col_no_update_streak += 1
                    if (
                        early_stop_consecutive_cols > 0
                        and col_no_update_streak >= early_stop_consecutive_cols
                    ):
                        total_improved += sweep_improved
                        logging.info(
                            "  Refinement early stop: %d consecutive columns with no update "
                            "(threshold=%d)",
                            col_no_update_streak,
                            early_stop_consecutive_cols,
                        )
                        return Q, W_int, total_improved

            total_improved += sweep_improved
            logging.info("  Refinement sweep %d/%d: %d element updates",
                         sweep + 1, num_sweeps, sweep_improved)
            if sweep_improved == 0:
                break

        return Q, W_int, total_improved

    # ---- Main entry: Phase 1 + Phase 2 ---------------------------------

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
        alpha=0.25,
        refine_sweeps=3,
        refine_candidates=2,
        refine_anchor_lambda=0.0,
        refine_min_gain_eps=0.0,
        refine_early_stop_consecutive_cols=0,
    ):
        need_refine = refine_sweeps > 0

        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        # IMPORTANT: H may be shared across modules in a sequential group.
        H = self.H.clone()
        del self.H
        if (H == 0).all():
            H = torch.eye(self.columns).to(H)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        H_orig = H.clone() if need_refine else None
        W_orig = W.clone() if need_refine else None

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp_percent = percdamp
        damp_auto_increment = 0.0015
        while 1 > damp_percent > 0:
            try:
                damp = damp_percent * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp

                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                break
            except torch._C._LinAlgError:
                logging.warning(
                    f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, "
                    f"auto-incrementing by `{damp_auto_increment:.5f}`"
                )
                damp_percent += damp_auto_increment

        if not (0 < damp_percent < 1):
            raise ValueError(
                f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}"
            )

        dXXT_saved = self.dXXT.clone() if need_refine else None
        P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del self.dXXT

        has_dXXT_correction = (
            need_refine
            and dXXT_saved is not None
            and not (dXXT_saved == 0).all()
        )

        if need_refine:
            W_int = torch.zeros_like(W)
            Scale = torch.zeros_like(W)
            Zero = torch.zeros_like(W)
            maxq_i = int(self.quantizer.maxq.item())
            if self.quantizer.sym:
                min_int, max_int = -(maxq_i + 1), maxq_i
            else:
                min_int, max_int = 0, maxq_i
        else:
            W_int = None
            Scale = None
            Zero = None
            min_int, max_int = 0, 0

        # Phase 1: match gptaq_utils.GPTAQ.fasterquant (quantize + same P / updates).
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            if need_refine:
                W_int1 = torch.zeros_like(W1)
                Scale1 = torch.zeros_like(W1)
                Zero1 = torch.zeros_like(W1)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                ) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

                if need_refine:
                    _, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                    W_int1[:, i] = int_weight.flatten()
                    Scale1[:, i] = scale.flatten()
                    zbuf = self.quantizer.zero.to(w.device)
                    if zbuf.shape != scale.shape:
                        zbuf = zbuf.expand_as(scale)
                    Zero1[:, i] = zbuf.flatten()

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

            if need_refine:
                W_int[:, i1:i2] = W_int1
                Scale[:, i1:i2] = Scale1
                Zero[:, i1:i2] = Zero1

        torch.cuda.synchronize()

        if need_refine:
            dXXT_refine = dXXT_saved if has_dXXT_correction else None
            layer_gain_scale = (
                torch.diag(H_orig).float().mean() * Scale.float().pow(2).mean()
            )
            min_gain_eps_layer = float(refine_min_gain_eps) * float(
                layer_gain_scale.item()
            )
            logging.info(
                "Phase-2 refinement: sweeps=%d, candidates=%d, alpha=%.3f, "
                "anchor_lambda=%.4g, min_gain_eps=%.4g (layer=%.4g), "
                "early_stop_consecutive_cols=%d, "
                "dXXT_correction=%s",
                refine_sweeps,
                refine_candidates,
                alpha,
                refine_anchor_lambda,
                refine_min_gain_eps,
                min_gain_eps_layer,
                refine_early_stop_consecutive_cols,
                has_dXXT_correction,
            )
            Q, W_int, n_improved = self._refine(
                W_orig,
                Q,
                W_int,
                Scale,
                Zero,
                H_orig,
                dXXT_refine,
                alpha,
                min_int,
                max_int,
                refine_candidates,
                refine_sweeps,
                refine_anchor_lambda,
                min_gain_eps_layer,
                refine_early_stop_consecutive_cols,
            )
            logging.info("Phase-2 done: %d total element updates", n_improved)

        del P
        if need_refine:
            del H_orig, W_orig
            if dXXT_saved is not None:
                del dXXT_saved

        if actorder:
            Q = Q[:, invperm]
            if need_refine:
                W_int = W_int[:, invperm]
                Scale = Scale[:, invperm]
                Zero = Zero[:, invperm]

        if export_to_et:
            if not need_refine:
                W_int = torch.zeros_like(Q)
                Scale = torch.zeros_like(Q)
                Zero = torch.zeros_like(Q)
                for jc in range(self.columns):
                    col = Q[:, jc : jc + 1]
                    self.quantizer.find_params(col)
                    _, qint, sc = self.quantizer.fake_quantize(col)
                    W_int[:, jc] = qint.flatten()
                    Scale[:, jc] = sc.flatten()
                    zbuf = self.quantizer.zero.to(Q.device)
                    if zbuf.shape != sc.shape:
                        zbuf = zbuf.expand_as(sc)
                    Zero[:, jc] = zbuf.flatten()
            self.layer.register_buffer(
                "int_weight", W_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale)
            if not self.quantizer.sym:
                self.layer.register_buffer(
                    "zero", Zero.reshape(self.layer.weight.shape)
                )

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")
            pprint.pprint(
                {
                    "bits": self.quantizer.bits,
                    "scale": self.quantizer.scale,
                    "zero": self.quantizer.zero,
                }
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.dXXT = None
        self.Losses = None
        torch.cuda.empty_cache()
        memory_utils.cleanup_memory(verbos=False)


# ---------------------------------------------------------------------------
#  Layer-by-layer orchestration  (mirrors gptaq_utils.gptq_fwrd)
# ---------------------------------------------------------------------------

@torch.no_grad()
def gptq_refined_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    Layer-wise GPTAQ + refinement.

    Flow per transformer block:
      1. FP forward  → collect fp sub-layer inputs   (for dXXT)
      2. Quant-path forward → accumulate H and dXXT   (hooks on add_batch)
      3. fasterquant  → Phase 1 GPTAQ (P-matrix)  +  Phase 2 refinement
      4. Update inps through quantised layer for next block
    """
    logging.info("-----GPTAQ-Refined (Scheme C) Quantization-----")

    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
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

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()

    fp_inputs_cache = FPInputsCache(sequential)
    fp_inps = inps.clone()

    if args.offload_inps:
        inps = inps.cpu()
        fp_inps = fp_inps.cpu()

    refine_sweeps = getattr(args, "refine_sweeps", 3)
    refine_candidates = getattr(args, "refine_candidates", 2)
    refine_anchor_lambda = getattr(args, "refine_anchor_lambda", 0.0)
    refine_min_gain_eps = args.refine_min_gain_eps
    refine_early_stop_consecutive_cols = getattr(
        args, "refine_early_stop_consecutive_cols", 0
    )

    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers")
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        # --- 1. FP forward: collect fp sub-layer inputs -----------------
        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)
        for j in range(args.nsamples):
            fp_inps[j] = layer(
                fp_inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].to(fp_inps.device)
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        # --- 2. Per-sequential-group quantisation -----------------------
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

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(
                add_batch(first_module_name)
            )
            for j in range(args.nsamples):
                _ = layer(
                    inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            handle.remove()

            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                pbar.set_postfix(module=f"layers.{i}." + name)
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
                    refine_min_gain_eps=refine_min_gain_eps,
                    refine_early_stop_consecutive_cols=refine_early_stop_consecutive_cols,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = (
                    gptq[name].quantizer
                )
                gptq[name].free()

        # --- 3. Propagate quantised outputs for next layer --------------
        for j in range(args.nsamples):
            inps[j] = layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].to(inps.device)

        fp_inputs_cache.clear_cache()
        layers[i] = layer.to(orig_device)
        del layer
        del gptq
        memory_utils.cleanup_memory()

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----GPTAQ-Refined Quantization Done-----\n")
    return quantizers
