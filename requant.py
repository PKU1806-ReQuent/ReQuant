# coding=utf-8
"""ReQuant: discrete coordinate descent on integer weight grids (PTQ-agnostic).

Exports:
  * ``ReQuantConfig`` / ``requant_from_config`` — packaged core (any PTQ that
    provides ``Q, W_int, scale, zero``, Hessian ``H_orig``, optional ``dXXT``)
  * ``requant`` — same math, flat hyperparameters (backward compatible)
  * ``GPTQReQuant`` / ``FPInputsCache`` — GPTAQ/GPTQ Phase-1 + Phase-2
  * ``requant_layer`` — block-wise H + ``requant`` after AWQ/RTN
"""

import copy
import functools
import logging
import math
import pprint
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import memory_utils, quant_utils


# ---------------------------------------------------------------------------
#  FPInputsCache: store full-precision sub-layer inputs for dXXT
# ---------------------------------------------------------------------------

class FPInputsCache:
    def __init__(self, sequential):
        self.names = [n for names in sequential for n in names]
        self.fp_cache = {n: [] for n in self.names}
        self.handles = []

    def _cache(self, _m, inp, _out, name):
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        self.fp_cache[name].append(x.t())

    def add_hook(self, full):
        for name in self.names:
            mod = full.get(name, full.get(name + ".module", None))
            mod = getattr(mod, "module", mod)
            self.handles.append(
                mod.register_forward_hook(functools.partial(self._cache, name=name))
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
#  Core: vectorised per-row coordinate descent (PTQ-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class ReQuantConfig:
    """Hyperparameters for :func:`requant_from_config`.

    ``min_gain_eps`` is the **layer-scaled** threshold (negative of the
    ``neg_eps`` used inside the loop), same convention as the old
    ``requant_min_gain_eps * layer_scale`` path in ``GPTQReQuant``.
    """

    num_sweeps: int
    num_candidates: int = 2
    beta: float = 0.3
    lambda_anchor: float = 0.0
    min_gain_eps: float = 0.0
    early_stop_consecutive_cols: int = 0


def requant_config_from_phase2_args(args: Any) -> ReQuantConfig:
    """Build config from CLI ``--requant_*`` (GPTQReQuant / GPTAQ DP Phase-2)."""
    beta = getattr(args, "requant_beta", None)
    if beta is None:
        beta = 0.3
    return ReQuantConfig(
        num_sweeps=int(getattr(args, "requant_phase2_sweeps", 3)),
        num_candidates=int(getattr(args, "requant_phase2_candidates", 2)),
        beta=float(beta),
        lambda_anchor=float(getattr(args, "requant_anchor_lambda", 0.0)),
        min_gain_eps=float(getattr(args, "requant_min_gain_eps", 0.0)),
        early_stop_consecutive_cols=int(
            getattr(args, "requant_early_stop_consecutive_cols", 0)
        ),
    )


def requant_config_from_args(args: Any) -> ReQuantConfig:
    """Build config from ``--requant_*`` arguments."""
    beta = getattr(args, "requant_beta", None)
    if beta is None:
        beta = 0.3
    return ReQuantConfig(
        num_sweeps=int(getattr(args, "requant_phase2_sweeps", 3)),
        num_candidates=int(
            getattr(args, "requant_phase2_candidates", 2)
        ),
        beta=float(beta),
        lambda_anchor=float(
            getattr(args, "requant_anchor_lambda", 0.0)
        ),
        min_gain_eps=float(
            getattr(
                args,
                "requant_min_gain_eps",
                0.0,
            )
        ),
        early_stop_consecutive_cols=int(
            getattr(
                args,
                "requant_early_stop_consecutive_cols",
                0,
            )
        ),
    )


def requant_from_config(
    W_orig: torch.Tensor,
    Q: torch.Tensor,
    W_int: torch.Tensor,
    Scale: torch.Tensor,
    W_zero: torch.Tensor,
    H_orig: torch.Tensor,
    dXXT: Optional[torch.Tensor],
    min_int: int,
    max_int: int,
    cfg: ReQuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Coordinate descent on integer weight grid; use from any PTQ path.

    Preconditions (caller supplies):
      * ``Q`` / ``W_int`` / ``Scale`` / ``W_zero`` per-channel grid consistent
        with your quantizer; ``H_orig`` undamped Hessian in the same column
        basis as ``Q``.
      * ``dXXT`` is the GPTAQ-side cross term ``(X_fp - X_q) @ X_q.T``.
        This is ``-B`` if the paper defines ``B = (X_q - X_fp) @ X_q.T``.
        Pass ``None`` only when the caller deliberately wants pure weight MSE.
    """
    if cfg.num_sweeps <= 0 or cfg.num_candidates <= 0:
        return Q, W_int, 0

    _, d = Q.shape
    dev, rdtype = Q.device, Q.dtype
    beta = float(cfg.beta)

    G = (W_orig - Q) @ H_orig
    if dXXT is not None and beta != 0.0:
        # dXXT stores (X_fp - X_q) @ X_q.T, so this adds the -wB term from
        # the paper's gradient when B = (X_q - X_fp) @ X_q.T.
        G.add_(beta * (W_orig @ dXXT))

    H_diag = torch.diag(H_orig)
    ks_list = [
        k for k in range(-cfg.num_candidates, cfg.num_candidates + 1) if k != 0
    ]
    if not ks_list:
        return Q, W_int, 0
    ks = torch.tensor(ks_list, device=dev, dtype=rdtype).view(-1, 1)
    pos_inf = torch.tensor(float("inf"), device=dev, dtype=rdtype)
    neg_eps = torch.tensor(-float(cfg.min_gain_eps), device=dev, dtype=rdtype)

    lam = float(cfg.lambda_anchor)
    Q_anchor = Q.clone() if lam > 0 else None
    total_improved = 0

    for sweep in range(cfg.num_sweeps):
        sweep_improved = 0
        streak = 0
        for j in range(d):
            s_j = Scale[:, j]
            g_j = G[:, j]
            h_jj = H_diag[j]

            if lam > 0 and Q_anchor is not None:
                g_eff = g_j - lam * (Q[:, j] - Q_anchor[:, j])
                h_eff = h_jj + lam
            else:
                g_eff = g_j
                h_eff = h_jj

            dq = ks * s_j.unsqueeze(0)
            delta_L = -2.0 * dq * g_eff.unsqueeze(0) + (dq * dq) * h_eff
            z_new = W_int[:, j].unsqueeze(0) + ks
            delta_L = torch.where(
                (z_new >= min_int) & (z_new <= max_int), delta_L, pos_inf
            )

            best_gain, best_idx = delta_L.min(dim=0)
            improve = best_gain < neg_eps
            n = int(improve.sum().item())
            if n == 0:
                streak += 1
                esc = cfg.early_stop_consecutive_cols
                if 0 < esc <= streak:
                    total_improved += sweep_improved
                    logging.info("  requant early stop at col-streak=%d", streak)
                    return Q, W_int, total_improved
                continue

            sweep_improved += n
            int_step = ks[best_idx, 0].to(W_int.dtype) * improve.to(W_int.dtype)
            W_int[:, j] += int_step
            Q[:, j] = (
                W_int[:, j].to(s_j.dtype) - W_zero[:, j].to(s_j.dtype)
            ) * s_j
            G.sub_(torch.outer(int_step.to(s_j.dtype) * s_j, H_orig[j, :]))
            streak = 0

        total_improved += sweep_improved
        logging.info(
            "  requant sweep %d/%d: %d updates",
            sweep + 1, cfg.num_sweeps, sweep_improved,
        )
        if sweep_improved == 0:
            break

    return Q, W_int, total_improved


def requant(
    W_orig: torch.Tensor,
    Q: torch.Tensor,
    W_int: torch.Tensor,
    Scale: torch.Tensor,
    W_zero: torch.Tensor,
    H_orig: torch.Tensor,
    dXXT: Optional[torch.Tensor],
    beta: float,
    min_int: int,
    max_int: int,
    num_candidates: int,
    num_sweeps: int,
    lambda_anchor: float = 0.0,
    min_gain_eps: float = 0.0,
    early_stop_consecutive_cols: int = 0,
):
    """Same as :func:`requant_from_config` with flat hyperparameters."""
    cfg = ReQuantConfig(
        num_sweeps=num_sweeps,
        num_candidates=num_candidates,
        beta=beta,
        lambda_anchor=lambda_anchor,
        min_gain_eps=min_gain_eps,
        early_stop_consecutive_cols=early_stop_consecutive_cols,
    )
    return requant_from_config(
        W_orig, Q, W_int, Scale, W_zero, H_orig, dXXT, min_int, max_int, cfg,
    )


# ---------------------------------------------------------------------------
#  GPTQReQuant: Phase-1 quantisation + Phase-2 ``requant_from_config``
# ---------------------------------------------------------------------------

class GPTQReQuant:
    """Phase-1 quantiser + optional Phase-2 coordinate descent (``requant``).

    ``use_dxxt`` controls whether ``dXXT=(FP_X−Q_X)·X_qᵀ`` is accumulated for
    Phase-2 ReQuant. ``use_phase1_dxxt`` controls whether GPTAQ's Phase-1
    ``P = α·(dXXT·Hinvᵀ).triu(1)·Hinv`` correction is applied.

    Phase-2 uses :func:`requant_from_config`; when ``dXXT=None`` the beta-scaled
    cross term is skipped.
    """

    def __init__(
        self,
        layer,
        use_dxxt: bool = True,
        use_phase1_dxxt: Optional[bool] = None,
    ):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.use_dxxt = use_dxxt
        self.use_phase1_dxxt = use_dxxt if use_phase1_dxxt is None else use_phase1_dxxt
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = (
            torch.zeros((self.columns, self.columns), device=self.dev)
            if use_dxxt else None
        )
        self.nsamples = 0
        self.fp_inp = []

    def add_batch(self, inp, _out):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] * inp.shape[1]
        inp = inp.reshape(-1, inp.shape[-1]).t()

        factor = self.nsamples / (self.nsamples + tmp)
        self.H *= factor
        if self.use_dxxt:
            self.dXXT *= factor
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        if self.use_dxxt and self.fp_inp:
            # dXXT is intentionally (X_fp - X_q) @ X_q.T. ReQuant uses this
            # sign convention directly in G = eH + beta * W @ dXXT.
            dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
            self.dXXT += dX.matmul(inp.t())
            del self.fp_inp[0]

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
        alpha=0.25,
        requant_sweeps=3,
        requant_candidates=2,
        requant_anchor_lambda=0.0,
        requant_beta=0.3,
        requant_min_gain_eps=0.0,
        requant_early_stop_consecutive_cols=0,
        **extra_kwargs,
    ):
        if extra_kwargs:
            raise TypeError(f"unexpected fasterquant kwargs: {sorted(extra_kwargs)}")

        need_requant = requant_sweeps > 0
        W = self.layer.weight.data.clone().float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H.clone()
        del self.H
        if (H == 0).all():
            H = torch.eye(self.columns).to(H)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if self.use_dxxt:
            self.dXXT[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                q = copy.deepcopy(self.quantizer)
                q.find_params(W[:, i:(i + groupsize)])
                groups.append(q)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            if self.use_dxxt:
                self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        H_orig = H.clone() if need_requant else None
        W_orig = W.clone() if need_requant else None

        Q = torch.zeros_like(W)

        damp_percent = percdamp
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
                logging.warning("damp_percent=%.5f too low; +0.0015", damp_percent)
                damp_percent += 0.0015
        if not (0 < damp_percent < 1):
            raise ValueError(f"damp_percent must be in (0,1); got {damp_percent}")

        if self.use_dxxt:
            dXXT_saved = self.dXXT.clone() if need_requant else None
            if self.use_phase1_dxxt:
                P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
            else:
                P = None
            del self.dXXT
        else:
            dXXT_saved = None
            P = None

        has_dXXT_correction = (
            need_requant and dXXT_saved is not None and not (dXXT_saved == 0).all()
        )

        if need_requant:
            W_int = torch.zeros_like(W)
            Scale = torch.zeros_like(W)
            Zero = torch.zeros_like(W)
            maxq_i = int(self.quantizer.maxq.item())
            if self.quantizer.sym:
                min_int, max_int = -(maxq_i + 1), maxq_i
            else:
                min_int, max_int = 0, maxq_i
        else:
            W_int = Scale = Zero = None
            min_int = max_int = 0

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2] if P is not None else None

            if need_requant:
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
                                W[:, (i1 + i):(i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                update = err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                if P1 is not None:
                    update = update - w.unsqueeze(1) @ P1[i, i:].unsqueeze(0)
                W1[:, i:] -= update
                Err1[:, i] = err1

                if need_requant:
                    _, iw, sc = self.quantizer.fake_quantize(w.unsqueeze(1))
                    W_int1[:, i] = iw.flatten()
                    Scale1[:, i] = sc.flatten()
                    zbuf = self.quantizer.zero.to(w.device)
                    if zbuf.shape != sc.shape:
                        zbuf = zbuf.expand_as(sc)
                    Zero1[:, i] = zbuf.flatten()

            Q[:, i1:i2] = Q1
            tail = Err1 @ Hinv[i1:i2, i2:]
            if P is not None:
                tail = tail - W1 @ P[i1:i2, i2:]
            W[:, i2:] -= tail

            if need_requant:
                W_int[:, i1:i2] = W_int1
                Scale[:, i1:i2] = Scale1
                Zero[:, i1:i2] = Zero1

        if need_requant:
            dXXT_refine = dXXT_saved if has_dXXT_correction else None
            layer_gain_scale = (
                torch.diag(H_orig).float().mean() * Scale.float().pow(2).mean()
            )
            min_gain_eps_layer = (
                float(requant_min_gain_eps) * float(layer_gain_scale.item())
            )
            rq_cfg = ReQuantConfig(
                num_sweeps=requant_sweeps,
                num_candidates=requant_candidates,
                beta=requant_beta,
                lambda_anchor=requant_anchor_lambda,
                min_gain_eps=min_gain_eps_layer,
                early_stop_consecutive_cols=requant_early_stop_consecutive_cols,
            )
            logging.info(
                "Phase-2 requant: mode=%s sweeps=%d cand=%d alpha=%.3f beta=%.3f "
                "anchor=%.4g eps_raw=%.4g eps_layer=%.4g early=%d dXXT=%s",
                "gptaq" if self.use_phase1_dxxt else "gptq",
                requant_sweeps, requant_candidates, alpha, rq_cfg.beta,
                requant_anchor_lambda, requant_min_gain_eps, min_gain_eps_layer,
                requant_early_stop_consecutive_cols, has_dXXT_correction,
            )
            Q, W_int, n_improved = requant_from_config(
                W_orig, Q, W_int, Scale, Zero, H_orig, dXXT_refine,
                min_int, max_int, rq_cfg,
            )
            logging.info("Phase-2 requant done: %d updates", n_improved)

        del P
        if need_requant:
            del H_orig, W_orig
            if dXXT_saved is not None:
                del dXXT_saved

        if actorder:
            Q = Q[:, invperm]
            if need_requant:
                W_int = W_int[:, invperm]
                Scale = Scale[:, invperm]
                Zero = Zero[:, invperm]

        if export_to_et:
            if not need_requant:
                W_int = torch.zeros_like(Q)
                Scale = torch.zeros_like(Q)
                Zero = torch.zeros_like(Q)
                for jc in range(self.columns):
                    col = Q[:, jc:jc + 1]
                    self.quantizer.find_params(col)
                    _, qi, sc = self.quantizer.fake_quantize(col)
                    W_int[:, jc] = qi.flatten()
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
            pprint.pprint({
                "bits": self.quantizer.bits,
                "scale": self.quantizer.scale,
                "zero": self.quantizer.zero,
            })
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.dXXT = None
        torch.cuda.empty_cache()
        memory_utils.cleanup_memory(verbos=False)


# ---------------------------------------------------------------------------
#  requant_layer: AWQ / RTN post-processing; collect H and optional dXXT
# ---------------------------------------------------------------------------

def _recover_w_int(Q: torch.Tensor, quantizer) -> torch.Tensor:
    """sym:  W_int = round(Q/scale); asym: W_int = round(Q/scale) + zero."""
    dev = Q.device
    scale = quantizer.scale.to(dev).float()
    maxq = int(quantizer.maxq.item())
    Qf = Q.float()
    if quantizer.sym:
        return torch.clamp(torch.round(Qf / scale), -(maxq + 1), maxq)
    zero = quantizer.zero.to(dev).float()
    return torch.clamp(torch.round(Qf / scale) + zero, 0, maxq)


@torch.no_grad()
def requant_layer(
    layer: nn.Module,
    full: dict,
    sequential_names: Sequence[Sequence[str]],
    layer_idx: int,
    quantizers: dict,
    fp_weights: Optional[dict],
    inps: torch.Tensor,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    args,
    dev: torch.device,
    input_indices: Optional[Sequence[int]] = None,
    group: Optional[Any] = None,
    fp_inputs_cache: Optional[dict] = None,
):
    """Collect H/dXXT on a quantised block and run ``requant_from_config``.

    Used for AWQ / RTN where H is not accumulated at quant time. ``fp_weights``
    must contain the original full-precision weights keyed by either
    ``model.layers.{layer_idx}.{name}`` or ``name``. ``fp_inputs_cache`` should
    contain full-precision inner-linear inputs so the same dXXT term as GPTAQ
    can be used during ReQuant.
    """
    sweeps = getattr(args, "requant_sweeps", 0)
    if sweeps <= 0:
        return
    if fp_weights is None:
        logging.warning(
            "requant_layer skipped for layer %d: fp_weights is required so "
            "W_orig is not confused with the already-quantized Q.",
            layer_idx,
        )
        return

    num_candidates = getattr(args, "requant_candidates", 2)
    requant_beta = getattr(args, "requant_beta", None)
    if requant_beta is None:
        requant_beta = 0.3
    min_gain_eps = getattr(args, "requant_min_gain_eps", 0.003)
    percdamp = getattr(args, "percdamp", 0.01)
    nsamples = inps.shape[0]
    sample_indices = range(nsamples) if input_indices is None else input_indices

    for names in sequential_names:
        first_mod = full.get(names[0]) or full.get(names[0] + ".module")
        if first_mod is None:
            continue
        first_mod = getattr(first_mod, "module", first_mod)
        cols = first_mod.weight.shape[1]

        H = torch.zeros(cols, cols, device=dev)
        dXXT = torch.zeros(cols, cols, device=dev)
        n_acc = 0
        fp_cache = None if fp_inputs_cache is None else fp_inputs_cache.get(names[0])
        cache_pos = 0
        use_dxxt = fp_cache is not None

        def _hook(_m, inp, _out, _H=H, _dXXT=dXXT):
            nonlocal n_acc, cache_pos
            x = inp[0].detach()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            tmp = x.shape[0] * x.shape[1]
            x = x.reshape(-1, x.shape[-1]).t()
            factor = n_acc / (n_acc + tmp)
            _H.mul_(factor)
            if use_dxxt:
                _dXXT.mul_(factor)
            n_acc += tmp
            x = math.sqrt(2 / n_acc) * x.float()
            _H.add_(x.matmul(x.t()))
            if use_dxxt:
                if cache_pos >= len(fp_cache):
                    raise RuntimeError(
                        f"requant_layer layers.{layer_idx}.{names[0]}: "
                        "fp_inputs_cache is shorter than quantized calibration forwards."
                    )
                fp_x = fp_cache[cache_pos].to(device=x.device, dtype=x.dtype)
                _dXXT.add_((fp_x * math.sqrt(2 / n_acc) - x).matmul(x.t()))
                cache_pos += 1

        handle = first_mod.register_forward_hook(_hook)
        for j in sample_indices:
            layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        handle.remove()

        if dist.is_available() and dist.is_initialized():
            n_buf = torch.tensor([float(n_acc)], device=dev, dtype=torch.float64)
            H_acc = H * float(n_acc)
            d_acc = dXXT * float(n_acc)
            dist.all_reduce(n_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(H_acc, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(d_acc, op=dist.ReduceOp.SUM, group=group)
            n_total = float(n_buf.item())
            if n_total <= 0:
                raise RuntimeError(
                    f"requant_layer layers.{layer_idx}: no calibration samples collected."
                )
            H = H_acc / n_total
            dXXT = d_acc / n_total

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        dXXT[:, dead] = 0
        H_orig = H.clone()
        H.diagonal().add_(percdamp * torch.mean(torch.diag(H)))

        for name in names:
            q_key = f"model.layers.{layer_idx}.{name}"
            if q_key not in quantizers:
                continue
            quantizer = quantizers[q_key]
            if not quantizer.ready():
                continue
            mod = full.get(name) or full.get(name + ".module")
            if mod is None:
                continue
            mod = getattr(mod, "module", mod)

            Q_w = mod.weight.data.float().clone()
            Q_w[:, dead] = 0
            W_orig = fp_weights.get(q_key, fp_weights.get(name, None))
            if W_orig is None:
                logging.warning(
                    "requant_layer skipped layers.%d.%s: missing FP weight in "
                    "fp_weights under key '%s' or '%s'.",
                    layer_idx, name, q_key, name,
                )
                continue
            W_orig = W_orig.to(device=Q_w.device, dtype=Q_w.dtype).clone()
            W_orig[:, dead] = 0
            scale = quantizer.scale.to(dev).float()
            zero = quantizer.zero.to(dev).float()
            if scale.shape != Q_w.shape:
                scale = scale.expand_as(Q_w)
            if zero.shape != Q_w.shape:
                zero = zero.expand_as(Q_w)
            maxq = int(quantizer.maxq.item())
            W_int = _recover_w_int(Q_w, quantizer)
            min_int = -(maxq + 1) if quantizer.sym else 0
            max_int = maxq

            eps_layer = min_gain_eps * float(
                torch.diag(H_orig).float().mean() * scale.pow(2).mean()
            )
            cfg_rq = ReQuantConfig(
                num_sweeps=sweeps,
                num_candidates=num_candidates,
                beta=requant_beta,
                lambda_anchor=0.0,
                min_gain_eps=eps_layer,
                early_stop_consecutive_cols=0,
            )
            Q_new, _, n_upd = requant_from_config(
                W_orig, Q_w, W_int, scale, zero,
                H_orig, dXXT if use_dxxt else None, min_int, max_int, cfg_rq,
            )
            mod.weight.data = Q_new.to(mod.weight.dtype)
            logging.info(
                "requant_layer layers.%d.%s sweeps=%d beta=%.3f dXXT=%s updates=%d",
                layer_idx, name, sweeps, cfg_rq.beta, use_dxxt, n_upd,
            )
