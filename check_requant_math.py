"""Self-consistency checks for ``requant_from_config``.

Run from the ReQuant project root:

    cd /root/autodl-tmp/ReQuant
    python check_requant_math.py

The production ``requant_from_config`` keeps ``G`` and ``L_row`` via
incremental updates (``G.sub_(outer(dq, H[j]))`` etc.).  Any bug in those
increments would make ``G``/``L_row`` drift away from ground truth but the
per-column step would still "look" consistent with itself, so it is easy to
miss.

This script runs the production implementation alongside a naive reference
that recomputes ``G`` and ``L_row`` from scratch at every column, and
requires them to produce identical ``Q``/``W_int`` when ``min_gain_eps=0``
(i.e. accept every strictly improving move).  It additionally checks that
the true objective

    L_eff(Q) = sum_i [ e_i^T H e_i + 2*beta * e_i . (W_orig @ dXXT)_i ]

is monotonically non-increasing end-to-end.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import Optional, Tuple

import torch

from requant import ReQuantConfig, requant_from_config
from utils.quant_utils import WeightQuantizer


# ---------------------------------------------------------------------------
#  Reference implementation: recompute G and L_row from scratch every step.
# ---------------------------------------------------------------------------


def reference_requant_from_config(
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
    Q = Q.clone()
    W_int = W_int.clone()
    rows, d = Q.shape
    dev, rdtype = Q.device, Q.dtype

    beta = float(cfg.beta)
    lam = float(cfg.lambda_anchor)
    Q_anchor = Q.clone() if lam > 0 else None

    ks_list = [
        k for k in range(-cfg.num_candidates, cfg.num_candidates + 1) if k != 0
    ]
    if not ks_list or cfg.num_sweeps <= 0:
        return Q, W_int, 0
    ks = torch.tensor(ks_list, device=dev, dtype=rdtype).view(-1, 1)
    H_diag = torch.diag(H_orig)
    pos_inf = torch.tensor(float("inf"), device=dev, dtype=rdtype)
    min_gain_eps = torch.tensor(float(cfg.min_gain_eps), device=dev, dtype=rdtype)

    total_improved = 0
    for sweep in range(cfg.num_sweeps):
        sweep_improved = 0
        streak = 0
        for j in range(d):
            # Fresh recomputation of G and L_row every coordinate step.
            E = W_orig - Q
            G_eh = E @ H_orig
            if dXXT is not None and beta != 0.0:
                WB = W_orig @ dXXT
                G_full = G_eh + beta * WB
                L_row = (G_eh * E).sum(dim=1) + 2.0 * beta * (E * WB).sum(dim=1)
            else:
                G_full = G_eh
                L_row = (G_eh * E).sum(dim=1)
            L_row = L_row.clamp_min(1e-12)

            s_j = Scale[:, j]
            g_j = G_full[:, j]
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
            eps_row = min_gain_eps * (L_row / d)
            improve = best_gain < -eps_row
            n = int(improve.sum().item())
            if n == 0:
                streak += 1
                esc = cfg.early_stop_consecutive_cols
                if 0 < esc <= streak:
                    total_improved += sweep_improved
                    return Q, W_int, total_improved
                continue
            sweep_improved += n
            int_step = ks[best_idx, 0].to(W_int.dtype) * improve.to(W_int.dtype)
            W_int[:, j] += int_step
            Q[:, j] = (
                W_int[:, j].to(s_j.dtype) - W_zero[:, j].to(s_j.dtype)
            ) * s_j
            streak = 0

        total_improved += sweep_improved
        if sweep_improved == 0:
            break
    return Q, W_int, total_improved


# ---------------------------------------------------------------------------
#  True objective for end-to-end monotonicity check.
# ---------------------------------------------------------------------------


def compute_L_eff_total(
    W_orig: torch.Tensor,
    Q: torch.Tensor,
    H_orig: torch.Tensor,
    dXXT: Optional[torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Return per-row L_eff = e^T H e + 2*beta * e . WB, shape [rows]."""
    E = W_orig - Q
    L = ((E @ H_orig) * E).sum(dim=1)
    if dXXT is not None and beta != 0.0:
        WB = W_orig @ dXXT
        L = L + 2.0 * beta * (E * WB).sum(dim=1)
    return L


# ---------------------------------------------------------------------------
#  Case factory.
# ---------------------------------------------------------------------------


def build_case(
    rows: int,
    cols: int,
    bits: int,
    sym: bool,
    *,
    with_dxxt: bool,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Random (W_orig, Q, W_int, Scale, W_zero, H_orig, dXXT)."""
    g = torch.Generator(device=device).manual_seed(seed)
    W_orig = torch.randn(rows, cols, generator=g, device=device, dtype=dtype)

    qz = WeightQuantizer(W_orig.shape)
    qz.configure(bits=bits, perchannel=True, sym=sym, mse=False)
    W_in = W_orig.clone().float()
    qz.find_params(W_in)
    Q_ref_fp, W_int_ref, scale_ref = qz.fake_quantize(W_in)

    Q = Q_ref_fp.to(dtype).clone()
    W_int = W_int_ref.to(dtype).clone()

    Scale = scale_ref.to(device=device, dtype=dtype)
    if Scale.shape != Q.shape:
        Scale = Scale.expand_as(Q).clone()
    zero_ref = qz.zero.to(device=device, dtype=dtype)
    if zero_ref.shape != Q.shape:
        zero_ref = zero_ref.expand_as(Q).clone()
    W_zero = zero_ref

    # Hessian: H = X X^T (PSD) with some heterogeneous diagonal.
    n_tok = max(cols * 4, 64)
    X = torch.randn(cols, n_tok, generator=g, device=device, dtype=dtype)
    H_orig = X @ X.t() / n_tok
    # Keep symmetry exact for the matmul-invariants.
    H_orig = 0.5 * (H_orig + H_orig.t())

    if with_dxxt:
        Xq = X + 0.1 * torch.randn(
            cols, n_tok, generator=g, device=device, dtype=dtype
        )
        dXXT = ((X - Xq) @ Xq.t()) / n_tok
    else:
        dXXT = None

    maxq = int(qz.maxq.item())
    if sym:
        min_int, max_int = -(maxq + 1), maxq
    else:
        min_int, max_int = 0, maxq

    return {
        "W_orig": W_orig.to(dtype),
        "Q": Q,
        "W_int": W_int,
        "Scale": Scale,
        "W_zero": W_zero,
        "H_orig": H_orig,
        "dXXT": dXXT,
        "min_int": min_int,
        "max_int": max_int,
    }


# ---------------------------------------------------------------------------
#  Single-case runner.
# ---------------------------------------------------------------------------


def run_case(
    name: str,
    *,
    rows: int,
    cols: int,
    bits: int,
    sym: bool,
    with_dxxt: bool,
    cfg: ReQuantConfig,
    strict: bool,
    seed: int = 0,
    atol_Q: float = 0.0,
) -> bool:
    data = build_case(rows, cols, bits, sym, with_dxxt=with_dxxt, seed=seed)
    W_orig = data["W_orig"]
    Q0 = data["Q"].clone()
    W_int0 = data["W_int"].clone()

    # Production.
    Q_opt, W_int_opt, n_opt = requant_from_config(
        W_orig,
        data["Q"].clone(),
        data["W_int"].clone(),
        data["Scale"],
        data["W_zero"],
        data["H_orig"],
        data["dXXT"],
        data["min_int"],
        data["max_int"],
        cfg,
    )

    # Reference.
    Q_ref, W_int_ref, n_ref = reference_requant_from_config(
        W_orig,
        data["Q"].clone(),
        data["W_int"].clone(),
        data["Scale"],
        data["W_zero"],
        data["H_orig"],
        data["dXXT"],
        data["min_int"],
        data["max_int"],
        cfg,
    )

    beta = float(cfg.beta)
    L0 = compute_L_eff_total(W_orig, Q0, data["H_orig"], data["dXXT"], beta)
    L_opt = compute_L_eff_total(W_orig, Q_opt, data["H_orig"], data["dXXT"], beta)
    L_ref = compute_L_eff_total(W_orig, Q_ref, data["H_orig"], data["dXXT"], beta)

    Q_diff = (Q_opt - Q_ref).abs().max().item()
    Wi_diff = (W_int_opt - W_int_ref).abs().max().item()
    drop_opt = (L0 - L_opt).sum().item()
    drop_ref = (L0 - L_ref).sum().item()
    mono_ok = torch.all(L_opt <= L0 + 1e-9).item() and torch.all(
        L_ref <= L0 + 1e-9
    ).item()

    ok_match = (Q_diff <= atol_Q) and (Wi_diff == 0) and (n_opt == n_ref)
    tag = "OK" if (ok_match and mono_ok) else "FAIL"
    strict_req = " [strict]" if strict else ""
    print(
        f"[{tag}] {name}{strict_req}\n"
        f"    rows={rows} cols={cols} bits={bits} sym={sym} "
        f"dxxt={with_dxxt} beta={cfg.beta} lam={cfg.lambda_anchor} "
        f"eps={cfg.min_gain_eps}\n"
        f"    n_updates: prod={n_opt} ref={n_ref}\n"
        f"    max|Q_prod - Q_ref|   = {Q_diff:.3e}\n"
        f"    max|Wi_prod - Wi_ref| = {Wi_diff:.3e}\n"
        f"    sum L drop (prod)     = {drop_opt:.6e}\n"
        f"    sum L drop (ref)      = {drop_ref:.6e}\n"
        f"    monotonic L_opt<=L0   = {mono_ok}\n"
    )
    if strict and not ok_match:
        return False
    if not mono_ok:
        return False
    return True


# ---------------------------------------------------------------------------
#  Main.
# ---------------------------------------------------------------------------


def main() -> int:
    torch.manual_seed(0)
    base = dict(num_sweeps=4, num_candidates=2, beta=0.3, lambda_anchor=0.0)

    results = []

    # --- Strict: eps=0 should give byte-identical Q/W_int vs reference. ---
    cfg_strict = ReQuantConfig(
        **base, min_gain_eps=0.0, early_stop_consecutive_cols=0
    )
    results.append(
        run_case(
            "sym, no-dXXT, no-anchor",
            rows=3, cols=8, bits=4, sym=True, with_dxxt=False,
            cfg=cfg_strict, strict=True, seed=1,
        )
    )
    results.append(
        run_case(
            "asym, no-dXXT, no-anchor",
            rows=3, cols=8, bits=4, sym=False, with_dxxt=False,
            cfg=cfg_strict, strict=True, seed=2,
        )
    )
    results.append(
        run_case(
            "sym, dXXT (beta=0.3), no-anchor",
            rows=4, cols=16, bits=4, sym=True, with_dxxt=True,
            cfg=cfg_strict, strict=True, seed=3,
        )
    )
    results.append(
        run_case(
            "asym, dXXT (beta=0.3), no-anchor",
            rows=4, cols=16, bits=4, sym=False, with_dxxt=True,
            cfg=cfg_strict, strict=True, seed=4,
        )
    )
    # --- With anchor (strict still OK because eps=0). ---
    cfg_anchor = ReQuantConfig(
        num_sweeps=4, num_candidates=2, beta=0.3, lambda_anchor=0.05,
        min_gain_eps=0.0, early_stop_consecutive_cols=0,
    )
    results.append(
        run_case(
            "sym, dXXT, anchor=0.05",
            rows=4, cols=12, bits=4, sym=True, with_dxxt=True,
            cfg=cfg_anchor, strict=True, seed=5,
        )
    )
    # --- Non-zero eps: production L_row drift may disagree with the
    # reference's freshly-computed L_row at the threshold boundary, so we
    # only require monotonic descent, not byte-identical Q.
    cfg_eps = ReQuantConfig(
        num_sweeps=4, num_candidates=2, beta=0.3, lambda_anchor=0.0,
        min_gain_eps=0.01, early_stop_consecutive_cols=0,
    )
    results.append(
        run_case(
            "sym, dXXT, eps=0.01 (monotonic only)",
            rows=4, cols=16, bits=4, sym=True, with_dxxt=True,
            cfg=cfg_eps, strict=False, seed=6,
        )
    )

    # --- Bigger case: shape closer to a real layer column block. ---
    results.append(
        run_case(
            "sym, dXXT, rows=32 cols=64",
            rows=32, cols=64, bits=4, sym=True, with_dxxt=True,
            cfg=cfg_strict, strict=True, seed=7,
        )
    )

    ok = all(results)
    print("=" * 60)
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
