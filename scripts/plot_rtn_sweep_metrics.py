#!/usr/bin/env python3
"""Plot RTN ReQuant wikitext2 KL/PPL metrics versus sweep count."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SWEEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FP_WIKITEXT2_PPL = 6.14
GPTAQ_WIKITEXT2_PPL = 6.44
GPTAQ_WIKITEXT2_KL = 0.0433
# Green dashed baseline: GPTAQ + QuaRot @ WikiText2 (PPL/KL reference lines).
GPTAQ_BASELINE_LEGEND = "GPTAQ + QuaRot (W4A16)"
# Acc figure only: GPTAQ + QuaRot reference (average accuracy).
GPTAQ_ROT_ACC = 65.22
GPTAQ_ROT_ACC_LEGEND = "GPTAQ + QuaRot (W4A16)"

# Use None for sweeps that have not been evaluated yet; they will be skipped in plots.
SERIES = [
    {
        "label": "RTN + ReQuant (W4A16)",
        "ppl": [7.73, 6.84, 6.74, 6.72, 6.71, 6.70, 6.69, 6.69, 6.68],
        "kl": [1.89e-1, 9.64e-2, 8.57e-2, 8.28e-2, 8.04e-2, 7.87e-2, 7.88e-2, 7.74e-2, 7.70e-2],
        "acc": [61.65, 63.56, 63.45, 63.45, 64.49, 64.14, 64.40, 63.98, 63.88],
        "color": "#2c66a0",
        "marker": "o",
    },
    {
        "label": "RTN + QuaRot + ReQuant (W4A16)",
        "ppl": [7.58, 6.53, 6.49, 6.47, 6.46, 6.45, 6.45, 6.45, 6.45],
        "kl": [1.68e-1, 5.54e-2, 4.95e-2, 4.81e-2, 4.63e-2, 4.57e-2, 4.47e-2, 4.48e-2, 4.43e-2],
        "acc": [61.78, 64.54, 64.49, 64.76, 65.04, 64.90, 65.01, 65.22, 65.31],
        "color": "#e07b2c",
        "marker": "s",
    },
    {
        "label": "GPTAQ + QuaRot + ReQuant (W4A4)",
        "ppl": [7.41, 7.35, 7.36, 7.34, 7.32, 7.31, 7.30, 7.32, 7.32],
        "kl": [1.60e-1, 1.56e-1, 1.58e-1, 1.56e-1, 1.54e-1, 1.55e-1, 1.54e-1, 1.54e-1, 1.55e-1],
        "acc": [61.97, 62.17, 61.75, 61.99, 62.40, 62.23, 62.73, 62.38, 62.47],
        "color": "#834db8",
        "marker": "^",
    },
]

PALETTE = {
    "fp": "#c84b3a",
    "gptaq": "#1f8a3f",
    "text": "#1c2530",
    "muted": "#9aa0a6",
    "grid_major": "#e3e6ea",
    "grid_minor": "#f1f3f5",
    "panel": "#ffffff",
    "legend_face": "#ffffff",
    "legend_edge": "#cfd2d6",
}

# Slightly compact legend so it does not cover curves (used by PPL / KL / acc).
LEGEND_FONTSIZE = 8.75


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
                "STIXGeneral",
                "serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.linewidth": 1.1,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.frameon": True,
            "legend.facecolor": PALETTE["legend_face"],
            "legend.edgecolor": PALETTE["legend_edge"],
            "legend.framealpha": 0.92,
            "legend.borderpad": 0.38,
            "legend.handlelength": 1.75,
            "legend.handletextpad": 0.45,
            "legend.labelspacing": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.set_xticks(SWEEPS)
    ax.set_xlim(SWEEPS[0] - 0.3, SWEEPS[-1] + 0.3)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(True, axis="y", which="major", color=PALETTE["grid_major"], linewidth=0.9, zorder=0)
    ax.grid(True, axis="y", which="minor", color=PALETTE["grid_minor"], linewidth=0.6, zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", which="major", length=4.5, color=PALETTE["muted"])
    ax.tick_params(axis="y", which="minor", length=2.5, color=PALETTE["muted"])
    ax.spines["left"].set_color(PALETTE["muted"])
    ax.spines["bottom"].set_color(PALETTE["muted"])


def plot_metric(ax, metric_key, ylabel):
    y_min = float("inf")
    y_max = float("-inf")
    drew_any = False

    for series in SERIES:
        values = series.get(metric_key)
        if values is None:
            continue
        xs = [s for s, v in zip(SWEEPS, values) if v is not None]
        ys = [v for v in values if v is not None]
        if not ys:
            continue
        y_min = min(y_min, min(ys))
        y_max = max(y_max, max(ys))
        drew_any = True

        # Soft halo behind the line for a polished look.
        ax.plot(
            xs,
            ys,
            color=series["color"],
            linewidth=4.5,
            alpha=0.18,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            label=series["label"],
            marker=series["marker"],
            markersize=7.5,
            linewidth=2.2,
            color=series["color"],
            markerfacecolor=series["color"],
            markeredgecolor="white",
            markeredgewidth=1.4,
            solid_capstyle="round",
            zorder=3,
        )

    ax.set_xlabel("ReQuant sweeps")
    ax.set_ylabel(ylabel)
    style_axes(ax)
    if not drew_any:
        return None, None
    return y_min, y_max


def mark_best_point(ax, metric_key, value_format):
    series = SERIES[0]
    values = series[metric_key]
    best_idx = min(range(len(values)), key=values.__getitem__)
    best_x = SWEEPS[best_idx]
    best_y = values[best_idx]

    ax.scatter(
        best_x,
        best_y,
        s=240,
        facecolors=PALETTE["highlight"],
        edgecolors=series["color"],
        linewidths=2.2,
        zorder=4,
        alpha=0.95,
    )
    ax.scatter(
        best_x,
        best_y,
        s=60,
        facecolors=series["color"],
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )

    label = f"best @ s={best_x}\n{value_format.format(best_y)}"
    offset_x = -1.6 if best_x > SWEEPS[len(SWEEPS) // 2] else 1.4
    offset_y_ratio = 0.18

    span = max(values) - min(values)
    offset_y = span * offset_y_ratio if span > 0 else 0.05

    ax.annotate(
        label,
        xy=(best_x, best_y),
        xytext=(best_x + offset_x, best_y + offset_y),
        fontsize=10,
        color=PALETTE["text"],
        ha="center",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "color": PALETTE["muted"],
            "lw": 1.0,
            "shrinkA": 4,
            "shrinkB": 6,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": PALETTE["muted"],
            "lw": 0.8,
            "alpha": 0.95,
        },
    )


def add_fp_ppl_baseline(ax):
    ax.axhline(
        FP_WIKITEXT2_PPL,
        color=PALETTE["fp"],
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        alpha=0.9,
        label="Full precision",
        zorder=1.5,
    )


def add_gptaq_baseline(ax, value):
    ax.axhline(
        value,
        color=PALETTE["gptaq"],
        linestyle=(0, (1.5, 2)),
        linewidth=1.6,
        alpha=0.9,
        label=GPTAQ_BASELINE_LEGEND,
        zorder=1.5,
    )


def add_gptaq_rot_acc_baseline(ax):
    ax.axhline(
        GPTAQ_ROT_ACC,
        color=PALETTE["gptaq"],
        linestyle=(0, (1.5, 2)),
        linewidth=1.6,
        alpha=0.9,
        label=GPTAQ_ROT_ACC_LEGEND,
        zorder=1.5,
    )


def add_panel_tag(ax, tag):
    ax.text(
        0.015,
        0.96,
        tag,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=PALETTE["text"],
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": PALETTE["muted"],
            "lw": 0.8,
            "alpha": 0.95,
        },
    )


def save_figure(fig, output_path):
    pdf_output_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {output_path}")
    print(f"Saved plot to {pdf_output_path}")


def _styled_legend(ax, loc, ncol=1, bbox_to_anchor=None):
    kw = dict(
        loc=loc,
        ncol=ncol,
        borderaxespad=0.45,
        fontsize=LEGEND_FONTSIZE,
        columnspacing=0.85 if ncol > 1 else 0.9,
        fancybox=False,
    )
    if bbox_to_anchor is not None:
        kw["bbox_to_anchor"] = bbox_to_anchor
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(0.75)
    return leg


def make_ppl_figure(output_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ppl_min, ppl_max = plot_metric(ax, "ppl", "WikiText2 Perplexity")
    add_fp_ppl_baseline(ax)
    add_gptaq_baseline(ax, GPTAQ_WIKITEXT2_PPL)
    lower_ref = min(FP_WIKITEXT2_PPL, GPTAQ_WIKITEXT2_PPL, ppl_min)
    upper_ref = max(GPTAQ_WIKITEXT2_PPL, ppl_max)
    span_ppl = upper_ref - lower_ref
    ax.set_ylim(
        lower_ref - span_ppl * 0.08,
        upper_ref + span_ppl * 0.20,
    )
    _styled_legend(ax, "upper right")

    save_figure(fig, output_dir / "rtn_requant_sweep_wikitext2_ppl.png")
    plt.close(fig)


def make_kl_figure(output_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    kl_min, kl_max = plot_metric(ax, "kl", "WikiText2 KL Divergence")
    add_gptaq_baseline(ax, GPTAQ_WIKITEXT2_KL)
    lower_ref = min(kl_min, GPTAQ_WIKITEXT2_KL)
    upper_ref = max(kl_max, GPTAQ_WIKITEXT2_KL)
    span_kl = upper_ref - lower_ref
    ax.set_ylim(
        lower_ref - span_kl * 0.10,
        upper_ref + span_kl * 0.22,
    )
    _styled_legend(ax, "upper right")

    save_figure(fig, output_dir / "rtn_requant_sweep_wikitext2_kl.png")
    plt.close(fig)


def make_acc_figure(output_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    acc_min, acc_max = plot_metric(ax, "acc", "Average Accuracy (%)")
    if acc_min is None:
        plt.close(fig)
        return
    add_gptaq_rot_acc_baseline(ax)
    ref_min = min(acc_min, GPTAQ_ROT_ACC)
    ref_max = max(acc_max, GPTAQ_ROT_ACC)
    span_acc = max(ref_max - ref_min, 1.0)
    ax.set_ylim(
        ref_min - span_acc * 0.18,
        ref_max + span_acc * 0.22,
    )
    _styled_legend(ax, "lower right", ncol=2)

    save_figure(fig, output_dir / "rtn_requant_sweep_acc_avg.png")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "outputs/Meta-Llama-3-8B/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    make_ppl_figure(output_dir)
    make_kl_figure(output_dir)
    make_acc_figure(output_dir)
    print(f"[plot_rtn_sweep_metrics] wrote plots under {output_dir}")


if __name__ == "__main__":
    main()
