"""Markdown tables and figures from results CSVs (never from in-memory numbers)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE

# Headline metric per protocol (and the subset it is read from).
HEADLINE = {
    "hard": ("all", "roc_auc"),
    "rank_hard": ("full_candidates", "mrr"),
    "rank_random": ("all", "mrr"),
    "random": ("all", "roc_auc"),
    "diagnostic_all_vs_hard": ("all", "roc_auc"),
    "hard_active": ("all", "roc_auc"),
    "rank_hard_active": ("full_candidates", "mrr"),
}
FIG_TITLES = {
    "hard_active": "hard_active (lead, pair-level)\ndistance-2 positives vs activity-matched hard negatives",
    "rank_hard_active": "rank_hard_active\nvs ≤100 active distance-2 candidates (full queries)",
    "hard": "hard (headline)\ndistance-2 positives vs hard negatives",
    "rank_hard": "rank_hard\nvs ≤100 distance-2 candidates (full queries)",
    "rank_random": "rank_random\nvs 100 random candidates",
    "random": "random (easy)\nall positives vs random negatives",
}
PROTOCOL_TITLES = {
    "hard_active": "hard_active: distance-2 positives vs activity-matched hard negatives (lead)",
    "rank_hard_active": "rank_hard_active: vs ≤100 active distance-2 candidates",
    "hard": "hard: distance-2 positives vs hard negatives (headline)",
    "rank_hard": "rank_hard: distance-2 positives vs ≤100 distance-2 candidates",
    "rank_random": "rank_random: each positive vs 100 random candidates",
    "random": "random: all positives vs random negatives (easy)",
}
DIAGNOSTIC_WARNING = (
    "> **Warning — diagnostic only, not a result.** All hard negatives are at distance 2, while "
    "34–41% of positives are at distance ≥ 3, so in this comparison distance alone separates the "
    "classes. A learned model that picks up distance (or anything correlated with it) can look good "
    "here for the wrong reason; neighbourhood heuristics are pushed the other way, because they score "
    "every distance-3+ positive 0, below every hard negative. Neither direction says anything about "
    "the task. Use `hard` instead."
)
# Expected MRR of a random ranker with 100 candidates (rank uniform on 1..101).
RANDOM_MRR_101 = float(np.sum(1.0 / np.arange(1, 102)) / 101)


def fmt(v, lo, hi, digits=3) -> str:
    if pd.isna(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def metric_table(df: pd.DataFrame, protocol: str, subset: str, split: str, metrics: list[str],
                 models: list[str]) -> pd.DataFrame:
    d = df[(df["protocol"] == protocol) & (df["subset"] == subset) & (df["split"] == split)]
    out = pd.DataFrame(index=models)
    for m in metrics:
        x = d[d["metric"] == m].set_index("model")
        out[m] = [fmt(*x.loc[k, ["value", "ci_low", "ci_high"]]) if k in x.index else "" for k in models]
    out.index.name = "model"
    return out.reset_index()


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def _style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=9.5, pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _bars(ax, d: pd.DataFrame, models: list[str], color: str, ref: float | None, xlim=(0, 1)):
    d = d.set_index("model").reindex(models)
    y = np.arange(len(models))[::-1]
    ax.barh(y, d["value"], color=color, height=0.6)
    err = np.vstack([d["value"] - d["ci_low"], d["ci_high"] - d["value"]])
    ax.errorbar(d["value"], y, xerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    # Label just past the upper CI bound so it never collides with the whisker.
    for yi, v, hi in zip(y, d["value"], d["ci_high"].fillna(d["value"])):
        ax.text(hi + 0.015 * (xlim[1] - xlim[0]), yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(*xlim)
    if ref is not None:
        ax.axvline(ref, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))


PANELS = {
    "hard_active": ("ROC-AUC", 0.5, (0, 1)), "rank_hard_active": ("MRR (full-candidate queries)", None, None),
    "hard": ("ROC-AUC", 0.5, (0, 1)), "rank_hard": ("MRR (full-candidate queries)", None, None),
    "rank_random": ("MRR", RANDOM_MRR_101, None), "random": ("ROC-AUC", 0.5, (0, 1)),
}


def protocol_figure(df: pd.DataFrame, split: str, models: list[str], path: Path, note: str = "",
                    protocols=("hard", "rank_hard", "rank_random", "random")) -> None:
    panels = [(p, *PANELS[p]) for p in protocols]
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.2 * rows), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (p, label, ref, xlim) in zip(axes.ravel(), panels):
        subset, metric = HEADLINE[p]
        d = df[(df["protocol"] == p) & (df["subset"] == subset) & (df["metric"] == metric) & (df["split"] == split)]
        lim = (xlim[0], xlim[1] * 1.12) if xlim else (0, max(0.05, float(d["ci_high"].max()) * 1.3))
        _bars(ax, d, models, SERIES[0], ref, lim)
        _style(ax, FIG_TITLES[p], label)
    fig.text(0.01, 0.005, f"{split} split; bars = point estimate, whiskers = bootstrap 95% CI; dashed = random ranker.{note}",
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def breakdown_figure(df: pd.DataFrame, split: str, models: list[str], path: Path) -> None:
    subsets = ["distance=2", "distance=3", "distance=4+", "disconnected", "cold_start", "established"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    d0 = df[(df["protocol"] == "random") & (df["metric"] == "roc_auc") & (df["split"] == split)]
    n = df[(df["protocol"] == "random") & (df["metric"] == "n_positives") & (df["split"] == split)]
    for ax, sub in zip(axes.ravel(), subsets):
        d = d0[d0["subset"] == sub]
        npos = int(n.loc[n["subset"] == sub, "value"].iloc[0]) if (n["subset"] == sub).any() else 0
        color = SERIES[0] if sub.startswith(("distance", "disc")) else SERIES[1]
        _bars(ax, d, models, color, 0.5, (0, 1.15))
        _style(ax, f"{sub}\nn = {npos:,} positives", "ROC-AUC vs random negatives")
    fig.text(0.01, 0.005, f"{split} split, random protocol; positives restricted to the stratum, all random negatives "
             "kept. Dashed = 0.5.", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
