"""Markdown tables and figures from results CSVs (never from in-memory numbers)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE

# Headline metric per protocol (and the subset it is read from). From 2026-10-15 (audit M2) the
# pair-level protocols are headlined by the MACRO of the two within-group (group-matched) scores.
HEADLINE = {
    "hard": ("macro", "roc_auc"),
    "rank_hard": ("macro", "query_auc"),
    "rank_random": ("all", "mrr"),
    "random": ("all", "roc_auc"),
    "diagnostic_all_vs_hard": ("all", "roc_auc"),
    "hard_active": ("macro", "roc_auc"),
    "rank_hard_active": ("macro", "query_auc"),
}
# The pooled metrics the earlier SELECTION rules were defined on (negative mix, LightGBM / pair-focused
# search, node2vec and GNN grids). Kept so that changing the headline never silently changes a past choice.
POOLED = {
    "hard": ("all", "roc_auc"), "rank_hard": ("full_candidates", "mrr"), "rank_random": ("all", "mrr"),
    "random": ("all", "roc_auc"), "diagnostic_all_vs_hard": ("all", "roc_auc"),
    "hard_active": ("all", "roc_auc"), "rank_hard_active": ("full_candidates", "mrr"),
}
REFERENCE_MODELS = {"prefer_newer"}    # reference rows, never eligible as "best heuristic"
POOLED_NOTE = "pooled — affected by newcomer imbalance"


def val_best(base: pd.DataFrame, protocols, headline: dict | None = None) -> dict:
    """Validation-best heuristic per protocol by its headline metric (reference models excluded)."""
    headline = headline or HEADLINE
    out = {}
    for p in protocols:
        subset, metric = headline[p]
        d = base[(base.split == "val") & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)
                 & ~base.model.isin(REFERENCE_MODELS)]
        out[p] = d.sort_values(["value", "model"], ascending=[False, True])["model"].iloc[0]
    return out


FIG_TITLES = {
    "hard_active": "hard_active, macro (lead)\nmean of within-group ROC-AUCs",
    "rank_hard_active": "rank_hard_active, macro\nmean of within-group query AUCs",
    "hard": "hard, macro (includes activity)\nmean of within-group ROC-AUCs",
    "rank_hard": "rank_hard, macro (includes activity)\nmean of within-group query AUCs",
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
    "hard_active": ("macro ROC-AUC", 0.5, (0, 1)), "rank_hard_active": ("macro query AUC", 0.5, (0, 1)),
    "hard": ("macro ROC-AUC", 0.5, (0, 1)), "rank_hard": ("macro query AUC", 0.5, (0, 1)),
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
    fig.tight_layout(rect=(0, 0.035 if note else 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def breakdown_figure(df: pd.DataFrame, split: str, models: list[str], path: Path) -> None:
    # Cold-start panels removed (audit M2: positives-only strata); group-matched results are in the tables.
    subsets = ["distance=2", "distance=3", "distance=4+", "disconnected"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), dpi=150, sharex=True)
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


# ---------------------------------------------------------------- shared Markdown sections (audit M2)

PAIR_LEVEL = ("hard_active", "rank_hard_active", "hard", "rank_hard")
RANKING = {"rank_hard", "rank_hard_active"}
TAGS = {"hard_active": "lead pair-level protocol", "rank_hard_active": "lead ranking protocol",
        "hard": "includes activity", "rank_hard": "includes activity"}


def pair_level_md(get, models: list[str], split: str, ks_hard=(10, 20, 50), protocols=PAIR_LEVEL) -> list[str]:
    """get(model, split, protocol, subset, metric) -> formatted cell ('' if missing)."""
    out = []
    for p in protocols:
        main, second = ("query_auc", "mrr") if p in RANKING else ("roc_auc", "average_precision")
        t = pd.DataFrame({"model": models})
        t[f"macro {main} (headline)"] = [get(m, split, p, "macro", main) for m in models]
        t[f"established {main}"] = [get(m, split, p, "established_matched", main) for m in models]
        t[f"newcomer {main}"] = [get(m, split, p, "newcomer_matched", main) for m in models]
        t[f"macro {second}"] = [get(m, split, p, "macro", second) for m in models]
        note = (" Ranking: candidates are restricted to the query's group, so candidate counts vary; query AUC "
                "(count-independent) is the headline, MRR is secondary." if p in RANKING else "")
        out += [f"### {p} — within-group, macro ({TAGS[p]})", "",
                "Positives and negatives/candidates from the same group (established: no author with ≤ 2 prior "
                "works; newcomer: at least one). Macro = mean of the two." + note, "", to_md(t), ""]
        if p in RANKING:
            pm = [f"hits@{k}" for k in ks_hard] + ["mrr", "query_auc"]
            sub = "full_candidates"
        else:
            pm, sub = ["roc_auc", "average_precision"], "all"
        t = pd.DataFrame({"model": models})
        for m_ in pm:
            t[m_] = [get(m, split, p, sub, m_) for m in models]
        out += [f"#### {p} — pooled ({sub}; {POOLED_NOTE})", "", to_md(t), ""]
    return out


def positives_only_appendix(get, models: list[str], split: str, protocols=PAIR_LEVEL) -> list[str]:
    rows = []
    for p in protocols:
        metric = "query_auc" if p in RANKING else "roc_auc"
        for sub in ("cold_start", "established"):
            rows.append({"protocol / subset": f"{p} {metric} — {sub} (positives only)",
                         **{m: get(m, split, p, sub, metric) for m in models}})
    return ["### Superseded: cold-start breakdown with positives-only strata", "",
            "> Kept for the record. These strata restricted only the POSITIVES and kept all negatives, so they "
            "partly measured the newcomer imbalance between positives and negatives (audit M2). Use the "
            "within-group tables above instead.", "", to_md(pd.DataFrame(rows)), ""]
