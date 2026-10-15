"""Phase 9: final figures for the report and README (figures/final/), drawn only from results / report files.

  python scripts/final_figures.py

Style: one width (6.3 in, A4 text width), fonts >= 8 pt, Okabe-Ito colourblind-safe palette, direct labels or legends
so no series is identified by colour alone, test split unless a panel says otherwise.

  fig1_headline.png        macro hard_active / rank_hard_active with 95% CIs, main models (final_comparison.csv,
                           two_stage.csv)
  fig2_activity.png        pooled hard vs macro hard_active per model (the activity confound), and single activity
                           features as scorers (activity_detector.csv)
  fig3_rq2_text.png        RQ2 text ablation of the pair-focused LightGBM by group (rq2_text_ablation.csv)
  fig4_attribution.png     SHAP discrimination by meaning-based feature group, pooled hard vs hard_active
                           (shap_importance.csv)
  fig5_shortcut.png        distance-shortcut check per model and the GNN edge-occlusion test
                           (gnn_shortcut_check.csv, gnn_explanations.csv)
  fig6_calibration.png     reliability diagrams on hard_active pairs (report/calibration/reliability.csv,
                           calibration.csv)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, OUT = Path("results"), Path("figures/final")
W = 6.3
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7", "sky": "#56B4E9",
      "red": "#D55E00", "grey": "#7F7F7F", "black": "#000000"}
INK, INK2, GRID = "#111111", "#555555", "#DDDDDD"
PF, GEN, AA = "lightgbm_pair_focused", "lightgbm", "adamic_adar"
GNN_PF, GNN_GEN = "gnn_pair_focused_sage_structural", "gnn_general_sage_text"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "font.family": "DejaVu Sans"})
LABEL = {"prefer_newer": "prefer-newer (reference)", AA: "Adamic-Adar", "jaccard": "Jaccard", "logreg": "logistic reg.",
         GEN: "general LightGBM", PF: "pair-focused LightGBM", "two_stage": "two-stage", "node2vec": "node2vec",
         GNN_PF: "best GNN (shortcut-trained)", f"{PF}+{GNN_PF}": "hybrid (secondary)"}


def axstyle(ax, title=None, grid="x"):
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def pick(d, **kw):
    x = d
    for k, v in kw.items():
        x = x[x[k] == v]
    if len(x) != 1:
        raise KeyError((kw, len(x)))
    return x.iloc[0]


def note(f, text, y=-0.05):
    """Figure caption wrapped to the A4 text width."""
    f.text(0.0, y, textwrap.fill(text, 120), fontsize=7, color=INK2, va="top")


def save(f, name):
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(f)


def fig1():
    fc, ts = pd.read_csv(RES / "final_comparison.csv"), pd.read_csv(RES / "two_stage.csv")
    order = ["prefer_newer", AA, "jaccard", "node2vec", GNN_PF, "logreg", GEN, "two_stage", PF, f"{PF}+{GNN_PF}"]
    f, axes = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
    for ax, (p, metric, title) in zip(axes, (("hard_active", "roc_auc", "macro hard_active ROC-AUC"),
                                             ("rank_hard_active", "query_auc", "macro rank_hard_active query AUC"))):
        aa = pick(fc, model=AA, split="test", protocol=p, subset="macro", metric=metric).value
        for i, m in enumerate(order):
            r = pick(ts if m == "two_stage" else fc, model=m, split="test", protocol=p, subset="macro", metric=metric)
            col = OI["blue"] if m == PF else (OI["grey"] if m in ("prefer_newer",) else
                                               OI["orange"] if m in (GNN_PF, f"{PF}+{GNN_PF}", "node2vec") else OI["black"])
            ax.errorbar(r.value, i, xerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt="o", color=col,
                        markersize=4, capsize=2, linewidth=1)
            ax.text(r.ci_high + 0.006, i, f"{r.value:.3f}", va="center", fontsize=7, color=INK2)
        ax.axvline(aa, color=OI["grey"], linestyle=":", linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL[m] for m in order])
        ax.set_xlim(0.5, 0.9)
        axstyle(ax, title)
    axes[0].invert_yaxis()
    f.text(0.0, -0.04, "Test split (2025 predictions). Mean of 5 seeds with seed-mean bootstrap 95% CI (5,000 resamples); "
           "dotted line = Adamic-Adar. Blue = headline model; orange = representation-learning models.",
           fontsize=7, color=INK2)
    save(f, "fig1_headline.png")


def fig2():
    fc, ts = pd.read_csv(RES / "final_comparison.csv"), pd.read_csv(RES / "two_stage.csv")
    act = pd.read_csv(RES / "activity_detector.csv")
    models = ["prefer_newer", AA, "logreg", GEN, "two_stage", PF]
    f, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(models))
    for j, (p, sub, lab, col) in enumerate((("hard", "all", "pooled hard (includes activity)", OI["orange"]),
                                            ("hard_active", "macro", "macro hard_active", OI["blue"]))):
        v = [pick(ts if m == "two_stage" else fc, model=m, split="test", protocol=p, subset=sub, metric="roc_auc").value
             for m in models]
        ax.barh(y + (j - 0.5) * 0.38, v, height=0.36, color=col, label=lab, hatch="" if j else "//", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.9)
    ax.axvline(0.5, color=OI["grey"], linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False)
    axstyle(ax, "ROC-AUC by protocol (test)")
    ax = axes[1]
    feats = [("recent_works_min", "recent works (min)"), ("years_since_last_max", "−years since last work (max)")]
    for i, (f_, lab) in enumerate(feats):
        for j, (p, sub, col) in enumerate((("hard", "all", OI["orange"]), ("hard_active", "macro", OI["blue"]))):
            v = pick(act, model=f"single_feature[{f_}]", split="test", protocol=p, subset=sub, metric="roc_auc").value
            ax.barh(i + (j - 0.5) * 0.38, v, height=0.36, color=col, hatch="//" if j == 0 else "", edgecolor="white")
            ax.text(v + 0.01, i + (j - 0.5) * 0.38, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([lab for _, lab in feats])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.9)
    ax.axvline(0.5, color=OI["grey"], linewidth=0.8)
    axstyle(ax, "One activity feature as a score")
    note(f, "Pooled hard negatives are not matched on prediction-year activity; hard_active negatives are. "
           "Hatched = pooled hard.", -0.11)
    f.tight_layout()
    save(f, "fig2_activity.png")


def fig3():
    rq = pd.read_csv(RES / "rq2_text_ablation.csv")
    sets = [("structural", "structural"), ("structural+text", "+ text"), ("all", "+ text + temporal")]
    cols = [OI["grey"], OI["green"], OI["blue"]]
    f, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, (p, metric, title) in zip(axes, (("hard_active", "roc_auc", "hard_active ROC-AUC"),
                                             ("rank_hard_active", "query_auc", "rank_hard_active query AUC"))):
        for gi, (sub, glab) in enumerate((("macro", "macro"), ("established_matched", "established"),
                                          ("newcomer_matched", "newcomer"))):
            for si_, ((lab, short), c) in enumerate(zip(sets, cols)):
                r = pick(rq, model=f"{PF}[{lab}]", split="test", protocol=p, subset=sub, metric=metric)
                x = gi + (si_ - 1) * 0.26
                ax.bar(x, r.value, width=0.24, color=c, label=short if gi == 0 else None)
                ax.errorbar(x, r.value, yerr=[[r.value - r.ci_low], [r.ci_high - r.value]], color=INK, capsize=1.5,
                            linewidth=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["macro", "established", "newcomer"])
        ax.set_ylim(0.65, 0.85)
        axstyle(ax, title, grid="y")
    f.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    note(f, "Pair-focused LightGBM, same parameters for every feature set, 5 seeds, test split; "
           "seed-mean 95% CIs.", -0.1)
    f.tight_layout()
    save(f, "fig3_rq2_text.png")


def fig4():
    si = pd.read_csv(RES / "shap_importance.csv")
    groups = [("time_weighted_structure", "time-weighted\nstructure"), ("text_similarity", "text\nsimilarity"),
              ("pure_structure", "pure\nstructure"), ("author_activity_career", "author activity\n/ career")]
    f, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    for ax, (m, title) in zip(axes, ((GEN, "general LightGBM"), (PF, "pair-focused LightGBM (headline)"))):
        for j, (p, lab, col, h) in enumerate((("hard", "pooled hard", OI["orange"], "//"),
                                              ("hard_active", "hard_active", OI["blue"], ""))):
            v = [pick(si, model=m, split="test", protocol=p, subset=f"meaning={g}", metric="shap_auc").value for g, _ in groups]
            x = np.arange(len(groups)) + (j - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=col, hatch=h, edgecolor="white", label=lab)
        ax.axhline(0.5, color=OI["grey"], linewidth=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for _, g in groups], fontsize=7)
        ax.set_ylim(0.25, 0.9)
        axstyle(ax, title, grid="y")
    axes[0].set_ylabel("ROC-AUC of the group's SHAP sum")
    axes[0].legend(frameon=False, loc="upper right")
    note(f, "SHAP discrimination: each feature group's summed SHAP contribution used as a score inside the "
           "pair set (test, seed-0 models). Below 0.5 = the attribution points the wrong way.", -0.06)
    f.tight_layout()
    save(f, "fig4_attribution.png")


def fig5():
    gs, ge = pd.read_csv(RES / "gnn_shortcut_check.csv"), pd.read_csv(RES / "gnn_explanations.csv")
    sc = gs[gs.subset == "far_negatives_vs_distance2_positives"].groupby("model").value.agg(["mean", "std"])
    models = [(f"{PF}_shortcut", "pair-focused LightGBM,\nbefore fix"), (PF, "pair-focused LightGBM,\nafter fix"),
              (GNN_PF, "pair-focused GNN\n(not retrained)"), ("gnn_general_sage_text", "general GNN")]
    f, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, (m, lab) in enumerate(models):
        v = sc.loc[m, "mean"]
        ax.barh(i, v, color=OI["red"] if v > 0.5 else OI["blue"], height=0.6)
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    ax.axvline(0.5, color=OI["grey"], linewidth=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([lab for _, lab in models], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    axstyle(ax, "Shortcut check (test)")
    ax.set_xlabel("ROC-AUC, far negatives vs distance-2 positives")
    ax = axes[1]
    cases = [("true_positive", "true pos."), ("false_positive", "false pos."), ("missed_positive", "missed pos.")]
    for j, (m, lab, col, mk) in enumerate(((GNN_PF, "pair-focused GNN", OI["red"], "o"),
                                           (GNN_GEN, "general GNN", OI["blue"], "s"))):
        for ci, (c, _) in enumerate(cases):
            for k in (0, 1):
                v = pick(ge, model=m, subset=f"case={c}|k={k}", metric="logit_change_without_common_neighbour_paths").value
                ax.scatter(ci + (j - 0.5) * 0.25 + (k - 0.5) * 0.08, v, color=col, marker=mk, s=22,
                           label=lab if ci == 0 and k == 0 else None)
    ax.axhline(0, color=OI["grey"], linewidth=0.8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c for _, c in cases])
    axstyle(ax, "Remove shared-neighbour paths (val)", grid="y")
    ax.set_ylabel("change in logit")
    ax.legend(frameon=False, loc="upper right")
    note(f, "Left: > 0.5 means far non-collaborators are ranked above near collaborators. Right: edge occlusion "
           "on 2 validation pairs per case; above 0 = the shared co-author lowered the score.", -0.06)
    f.tight_layout()
    save(f, "fig5_shortcut.png")


def fig6():
    rel = pd.read_csv(Path("report/calibration/reliability.csv"))
    cal = pd.read_csv(RES / "calibration.csv")
    f, ax = plt.subplots(figsize=(W * 0.55, 2.8))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for m, lab, col, mk in ((PF, "pair-focused LightGBM", OI["blue"], "o"), (GEN, "general LightGBM", OI["orange"], "s")):
        d = rel[(rel.model == m) & (rel.split == "test") & (rel.set == "hard_active")]
        b = pick(cal, model=m, split="test", protocol="hard_active", subset="all", metric="brier").value
        ax.plot(d.mean_pred, d.obs_rate, marker=mk, markersize=3.5, color=col, linewidth=1.2, label=f"{lab} (Brier {b:.2f})")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed positive rate")
    ax.legend(frameon=False, loc="upper left")
    axstyle(ax, "Reliability on hard_active pairs (test)", grid="both")
    save(f, "fig6_calibration.png")


def main():
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
