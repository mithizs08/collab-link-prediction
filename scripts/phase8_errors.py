"""Phase 8, step 5: error analysis of the headline model (pair-focused LightGBM) within the newcomer groups.

  python scripts/phase8_errors.py --split val     # validation first (rule 8)
  python scripts/phase8_errors.py --split test    # final numbers

Scores: seed-averaged pair-focused LightGBM (5 seeds, main parameters) on the split's hard_active pairs
(distance-2 positives vs activity-matched distance-2 negatives); general LightGBM and Adamic-Adar for comparison.
Groups: established / newcomer (an author with <= 2 works before the cutoff), as in the headline protocol.

1. Top 100 false positives (highest-scored negatives) and top 100 false negatives (lowest-scored positives) per group,
   profiled against all positives / negatives of the group: degree, career stage, recent output, topic similarity,
   shared institutions, common neighbours (UTS and external), recency-weighted Adamic-Adar, cross-field share.
2. Cross-discipline: author field = modal OpenAlex field (primary topic, scripts/extract_work_fields.py) of the author's
   works published before the cutoff (rule 1; ties -> smallest field id); a pair is same-field / cross-field when both
   authors have a field, `unknown` otherwise. ROC-AUC within same-field and within cross-field pairs.
3. Hypotheses (exploratory, raw p, not in the Holm family): H1 false positives sit in dense, recent neighbourhoods
   (more common neighbours than true positives); H2 missed positives are topically distant (lower text similarity,
   more cross-field than other positives); H3 performance drops with common-neighbour count.
4. Right-censoring: of the top false positives (prediction year Y), how many co-author (any UTS joint work, any team
   size) in a later year present in the data, compared with all negatives of the group. Limitation, not a win.
5. Newcomers: established vs newcomer group scores of every model in results/final_comparison.csv (no new model run).

Outputs: results/error_analysis.csv, figures/error_analysis_<split>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from linkpred.data.splits import all_uts_coauthor_pairs, load_tables  # noqa: E402
from linkpred.eval import bootstrap as bs  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.eval.protocols import newcomer_pairs  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from phase8_shap import other_split  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

PF, GEN, AA = "lightgbm_pair_focused", "lightgbm", "adamic_adar"
PROFILE = ["deg_min", "deg_max", "career_age_min", "career_age_max", "recent_works_min", "text_cos",
           "nbr_text_sim_max", "shared_institutions", "common_neighbors", "cn_uts", "cn_non_uts", "rw_aa"]
TOP = 100


def row(model, split, subset, metric, value, lo=None, hi=None, protocol="hard_active"):
    return {"model": model, "split": split, "protocol": protocol, "subset": subset,
            "negative_type": "hard_active" if protocol.endswith("active") else "hard", "seed": -1,
            "metric": metric, "value": float(value), "ci_low": lo, "ci_high": hi}


def author_fields(processed: Path, cutoff: int) -> pd.Series:
    """author_id -> modal field_id of the author's works before `cutoff` (ties: smallest field id)."""
    auth = pd.read_parquet(processed / "authorships.parquet", columns=["work_id", "author_id"])
    works = pd.read_parquet(processed / "works.parquet", columns=["work_id", "year"])
    wf = pd.read_parquet(processed / "work_fields.parquet", columns=["work_id", "field_id"])
    a = auth.merge(works, on="work_id").merge(wf, on="work_id")
    a = a[(a.year < cutoff) & a.field_id.notna()].drop_duplicates(["author_id", "work_id"])
    c = a.groupby(["author_id", "field_id"]).size().rename("n").reset_index()
    c = c.sort_values(["author_id", "n", "field_id"], ascending=[True, False, True])
    return c.drop_duplicates("author_id").set_index("author_id").field_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    split, seed = args.split, cfg["seed"]
    res, fig, processed = Path(cfg["paths"]["results"]), Path(cfg["paths"]["figures"]), Path(cfg["paths"]["processed"])
    n_boot = int(cfg["evaluation"]["n_bootstrap"])
    E = Experiment(cfg)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    pfp, gp = search_best(cfg, PF), search_best(cfg, GEN)
    sp_ = E.splits[split]
    X = E.Xsets(split, ALL_FEATURES)
    d2 = (sp_["positives"]["distance"] == 2).values
    P = pd.concat([sp_["positives"][d2][["u_idx", "v_idx"]].assign(label=1),
                   sp_["neg_hard_active"][["u_idx", "v_idx"]].assign(label=0)], ignore_index=True)
    feats = pd.concat([X["positives"][d2], X["neg_hard_active"]], ignore_index=True)
    sc = {PF: [], GEN: []}
    for s in cfg["seeds"]:
        mp = E.fit("lightgbm", ALL_FEATURES, lmix, s, pfp, active=True, d2_only=True)
        mg = E.fit("lightgbm", ALL_FEATURES, lmix, s, gp)
        sc[PF].append(FM.predict(mp, feats))
        sc[GEN].append(FM.predict(mg, feats))
    P[PF], P[GEN], P[AA] = np.mean(sc[PF], axis=0), np.mean(sc[GEN], axis=0), feats["adamic_adar"].values
    P = pd.concat([P, feats[PROFILE].reset_index(drop=True)], axis=1)
    P["newcomer"] = newcomer_pairs(sp_, P.u_idx.values, P.v_idx.values)
    ids = sp_["nodes"]["author_id"].values
    P["author_u"], P["author_v"] = ids[P.u_idx], ids[P.v_idx]
    fld = author_fields(processed, cfg["splits"][split]["cutoff"])
    fu, fv = P.author_u.map(fld), P.author_v.map(fld)
    P["field_pair"] = np.where(fu.isna() | fv.isna(), "unknown", np.where(fu == fv, "same_field", "cross_field"))
    P["cross"] = (P.field_pair == "cross_field").astype(float)
    # right-censoring: joint UTS work in any year after the window
    later = all_uts_coauthor_pairs(load_tables(processed)["auth"])
    later = later[later.year > cfg["splits"][split]["window_end"]]
    lk = set(zip(later.author_u, later.author_v))
    P["later_collab"] = [(min(a, b), max(a, b)) in lk for a, b in zip(P.author_u, P.author_v)]
    last_year = int(later.year.max()) if len(later) else cfg["splits"][split]["window_end"]
    n_later_years = int(later.year.nunique())

    rows = []
    rng = np.random.default_rng([seed, 851, 0 if split == "val" else 1])
    for grp, gm in (("established", ~P.newcomer), ("newcomer", P.newcomer)):
        G_ = P[gm]
        pos, neg = G_[G_.label == 1], G_[G_.label == 0]
        fp = neg.sort_values(PF, ascending=False).head(TOP)
        fn = pos.sort_values(PF, ascending=True).head(TOP)
        sets = {"top100_false_positives": fp, "top100_false_negatives": fn, "all_positives": pos, "all_negatives": neg}
        for nm, d in sets.items():
            for f in PROFILE + ["cross"]:
                rows.append(row(PF, split, f"group={grp}|set={nm}", f"median_{f}" if f != "cross" else "share_cross_field",
                                d[f].median() if f != "cross" else d[f].mean()))
            rows.append(row(PF, split, f"group={grp}|set={nm}", "n_pairs", len(d)))
        # H1: false positives have more common neighbours than true positives (one-sided Mann-Whitney)
        rows.append(row(PF, split, f"group={grp}|hypothesis=H1", "mannwhitney_p_fp_more_common_neighbours_than_positives",
                        mannwhitneyu(fp.common_neighbors, pos.common_neighbors, alternative="greater").pvalue))
        rows.append(row(PF, split, f"group={grp}|hypothesis=H1", "mannwhitney_p_fp_higher_rw_aa_than_positives",
                        mannwhitneyu(fp.rw_aa, pos.rw_aa, alternative="greater").pvalue))
        # H2: missed positives are topically distant: lower text_cos (one-sided) and more cross-field (Fisher)
        rows.append(row(PF, split, f"group={grp}|hypothesis=H2", "mannwhitney_p_fn_lower_text_cos_than_positives",
                        mannwhitneyu(fn.text_cos, pos.drop(fn.index).text_cos, alternative="less").pvalue))
        tab = [[int(fn.cross.sum()), int(len(fn) - fn.cross.sum())],
               [int(pos.drop(fn.index).cross.sum()), int(len(pos) - len(fn) - pos.drop(fn.index).cross.sum())]]
        rows.append(row(PF, split, f"group={grp}|hypothesis=H2", "fisher_p_fn_more_cross_field_than_positives",
                        fisher_exact(tab, alternative="greater").pvalue))
        # H3: ROC-AUC by common-neighbour tercile (does the model degrade in dense neighbourhoods?)
        q = pd.qcut(G_.common_neighbors.rank(method="first"), 3, labels=["low", "mid", "high"])
        for t in ("low", "mid", "high"):
            d = G_[q == t]
            a = d[d.label == 1][PF].values
            b = d[d.label == 0][PF].values
            lo, hi = bs.classification_ci(a, b, "roc_auc", rng, n_boot)
            rows.append(row(PF, split, f"group={grp}|hypothesis=H3|cn_tercile={t}", "roc_auc", roc_auc(a, b), lo, hi))
            rows.append(row(PF, split, f"group={grp}|hypothesis=H3|cn_tercile={t}", "positive_share", d.label.mean()))
        # right-censoring (only checkable when the data extend beyond the prediction window)
        rows.append(row(PF, split, f"group={grp}|censoring", "later_years_in_data", n_later_years))
        if n_later_years:
            rows.append(row(PF, split, f"group={grp}|set=top100_false_positives", f"share_collaborate_by_{last_year}",
                            fp.later_collab.mean()))
            rows.append(row(PF, split, f"group={grp}|set=all_negatives", f"share_collaborate_by_{last_year}",
                            neg.later_collab.mean()))
            rows.append(row(PF, split, f"group={grp}|set=top100_false_positives", f"n_collaborate_by_{last_year}",
                            fp.later_collab.sum()))
    # cross-discipline performance (pooled hard_active pairs within each field class; all three models)
    for fc in ("same_field", "cross_field"):
        d = P[P.field_pair == fc]
        rows.append(row(PF, split, f"field_pair={fc}", "n_positives", (d.label == 1).sum()))
        rows.append(row(PF, split, f"field_pair={fc}", "n_negatives", (d.label == 0).sum()))
        rows.append(row(PF, split, f"field_pair={fc}", "newcomer_share", d.newcomer.mean()))
        for m in (PF, GEN, AA):
            a, b = d[d.label == 1][m].values, d[d.label == 0][m].values
            lo, hi = bs.classification_ci(a, b, "roc_auc", rng, n_boot)
            rows.append(row(m, split, f"field_pair={fc}", "roc_auc", roc_auc(a, b), lo, hi))
            for grp, gm in (("established", ~d.newcomer), ("newcomer", d.newcomer)):
                dd = d[gm]
                a, b = dd[dd.label == 1][m].values, dd[dd.label == 0][m].values
                if len(a) >= 20 and len(b) >= 20:
                    lo, hi = bs.classification_ci(a, b, "roc_auc", rng, n_boot)
                    rows.append(row(m, split, f"field_pair={fc}|group={grp}", "roc_auc", roc_auc(a, b), lo, hi))
    rows.append(row(PF, split, "field_pair=unknown", "share_of_pairs", (P.field_pair == "unknown").mean()))
    rows.append(row(PF, split, "field_pair=cross_field", "share_of_positives", P[P.label == 1].cross.mean()))
    rows.append(row(PF, split, "field_pair=cross_field", "share_of_negatives", P[P.label == 0].cross.mean()))
    # newcomers: group scores from the final comparison (no new run)
    fc = pd.read_csv(res / "final_comparison.csv")
    fc = fc[(fc.split == split) & fc.subset.isin(["established_matched", "newcomer_matched"])
            & fc.metric.isin(["roc_auc", "query_auc"])]
    for r in fc.itertuples():
        rows.append({"model": r.model, "split": split, "protocol": r.protocol, "subset": f"newcomer_check|{r.subset}",
                     "negative_type": r.negative_type, "seed": int(r.seed), "metric": r.metric, "value": float(r.value),
                     "ci_low": None if pd.isna(r.ci_low) else r.ci_low, "ci_high": None if pd.isna(r.ci_high) else r.ci_high})
    write_results(other_split(res / "error_analysis.csv", split) + rows, res / "error_analysis.csv", append=False)
    figure(res, fig, split)


def figure(res: Path, fig: Path, split: str):
    """Profiles of the top false positives / negatives relative to the group's negatives / positives, and
    ROC-AUC on same- vs cross-field pairs (read back from results/error_analysis.csv)."""
    d = pd.read_csv(res / "error_analysis.csv")
    d = d[d.split == split]
    feats = ["deg_min", "career_age_min", "recent_works_min", "text_cos", "nbr_text_sim_max", "shared_institutions",
             "common_neighbors", "cn_non_uts", "rw_aa"]
    f_, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, grp in zip(axes[:2], ("established", "newcomer")):
        def med(setn, f):
            x = d[(d.subset == f"group={grp}|set={setn}") & (d.metric == f"median_{f}")].value
            return float(x.iloc[0])
        y = np.arange(len(feats))
        for j, (setn, ref, lab) in enumerate((("top100_false_positives", "all_negatives", "top-100 FP ÷ all negatives"),
                                              ("top100_false_negatives", "all_positives", "top-100 FN ÷ all positives"))):
            ratio = [np.log2((med(setn, f) + 1e-3) / (med(ref, f) + 1e-3)) for f in feats]
            ax.barh(y + j * 0.4, ratio, height=0.38, color=SERIES[j], label=lab)
        ax.axvline(0, color=INK_2, linewidth=0.6)
        ax.set_yticks(y + 0.2)
        ax.set_yticklabels(feats, fontsize=7.5, color=INK)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{grp} pairs: median ratio (log2)", loc="left", fontsize=9, color=INK)
        ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
        ax.grid(axis="x", color=GRID)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax = axes[2]
    for j, m in enumerate((PF, GEN, AA)):
        for k, fc in enumerate(("same_field", "cross_field")):
            r = d[(d.model == m) & (d.subset == f"field_pair={fc}") & (d.metric == "roc_auc")].iloc[0]
            x = k + (j - 1) * 0.27
            ax.bar(x, r.value, width=0.25, color=SERIES[j], label=m if k == 0 else None)
            ax.errorbar(x, r.value, yerr=[[r.value - r.ci_low], [r.ci_high - r.value]], color=INK_2, capsize=2, lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["same field", "cross field"], fontsize=8, color=INK)
    ax.set_ylim(0.5, 0.95)
    ax.set_facecolor(SURFACE)
    ax.set_title("ROC-AUC on hard_active pairs by field relation", loc="left", fontsize=9, color=INK)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    ax.grid(axis="y", color=GRID)
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    f_.suptitle(f"Error analysis of the pair-focused LightGBM ({split}; seed-averaged scores; 95% bootstrap CIs)",
                x=0.01, ha="left", fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"error_analysis_{split}.png", dpi=150)
    plt.close(f_)


if __name__ == "__main__":
    main()
