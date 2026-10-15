"""Phase 8, steps 1-3: SHAP for the pair-focused (headline) and general LightGBM; meaning-based feature groups;
the activity detector.

  python scripts/phase8_shap.py --split val     # validation first (rule 8)
  python scripts/phase8_shap.py --split test    # final figures / numbers

Models: seed 0 of each model (same parameters, mix and training data as the reported models). SHAP values are exact
TreeSHAP on the booster truncated at `best_iteration` (the trees that produce the reported scores), in log-odds;
additivity (sum of SHAP + expected value = raw score) is asserted. Pair sets of the chosen split:
  hard_active  distance-2 positives vs activity-matched distance-2 negatives (the headline protocol's pairs)
  hard         distance-2 positives vs distance-2 negatives NOT matched on activity (pooled `hard`)
Groups: within the established / newcomer groups (newcomer = an author with <= 2 works before the cutoff).

Outputs
  results/shap_importance.csv   mean |SHAP| per feature (subset `feature=<f>`), shares by original block
                                (`block=<b>`) and by meaning (`meaning=<g>`), top interactions (`interaction=<a>x<b>`),
                                Spearman rank agreement of SHAP with gain importance (`agreement=gain`)
  results/activity_detector.csv single activity features used as scores under hard / hard_active (protocols.evaluate)
  report/shap/*.csv             sampled per-pair SHAP values, feature values, partial dependence (figure inputs)
  figures/shap_*_<split>.png, figures/activity_detector_<split>.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.eval.protocols import evaluate, newcomer_pairs  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES, FAMILIES  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

log = logging.getLogger("phase8_shap")
PF, GEN = "lightgbm_pair_focused", "lightgbm"
BLOCK = {f: b for b, fs in FAMILIES.items() for f in fs}
# Meaning-based groups (Phase 8, step 2): what each feature measures, not which code block computes it.
MEANING = {
    "pure_structure": ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment",
                       "katz", "deg_min", "deg_max", "distance", "disconnected", "same_community", "cn_uts",
                       "cn_non_uts", "shared_institutions"],
    "time_weighted_structure": ["rw_cn", "rw_aa", "years_since_shared"],
    "author_activity_career": ["career_age_min", "career_age_max", "recent_works_min", "recent_works_max",
                               "years_since_last_min", "years_since_last_max"],
    "text_similarity": ["text_cos", "nbr_text_sim_min", "nbr_text_sim_max", "text_both"],
}
MEAN_OF = {f: g for g, fs in MEANING.items() for f in fs}
assert sorted(MEAN_OF) == sorted(ALL_FEATURES)
# Single activity features scored directly (sign: higher = predicted collaboration).
ACTIVITY_SCORERS = {"recent_works_min": 1.0, "years_since_last_max": -1.0}
SETS = ("hard_active", "hard")


def truncated(model) -> lgb.Booster:
    """The booster restricted to the trees used for prediction (early stopping keeps extra trees)."""
    b = lgb.Booster(model_str=model.booster_.model_to_string(num_iteration=model.best_iteration_))
    b.params = {**(b.params or {}), "objective": "binary"}      # shap reads the objective from params
    return b


def pair_set(E, split, which):
    sp_ = E.splits[split]
    X = E.Xsets(split, ALL_FEATURES)
    d2 = (sp_["positives"]["distance"] == 2).values
    neg = "neg_hard_active" if which == "hard_active" else "neg_hard"
    Xs = pd.concat([X["positives"][d2], X[neg]], ignore_index=True)
    y = np.r_[np.ones(d2.sum()), np.zeros(len(X[neg]))]
    u = np.r_[sp_["positives"]["u_idx"].values[d2], sp_[neg]["u_idx"].values]
    v = np.r_[sp_["positives"]["v_idx"].values[d2], sp_[neg]["v_idx"].values]
    newc = newcomer_pairs(sp_, u, v)
    return Xs, y, newc


def row(model, split, protocol, subset, metric, value, seed=0):
    return {"model": model, "split": split, "protocol": protocol, "subset": subset,
            "negative_type": "hard_active" if protocol.endswith("active") else "hard", "seed": int(seed),
            "metric": metric, "value": float(value), "ci_low": None, "ci_high": None}


def other_split(path: Path, split: str) -> list[dict]:
    """Rows of an existing results file for the OTHER split (so val and test runs can be made separately)."""
    if not path.exists():
        return []
    d = pd.read_csv(path)
    out = d[d.split != split].to_dict("records")
    for r in out:
        r["seed"] = int(r["seed"])
        for k in ("ci_low", "ci_high"):
            r[k] = None if pd.isna(r[k]) else r[k]
    return out


def style(ax, title, xlabel="", grid_axis="x"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=9, pad=6)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def beeswarm(ax, S, X, feats, title, rng):
    """Compact beeswarm: one row per feature, SHAP on x, colour = feature-value percentile within the set."""
    cmap = plt.get_cmap("coolwarm")
    for i, f in enumerate(feats[::-1]):
        s, x = S[f].values, X[f].values
        pct = pd.Series(x).rank(pct=True).values
        jit = rng.uniform(-0.32, 0.32, len(s))
        ax.scatter(s, np.full(len(s), i) + jit, c=cmap(pct), s=3, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1], fontsize=7.5, color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.6)
    style(ax, title, "SHAP value (log-odds)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from report/shap and results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.figures_only:
        return figures(Path(cfg["paths"]["results"]), Path(cfg["paths"]["report"]) / "shap",
                       Path(cfg["paths"]["figures"]), args.split)
    set_seed(cfg["seed"])
    split, seed = args.split, cfg["seed"]
    res, rep, fig = Path(cfg["paths"]["results"]), Path(cfg["paths"]["report"]) / "shap", Path(cfg["paths"]["figures"])
    rep.mkdir(parents=True, exist_ok=True)
    E = Experiment(cfg)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    models = {PF: E.fit("lightgbm", ALL_FEATURES, lmix, seed, search_best(cfg, PF), active=True, d2_only=True),
              GEN: E.fit("lightgbm", ALL_FEATURES, lmix, seed, search_best(cfg, GEN))}
    gain = pd.read_csv(Path(cfg["paths"]["report"]) / "features" / "importance.csv")

    rows, shap_vals, feat_vals = [], {}, {}
    rng = np.random.default_rng([seed, 801, 0 if split == "val" else 1])
    for name, m in models.items():
        bst = truncated(m)
        ex = shap.TreeExplainer(bst)
        for which in SETS:
            X, y, newc = pair_set(E, split, which)
            sv = ex.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            raw = bst.predict(X, raw_score=True)
            ev = ex.expected_value[1] if np.ndim(ex.expected_value) else ex.expected_value
            assert np.allclose(sv.sum(axis=1) + ev, raw, atol=1e-4), "SHAP additivity failed"
            S = pd.DataFrame(sv, columns=ALL_FEATURES)
            shap_vals[(name, which)], feat_vals[(name, which)] = S, X
            for grp, mask in (("all", np.ones(len(y), bool)), ("established", ~newc), ("newcomer", newc)):
                mabs = S[mask].abs().mean()
                tag = "" if grp == "all" else f"|group={grp}"
                rows += [row(name, split, which, f"feature={f}{tag}", "mean_abs_shap", mabs[f]) for f in ALL_FEATURES]
                tot = mabs.sum()
                rows += [row(name, split, which, f"block={b}{tag}", "shap_share", mabs[fs].sum() / tot)
                         for b, fs in FAMILIES.items()]
                rows += [row(name, split, which, f"meaning={g}{tag}", "shap_share", mabs[fs].sum() / tot)
                         for g, fs in MEANING.items()]
                # Discrimination (not magnitude): ROC-AUC of SHAP contributions used as a score within this pair set.
                # A feature that is constant in the set (e.g. distance on distance-2 pairs) can have a large mean |SHAP|
                # (measured against the training-data average) yet no discriminative value: its SHAP then varies only
                # through interactions (exactly 0.5 for an additive model).
                yy = y[mask]
                auc = lambda sc: roc_auc(sc[yy == 1], sc[yy == 0])
                rows += [row(name, split, which, f"feature={f}{tag}", "shap_auc", auc(S[f].values[mask])) for f in ALL_FEATURES]
                rows += [row(name, split, which, f"block={b}{tag}", "shap_auc", auc(S[fs].sum(axis=1).values[mask]))
                         for b, fs in FAMILIES.items()]
                rows += [row(name, split, which, f"meaning={g}{tag}", "shap_auc", auc(S[fs].sum(axis=1).values[mask]))
                         for g, fs in MEANING.items()]
                rows.append(row(name, split, which, f"total{tag}", "shap_auc", auc(S.sum(axis=1).values[mask])))
            # SHAP vs gain importance (gain: mean over the 5 reported seeds of the same model).
            g = gain[gain.model == name].groupby("feature").gain.mean().reindex(ALL_FEATURES)
            rho = spearmanr(S.abs().mean().values, g.values).statistic
            rows.append(row(name, split, which, "agreement=gain", "spearman_mean_abs_shap_vs_gain", rho))
            gs = g / g.sum()
            rows += [row(name, split, which, f"gain_block={b}", "gain_share", gs[fs].sum()) for b, fs in FAMILIES.items()]
            rows += [row(name, split, which, f"gain_meaning={k}", "gain_share", gs[fs].sum()) for k, fs in MEANING.items()]
            # sampled per-pair values for figures (all pairs if <= 3000)
            idx = np.sort(rng.choice(len(S), min(3000, len(S)), replace=False))
            out = pd.concat([S.iloc[idx].add_prefix("shap_").reset_index(drop=True), X.iloc[idx].reset_index(drop=True)],
                            axis=1)
            out.assign(label=y[idx], newcomer=newc[idx]).to_csv(rep / f"{name}_{which}_{split}.csv", index=False)
        # interactions on the headline pairs (sample of 1,000 for cost)
        X, y, _ = pair_set(E, split, "hard_active")
        idx = np.sort(rng.choice(len(X), min(1000, len(X)), replace=False))
        Iv = ex.shap_interaction_values(X.iloc[idx])
        Iv = Iv[1] if isinstance(Iv, list) else Iv
        M = np.abs(Iv).mean(axis=0)
        np.fill_diagonal(M, 0)
        iu = np.triu_indices(len(ALL_FEATURES), 1)
        rows.append(row(name, split, "hard_active", "interaction=max_any_pair", "mean_abs_interaction", M[iu].max()))
        if M[iu].max() < 1e-9:
            # 2-leaf trees: every tree splits on one feature, so the model is additive and has no interactions.
            log.info("%s is additive (max interaction %.2e): no interaction plots", name, M[iu].max())
            continue
        order = np.argsort(-M[iu])[:3]
        for k in order:
            a, b = ALL_FEATURES[iu[0][k]], ALL_FEATURES[iu[1][k]]
            rows.append(row(name, split, "hard_active", f"interaction={a}x{b}", "mean_abs_interaction", M[iu][k]))
            pd.DataFrame({a: X.iloc[idx][a].values, b: X.iloc[idx][b].values,
                          "interaction": Iv[:, iu[0][k], iu[1][k]] * 2}).to_csv(
                rep / f"{name}_interaction_{a}x{b}_{split}.csv", index=False)
        log.info("%s SHAP done", name)

    # ---- activity detector: single activity features as scores, and partial dependence of the general model
    sp_ = E.splits[split]
    act_rows = []
    for f, sign in ACTIVITY_SCORERS.items():
        pre = {k: sign * v[f].values for k, v in E.X[split].items()}
        r, _ = evaluate(None, sp_, split, f"single_feature[{f}]", seed, seed, 0, protocols=("hard", "hard_active"),
                        k_full=E.k_full, precomputed=pre)
        act_rows += [x for x in r if x["subset"] in ("all", "macro", "established_matched", "newcomer_matched")
                     and x["metric"] == "roc_auc"]
    pd_rows = []
    for f in ("recent_works_min", "years_since_last_max"):
        for which in SETS:
            X, y, _ = pair_set(E, split, which)
            grid = np.unique(np.quantile(X[f], np.linspace(0.02, 0.98, 15)))
            for gv in grid:
                Xg = X.copy()
                Xg[f] = gv
                pd_rows.append({"model": GEN, "set": which, "feature": f, "value": gv,
                                "mean_prob": float(models[GEN].predict_proba(Xg, num_iteration=models[GEN].best_iteration_)[:, 1].mean())})
            q = pd.qcut(X[f], 8, duplicates="drop")
            emp = pd.DataFrame({"bin_mid": q.apply(lambda iv: iv.mid).astype(float), "y": y}).groupby("bin_mid").y.mean()
            pd_rows += [{"model": "empirical_positive_rate", "set": which, "feature": f, "value": k, "mean_prob": vv}
                        for k, vv in emp.items()]
    pd.DataFrame(pd_rows).to_csv(rep / f"partial_dependence_{split}.csv", index=False)
    write_results(other_split(res / "activity_detector.csv", split) + act_rows, res / "activity_detector.csv", append=False)

    write_results(other_split(res / "shap_importance.csv", split) + rows, res / "shap_importance.csv", append=False)
    figures(res, rep, fig, split)


# ------------------------------------------------------------------------------------------- figures
def figures(res: Path, rep: Path, fig: Path, split: str):
    """All figures read report/shap/*.csv and results/shap_importance.csv (never in-memory model output)."""
    si = pd.read_csv(res / "shap_importance.csv")
    si = si[si.split == split]
    rng = np.random.default_rng(0)

    def top(model, which, k):
        d = si[(si.model == model) & (si.protocol == which) & (si.metric == "mean_abs_shap")
               & ~si.subset.str.contains("group=")]
        d = d.assign(f=d.subset.str.replace("feature=", "")).sort_values("value", ascending=False)
        return d.f.tolist()[:k]

    # 1. beeswarm side by side (hard_active pairs)
    f_, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE)
    for ax, m, lab in zip(axes, (PF, GEN), ("pair-focused LightGBM (headline)", "general LightGBM")):
        d = pd.read_csv(rep / f"{m}_hard_active_{split}.csv")
        feats = top(m, "hard_active", 12)
        beeswarm(ax, d[[f"shap_{f}" for f in feats]].set_axis(feats, axis=1), d[feats], feats, f"{lab}: top 12 features", rng)
    f_.suptitle(f"SHAP on {split} hard_active pairs (colour: feature value percentile, blue low → red high)",
                x=0.01, ha="left", fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"shap_beeswarm_{split}.png", dpi=150)
    plt.close(f_)

    # 2. dependence plots, top 6 per model
    for m, lab in ((PF, "pair-focused LightGBM"), (GEN, "general LightGBM")):
        d = pd.read_csv(rep / f"{m}_hard_active_{split}.csv")
        feats = top(m, "hard_active", 6)
        f_, axes = plt.subplots(2, 3, figsize=(11, 6), facecolor=SURFACE)
        for ax, f in zip(axes.ravel(), feats):
            c = np.where(d["newcomer"].astype(bool), SERIES[1], SERIES[0])
            ax.scatter(d[f], d[f"shap_{f}"], s=4, c=c, alpha=0.5, linewidths=0, rasterized=True)
            ax.axhline(0, color=INK_2, linewidth=0.6)
            style(ax, f, "feature value", grid_axis="both")
            ax.set_ylabel("SHAP (log-odds)", color=INK_2, fontsize=7.5)
        f_.suptitle(f"{lab}: SHAP dependence, top 6 features ({split} hard_active; orange = newcomer pair)",
                    x=0.01, ha="left", fontsize=10, color=INK)
        f_.tight_layout()
        f_.savefig(fig / f"shap_dependence_{m}_{split}.png", dpi=150)
        plt.close(f_)

    # 3. interaction values, top 3 pairs per model
    f_, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=SURFACE)
    pf_max = si[(si.model == PF) & (si.subset == "interaction=max_any_pair")].value.iloc[0]
    for r_, (m, lab) in enumerate(((GEN, "general"),)):
        ints = si[(si.model == m) & (si.metric == "mean_abs_interaction") & (si.subset != "interaction=max_any_pair")]
        ints = ints.sort_values("value", ascending=False)
        for ax, s in zip(axes, ints.subset):
            a, b = s.replace("interaction=", "").split("x", 1)
            d = pd.read_csv(rep / f"{m}_interaction_{a}x{b}_{split}.csv")
            sc = ax.scatter(d[a], d["interaction"], c=d[b].rank(pct=True), cmap="coolwarm", s=5, alpha=0.6,
                            linewidths=0, rasterized=True)
            ax.axhline(0, color=INK_2, linewidth=0.6)
            style(ax, f"{lab}: {a} × {b}", a, grid_axis="both")
            ax.set_ylabel("interaction SHAP", color=INK_2, fontsize=7.5)
        fig_note = "colour: second feature percentile"
    f_.suptitle(f"General LightGBM: SHAP interaction values, top 3 pairs ({split} hard_active, 1,000 pairs; {fig_note}). "
                f"The pair-focused model has 2-leaf trees and is additive: max |interaction| = {pf_max:.1e}.",
                x=0.01, ha="left", fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"shap_interactions_{split}.png", dpi=150)
    plt.close(f_)

    # 4. shares: original blocks vs meaning groups, both models, both pair sets
    f_, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for ax, kind, title in ((axes[0], "block=", "original blocks"), (axes[1], "meaning=", "meaning-based groups")):
        d = si[(si.metric == "shap_share") & si.subset.str.startswith(kind) & ~si.subset.str.contains("group=")]
        labels = list(dict.fromkeys(d.subset.str.replace(kind, "")))
        combos = [(PF, "hard_active"), (GEN, "hard_active"), (GEN, "hard")]
        w = 0.8 / len(combos)
        for j, (m, which) in enumerate(combos):
            vals = [d[(d.model == m) & (d.protocol == which) & (d.subset == kind + g)].value.iloc[0] for g in labels]
            ax.barh(np.arange(len(labels)) + j * w, vals, height=w * 0.9, color=SERIES[j],
                    label=f"{'pair-focused' if m == PF else 'general'} on {which}")
        ax.set_yticks(np.arange(len(labels)) + w)
        ax.set_yticklabels([g.replace("_", " ") for g in labels], fontsize=7.5, color=INK)
        style(ax, f"Share of mean |SHAP|: {title}", "share")
        ax.legend(fontsize=7, frameon=False)
    f_.suptitle(f"Where the attribution goes ({split})", x=0.01, ha="left", fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"shap_groups_{split}.png", dpi=150)
    plt.close(f_)

    # 5. activity detector: partial dependence + empirical rate, pooled hard vs hard_active
    pdp = pd.read_csv(rep / f"partial_dependence_{split}.csv")
    f_, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for ax, f in zip(axes, ("recent_works_min", "years_since_last_max")):
        for j, which in enumerate(SETS):
            d = pdp[(pdp.feature == f) & (pdp.set == which)]
            g = d[d.model == GEN]
            e = d[d.model == "empirical_positive_rate"]
            ax.plot(g.value, g.mean_prob, color=SERIES[j], linewidth=2, label=f"model PD on {which}")
            ax.plot(e.value, e.mean_prob, color=SERIES[j], linewidth=1, linestyle="--", marker="o", markersize=3,
                    label=f"positive rate on {which}")
        style(ax, f"general LightGBM: {f}", f, grid_axis="both")
        ax.set_ylabel("probability / rate", color=INK_2, fontsize=7.5)
        ax.legend(fontsize=7, frameon=False)
    f_.suptitle(f"Activity detector ({split}): pooled hard pairs vs activity-matched hard_active pairs",
                x=0.01, ha="left", fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"activity_detector_{split}.png", dpi=150)
    plt.close(f_)


if __name__ == "__main__":
    main()
