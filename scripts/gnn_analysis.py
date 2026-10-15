"""Phase 6, local part (after scripts/import_colab_results.py): paired tests, hybrid, report.

  python scripts/gnn_analysis.py [--scores-dir data/interim/gnn] [--results-dir results] [--n-boot N]

- Paired tests on test, per seed: every final GNN vs Adamic-Adar, vs the validation-best heuristic
  per protocol, and vs the pair-focused LightGBM (refitted here; identical to Phase 4/5).
- Hybrid: pair-focused LightGBM (same features/settings) + the best GNN's score as one extra
  feature. Training-pair GNN scores are OUT-OF-FOLD (GNNs trained on the train split without
  those pairs); val/test GNN scores come from the final run with the same seed.
Writes results/gnn_paired.csv, results/gnn_hybrid.csv, results/gnn.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.feature_report import msd, seed_summary  # noqa: E402
from linkpred.eval.protocols import PROTOCOL_ORDER, PairScores, evaluate  # noqa: E402
from linkpred.features.build import features_for  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from run_feature_models import Experiment, paired, select_mix  # noqa: E402

MAIN = ("hard_active", "rank_hard_active", "hard", "rank_hard", "rank_random", "random")
PAIR = ("hard_active", "rank_hard_active", "hard", "rank_hard")
NEG = {"random": "random", "rank_random": "random", "hard_active": "hard_active", "rank_hard_active": "hard_active"}
PF = "lightgbm_pair_focused"


def load_scores(d: Path, split: str) -> dict:
    z = np.load(d / f"scores_{split}.npz")
    return {k: z[k].astype(np.float64) for k in PairScores.SOURCES}


def fit_hybrid(E, sd: Path, s: int, gnn_name: str, pf_params: dict, lmix) -> dict:
    """Retrained pair-focused LightGBM (distance-2 positives, audit M1) + out-of-fold GNN score, seed s.

    The out-of-fold scores were computed on Colab for the pre-audit training set (all positives +
    activity-matched negatives); the distance-2 training pairs are a subset of it, so their scores are
    looked up by pair. Returns {split: {pair set: scores}}."""
    cols = ALL_FEATURES + ["gnn_score"]
    z = np.load(sd / "oof" / f"seed{s}" / "train_pairs.npz")
    oof = pd.Series(z["score"], index=pd.MultiIndex.from_arrays([np.minimum(z["u"], z["v"]), np.maximum(z["u"], z["v"])]))
    pairs, y = FM.training_pairs(E.splits["train"], lmix[0], lmix[1], active=True, d2_only=True)
    key = pd.MultiIndex.from_arrays([np.minimum(pairs.u_idx, pairs.v_idx), np.maximum(pairs.u_idx, pairs.v_idx)])
    pos = oof.index.get_indexer(key)
    assert (pos >= 0).all(), "every distance-2 training pair must have an out-of-fold GNN score"
    score = oof.values[pos]
    assert np.isfinite(score).all()
    X = features_for(E.feat["train"], pairs.u_idx.values, pairs.v_idx.values, ALL_FEATURES).assign(gnn_score=score)
    fd = sd / "final" / gnn_name / f"seed{s}"
    sc = {sp_: load_scores(fd, sp_) for sp_ in ("val", "test")}
    Xs = {sp_: {k: E.Xsets(sp_, ALL_FEATURES)[k].assign(gnn_score=sc[sp_][k]) for k in PairScores.SOURCES}
          for sp_ in ("val", "test")}
    Xv, yv = FM.hard_eval_set(E.splits["val"], Xs["val"], "neg_hard_active")
    m = FM.fit_lightgbm(X[cols], y, Xv[cols], yv, pf_params, s)
    return {sp_: {k: FM.predict(m, Xs[sp_][k][cols]) for k in PairScores.SOURCES} for sp_ in ("val", "test")}


def pf_lgbm_scores(E, s, params, lmix, d2_only=True) -> dict:
    m = E.fit("lightgbm", ALL_FEATURES, lmix, s, params, active=True, d2_only=d2_only)
    return {sp_: FM.score_sets(m, E.Xsets(sp_, ALL_FEATURES)) for sp_ in ("val", "test")}


def shortcut_rows(E, name, seed, scores) -> list[dict]:
    """Distance-shortcut check on the random protocol (test): does the model rank FAR random negatives
    above NEAR positives? AUC(far negatives vs distance-2 positives) > 0.5 means it does."""
    X = E.X["test"]
    dpos, dneg = X["positives"]["distance"].values, X["neg_random"]["distance"].values
    sp, sn = scores["positives"], scores["neg_random"]
    base = {"model": name, "split": "test", "protocol": "random", "negative_type": "random", "seed": seed}
    out = [{**base, "subset": "far_negatives_vs_distance2_positives", "metric": "roc_auc",
            "value": roc_auc(sn[dneg >= 3], sp[dpos == 2])},
           {**base, "subset": "far_positives_vs_far_negatives", "metric": "roc_auc",
            "value": roc_auc(sp[dpos >= 3], sn[dneg >= 3])}]
    for lab, x in (("distance2_positives", sp[dpos == 2]), ("far_positives", sp[dpos >= 3]),
                   ("far_negatives", sn[dneg >= 3])):
        out.append({**base, "subset": lab, "metric": "mean_score_percentile",
                    "value": float(np.mean(np.searchsorted(np.sort(np.r_[sp, sn]), x) / (len(sp) + len(sn))))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scores-dir", type=Path, default=Path("data/interim/gnn"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true", help="rebuild results/gnn.md from existing CSVs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.n_boot is not None:
        cfg["evaluation"]["n_bootstrap"] = args.n_boot
    res, sd = args.results_dir, args.scores_dir
    if args.report_only:
        oof = sorted((sd / "oof").glob("seed*/meta.json"))
        hyb = f"{PF}+{json.loads(oof[0].read_text())['model']}" if oof else None
        models = sorted(pd.read_csv(res / "gnn.csv").model.unique())
        return write_report(cfg, res, models, hyb)
    set_seed(cfg["seed"])
    E = Experiment(cfg)
    k_full = cfg["negatives"]["ranking_candidates"]
    n_boot = cfg["evaluation"]["n_bootstrap"]
    # The Colab-evaluated rows live in gnn_colab.csv, which import_colab_results.py rewrites on every import
    # (audit N7: an earlier version renamed gnn.csv only on the first run, so a re-import was never refreshed).
    colab = res / "gnn_colab.csv"
    if not colab.exists():
        raise SystemExit(f"{colab} missing: run scripts/import_colab_results.py first")
    colab_df = pd.read_csv(colab)
    seeds = args.seeds if args.seeds is not None else sorted(int(x) for x in colab_df.seed.unique())
    models = sorted(colab_df.model.unique())
    base = pd.read_csv(Path("results") / "baselines.csv")
    val_best = report.val_best(base, PROTOCOL_ORDER)

    # 1. Re-evaluate every saved GNN score with the current protocols (adds the group-matched rows).
    rows, graw = [], {}
    for name in models:
        sub = "ablation" if name.endswith("_dot") else "final"
        for s in seeds:
            for split in ("val", "test"):
                sc = load_scores(sd / sub / name / f"seed{s}", split)
                r, x = evaluate(None, E.splits[split], split, name, s, cfg["seed"], n_boot, k_full=k_full, precomputed=sc)
                rows += r
                graw[(name, s, split)] = (x, sc)
    write_results(rows, res / "gnn.csv", append=False)
    new = pd.DataFrame(rows)
    key = ["model", "split", "protocol", "subset", "metric", "seed"]
    chk = colab_df.merge(new, on=key, suffixes=("_colab", "_local"))
    (res / "gnn_reeval_check.json").write_text(json.dumps({
        "rows_compared": int(len(chk)),
        "max_abs_value_diff": float((chk.value_colab - chk.value_local).abs().max()),
        "note": "Colab evaluated float64 scores; local re-evaluation uses the saved float32 scores."}, indent=2) + "\n")

    # 2. References: heuristics (from features) and the retrained pair-focused LightGBM.
    pf_params = search_best(cfg, PF)
    sc_params = search_best(cfg, PF + "_shortcut")
    lmix = select_mix(pd.read_csv(Path("results") / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    href = E.heuristic_raw("test")

    def raw_of(scores, split, name, seed):
        return evaluate(None, E.splits[split], split, name, seed, cfg["seed"], 0, k_full=k_full, precomputed=scores)[1]

    pf_raw, pf_sc, short_sc = {}, {}, {}
    for s in seeds:
        pf_sc[s] = pf_lgbm_scores(E, s, pf_params, lmix, d2_only=True)
        short_sc[s] = pf_lgbm_scores(E, s, sc_params, lmix, d2_only=False)
        pf_raw[s] = raw_of(pf_sc[s]["test"], "test", PF, s)

    # 3. Per-seed paired tests (secondary robustness column; primary tests in final_comparison.py).
    prow = []
    for name in models:
        for s in seeds:
            r = graw[(name, s, "test")][0]
            for p in MAIN:
                nt = NEG.get(p, "hard")
                refs = {val_best[p]} | ({"adamic_adar"} if p in PAIR else set())
                for h in sorted(refs):
                    prow += paired(cfg, r, href[h][1], p, name, h, s, nt)
                prow += paired(cfg, r, pf_raw[s], p, name, PF, s, nt)

    # 4. Hybrid on the retrained pair-focused LightGBM.
    hyb_rows, hyb_name = [], None
    oof_dirs = sorted((sd / "oof").glob("seed*/meta.json"))
    if oof_dirs:
        gnn_name = json.loads(oof_dirs[0].read_text())["model"]
        hyb_name = f"{PF}+{gnn_name}"
        for s in seeds:
            if not (sd / "oof" / f"seed{s}" / "meta.json").exists():
                continue
            hs = fit_hybrid(E, sd, s, gnn_name, pf_params, lmix)
            for sp_ in ("val", "test"):
                rows_, x = evaluate(None, E.splits[sp_], sp_, hyb_name, s, cfg["seed"], n_boot, k_full=k_full, precomputed=hs[sp_])
                hyb_rows += rows_
                if sp_ == "test":
                    for p in PAIR:
                        prow += paired(cfg, x, pf_raw[s], p, hyb_name, PF, s, NEG.get(p, "hard"))
        write_results(hyb_rows, res / "gnn_hybrid.csv", append=False)
    write_results(prow, res / "gnn_paired.csv", append=False)

    # 5. Distance-shortcut check (audit M1) for the pair-focused GNNs, with the LightGBM pair as reference.
    srows = []
    for name in [m for m in models if m.startswith("gnn_pair_focused")] + ["gnn_general_sage_text"]:
        for s in seeds:
            srows += shortcut_rows(E, name, s, graw[(name, s, "test")][1])
    for s in seeds:
        srows += shortcut_rows(E, PF + "_shortcut", s, short_sc[s]["test"])
        srows += shortcut_rows(E, PF, s, pf_sc[s]["test"])
    write_results(srows, res / "gnn_shortcut_check.csv", append=False)
    write_report(cfg, res, models, hyb_name)


def write_report(cfg, res: Path, models, hyb_name):
    gnn = pd.read_csv(res / "gnn.csv")
    fm = pd.read_csv(Path("results") / "feature_models.csv")
    parts = [gnn, fm[fm.model == PF]]
    if (res / "gnn_hybrid.csv").exists() and hyb_name:
        parts.append(pd.read_csv(res / "gnn_hybrid.csv"))
    s = seed_summary(pd.concat(parts))
    base = pd.read_csv(Path("results") / "baselines.csv")
    pr = pd.read_csv(res / "gnn_paired.csv")
    learned = models + [PF] + ([hyb_name] if hyb_name else [])
    order = learned + ["adamic_adar", "jaccard", "prefer_newer"]
    ver = json.loads((res / "gnn_versions.json").read_text()) if (res / "gnn_versions.json").exists() else {}
    ks_r = [f"hits@{k}" for k in cfg["evaluation"]["rank_ks_random"]]

    def get(m, split, p, subset, metric):
        if m in learned:
            d = s[(s.model == m) & (s.split == split) & (s.protocol == p) & (s.subset == subset) & (s.metric == metric)]
            return msd(d.iloc[0]) if len(d) else ""
        d = base[(base.model == m) & (base.split == split) & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)]
        return report.fmt(*d[["value", "ci_low", "ci_high"]].iloc[0]) if len(d) else ""

    out = ["# GNN link predictors (GraphSAGE, GAT)", "",
           "_Generated by `scripts/gnn_analysis.py` from `results/gnn.csv` (saved Colab scores re-evaluated locally with "
           "the current protocols; the Colab-evaluated rows are kept in `gnn_colab.csv`), `gnn_paired.csv`, "
           "`gnn_hybrid.csv`, `gnn_shortcut_check.csv`, `gnn_grid.csv`, `gnn_determinism.csv`. Learned models: mean ± sd "
           "over seeds; reference rows: point [bootstrap 95% CI]. Headline = macro within-group score (audit M2); "
           "`lightgbm_pair_focused` is the retrained distance-2 model (audit M1); the hybrid is built on it. Seed-mean "
           "CIs and Holm-corrected primary tests: `results/final_comparison.md`._", "",
           f"Versions used for GNN training (Colab): `{json.dumps(ver)}`", ""]
    if hyb_name and (res / "gnn_shortcut_check.csv").exists():
        _, note = report.shortcut_labels(res, hyb_name.split("+", 1)[1], hyb_name)
        out += [note + " All `gnn_pair_focused_*` models share this training set (their shortcut checks are in the "
                "distance-shortcut section below); the `gnn_general_*` models use random negatives and are unaffected.", ""]
    chk = res / "gnn_reeval_check.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        out += [f"Re-evaluation check: {c['rows_compared']:,} pooled rows compared with the Colab evaluation, max |diff| "
                f"{c['max_abs_value_diff']:.2e} ({c['note']})", ""]
    for split in ("test", "val"):
        out += [f"## {split}", ""]
        out += report.pair_level_md(get, order, split, cfg["evaluation"]["rank_ks_hard"])
        for p, sub, metrics in (("rank_random", "all", ks_r + ["mrr"]), ("random", "all", ["roc_auc", "average_precision"])):
            t = pd.DataFrame({"model": order})
            for m_ in metrics:
                t[m_] = [get(m, split, p, sub, m_) for m in order]
            out += [f"### {p} — {sub}", "", report.to_md(t), ""]
    sc = res / "gnn_shortcut_check.csv"
    if sc.exists():
        d = pd.read_csv(sc)
        g = d.groupby(["model", "subset", "metric"]).value.agg(["mean", "std"]).reset_index()
        g["cell"] = [f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}" for a, b in zip(g["mean"], g["std"])]
        t = g.pivot_table(index="model", columns="subset", values="cell", aggfunc="first").reset_index()
        out += ["## Distance-shortcut check (test, random protocol; audit M1)", "",
                "`far_negatives_vs_distance2_positives`: ROC-AUC of far (distance ≥ 3) random negatives against distance-2 "
                "positives — **above 0.5 means the model ranks far non-collaborators above near collaborators**, i.e. it "
                "learned \"distance ≥ 3 ⇒ positive\". `far_positives_vs_far_negatives`: can it tell far positives from far "
                "negatives at all? `mean_score_percentile`: where each group sits in the pooled score distribution "
                "(0 = lowest, 1 = highest). Mean ± sd over 5 seeds.", "", report.to_md(t), ""]
    pv = pr[pr.metric.str.endswith("_p_value")].rename(columns={"value": "p"})
    dv = pr[~pr.metric.str.endswith("_p_value")]
    m = dv.merge(pv[["model", "protocol", "subset", "seed", "p"]], on=["model", "protocol", "subset", "seed"])
    rows = [{"comparison": k[0], "protocol": k[1], "metric": f"{d['subset'].iloc[0]} {d['metric'].iloc[0]}",
             "mean diff": f"{d['value'].mean():+.4f}",
             "mean 95% CI": f"[{d['ci_low'].mean():+.4f}, {d['ci_high'].mean():+.4f}]",
             "sig. better": f"{int(((d.p < 0.05) & (d.value > 0)).sum())}/{len(d)}",
             "sig. worse": f"{int(((d.p < 0.05) & (d.value < 0)).sum())}/{len(d)}", "median p": f"{d['p'].median():.3f}"}
            for k, d in m.groupby(["model", "protocol"], sort=False)]
    out += ["## Paired bootstrap tests per seed (test) — secondary robustness column", "",
            "On each protocol's headline metric (macro within-group for pair-level protocols). The five tests share the "
            "test pairs; primary tests are in `results/final_comparison.md`.", "", report.to_md(pd.DataFrame(rows)), ""]
    det = res / "gnn_determinism.csv"
    if det.exists():
        dd = pd.read_csv(det)
        out += ["## Run-to-run variation with the same seed (GPU determinism check)", "",
                report.to_md(dd[["model", "split", "protocol", "metric", "value"]]), ""]
    out += ["## Appendix", ""] + report.positives_only_appendix(get, order, "test")
    out += ["### diagnostic_all_vs_hard", "", report.DIAGNOSTIC_WARNING, ""]
    (res / "gnn.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
