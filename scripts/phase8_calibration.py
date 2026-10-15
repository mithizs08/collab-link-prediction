"""Phase 8, step 6: calibration of the headline and general LightGBM, and a calibrated two-stage model.

  python scripts/phase8_calibration.py --splits val          # validation first (rule 8)
  python scripts/phase8_calibration.py --splits val test     # final (test comparison is in the Holm family)

A. Reliability and Brier score. Seed-averaged probabilities (5 seeds) of the pair-focused LightGBM and the general
   LightGBM on the split's hard_active pairs and pooled hard pairs: Brier score, expected calibration error (10 equal-
   width bins), mean prediction vs observed positive rate, reliability curve (report/calibration/*.csv, figure).
   The evaluation sets are sampled about 1:1, the models were trained at 1:3, and real new collaborations are a tiny
   fraction of all pairs, so none of these probabilities is a real-world collaboration probability.

B. Calibrated two-stage. Both components of scripts/two_stage.py (stage 1: P(both active), selected parameters from
   results/two_stage_activity_search.csv; stage 2: pair-focused LightGBM) get a Platt scaling map (logistic regression
   on the logit) fitted on TRAINING-split out-of-fold predictions (5 folds, seeded; the fold models are fitted exactly
   like the final ones, including early stopping on validation) -- never on validation or test labels. The calibrated
   product is compared with the general LightGBM on pooled hard (ROC-AUC) and macro hard, as the uncalibrated
   two-stage was. Raw p-values; Holm in final_comparison.py (family in docs/decisions.md, "Phase 8 design").

Outputs: results/calibration.csv, results/calibrated_two_stage.csv, results/calibrated_two_stage_tests.csv,
         report/calibration/reliability_<split>.csv, figures/calibration_<split>.png.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from linkpred.features.build import features_for  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from phase7_common import averaged, n_boot_final, raws, summary_rows, test_rows  # noqa: E402
from phase8_shap import other_split  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402
from two_stage import active_both  # noqa: E402

log = logging.getLogger("calibration")
PF, GEN, CTS = "lightgbm_pair_focused", "lightgbm", "two_stage_calibrated"
PROTOS = ("hard", "hard_active", "rank_hard_active")
CELLS = (("hard", "all", "roc_auc"), ("hard", "macro", "roc_auc"), ("hard_active", "macro", "roc_auc"),
         ("rank_hard_active", "macro", "query_auc"))
TESTS = (("hard", "all", "roc_auc"), ("hard", "macro", "roc_auc"))
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt(p_oof, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_oof).reshape(-1, 1), y)
    return lambda p: m.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "mean_pred": float(p[m].mean()), "obs_rate": float(y[m].mean()), "n": int(m.sum())})
    ece = sum(r["n"] * abs(r["mean_pred"] - r["obs_rate"]) for r in out) / len(p)
    return out, ece


def eval_pairs(E, split, which):
    sp_ = E.splits[split]
    X = E.Xsets(split, ALL_FEATURES)
    d2 = (sp_["positives"]["distance"] == 2).values
    neg = "neg_hard_active" if which == "hard_active" else "neg_hard"
    return pd.concat([X["positives"][d2], X[neg]], ignore_index=True), np.r_[np.ones(d2.sum()), np.zeros(len(X[neg]))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res, rep, fig = Path(cfg["paths"]["results"]), Path(cfg["paths"]["report"]) / "calibration", Path(cfg["paths"]["figures"])
    rep.mkdir(parents=True, exist_ok=True)
    n_boot = n_boot_final(cfg)
    E = Experiment(cfg)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    pfp, gp = search_best(cfg, PF), search_best(cfg, GEN)
    a = pd.read_csv(res / "two_stage_activity_search.csv")
    best = a[a.metric == "roc_auc"].assign(v=lambda d: d.value.round(6)).sort_values("v", ascending=False, kind="stable").iloc[0]
    kv = dict(x.split("=") for x in best.model.split("|")[1:])
    lcfg = cfg["models"]["lightgbm"]
    ap1 = {"learning_rate": float(kv["lr"]), "num_leaves": int(kv["leaves"]), "min_child_samples": int(kv["min_child"]),
           "subsample": lcfg["subsample"], "colsample_bytree": lcfg["colsample_bytree"],
           "early_stopping_rounds": lcfg["early_stopping_rounds"]}

    auth = pd.read_parquet(Path(cfg["paths"]["processed"]) / "authorships.parquet", columns=["work_id", "author_id"])
    works = pd.read_parquet(Path(cfg["paths"]["processed"]) / "works.parquet", columns=["work_id", "year"])
    auth = auth.merge(works, on="work_id", validate="many_to_one")
    tr, va = E.splits["train"], E.splits["val"]
    # stage-1 data (as two_stage.py)
    pool = tr["train_pool_hard"]
    X1 = features_for(E.feat["train"], pool["u_idx"].values, pool["v_idx"].values, ALL_FEATURES)
    y1 = active_both(cfg, auth, "train", tr["nodes"], pool["u_idx"], pool["v_idx"])
    X1v = E.X["val"]["neg_hard"][ALL_FEATURES]
    y1v = active_both(cfg, auth, "val", va["nodes"], va["neg_hard"]["u_idx"], va["neg_hard"]["v_idx"])
    # stage-2 data (as Experiment.fit for the pair-focused model)
    X2, y2 = FM.training_set(tr, E.feat["train"], ALL_FEATURES, lmix[0], lmix[1], False, True, True)
    X2v, y2v = FM.hard_eval_set(va, E.Xsets("val", ALL_FEATURES), "neg_hard_active")

    probs = {m: {s: [] for s in args.splits} for m in (PF, GEN, "two_stage", CTS)}
    cal_rows = []
    for seed in cfg["seeds"]:
        maps = {}
        for stage, (X, y, Xv, yv, params) in {"stage1": (X1, y1, X1v, y1v, ap1), "stage2": (X2, y2, X2v, y2v, pfp)}.items():
            oof = np.zeros(len(y))
            for k, (a_, b_) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                m = FM.fit_lightgbm(X.iloc[a_], y[a_], Xv, yv, params, seed)
                oof[b_] = FM.predict(m, X.iloc[b_])
            maps[stage] = platt(oof, y)
            cal_rows.append({"model": f"{stage}_oof", "split": "train", "protocol": "hard", "subset": "train_oof",
                             "negative_type": "hard" if stage == "stage1" else "hard_active", "seed": int(seed),
                             "metric": "brier_before_platt", "value": float(np.mean((oof - y) ** 2)),
                             "ci_low": None, "ci_high": None})
            cal_rows.append({**cal_rows[-1], "metric": "brier_after_platt_in_sample",
                             "value": float(np.mean((maps[stage](oof) - y) ** 2))})
        m1 = FM.fit_lightgbm(X1, y1, X1v, y1v, ap1, seed)
        m2 = E.fit("lightgbm", ALL_FEATURES, lmix, seed, pfp, active=True, d2_only=True)
        mg = E.fit("lightgbm", ALL_FEATURES, lmix, seed, gp)
        for split in args.splits:
            Xs = E.Xsets(split, ALL_FEATURES)
            p1, p2, pg = FM.score_sets(m1, Xs), FM.score_sets(m2, Xs), FM.score_sets(mg, Xs)
            probs[PF][split].append(p2)
            probs[GEN][split].append(pg)
            probs["two_stage"][split].append({k: p1[k] * p2[k] for k in p1})
            probs[CTS][split].append({k: maps["stage1"](p1[k]) * maps["stage2"](p2[k]) for k in p1})
        log.info("seed %d done", seed)

    rows, tests, rel = list(cal_rows), [], []
    for split in args.splits:
        sp_ = E.splits[split]
        d2 = (sp_["positives"]["distance"] == 2).values
        # A. reliability / Brier on seed-averaged probabilities
        for m in (PF, GEN):
            avg = {k: np.mean([s[k] for s in probs[m][split]], axis=0) for k in probs[m][split][0]}
            for which, neg in (("hard_active", "neg_hard_active"), ("hard", "neg_hard")):
                p = np.r_[avg["positives"][d2], avg[neg]]
                y = np.r_[np.ones(d2.sum()), np.zeros(len(avg[neg]))]
                curve, ece = reliability(p, y)
                rel += [{"model": m, "split": split, "set": which, **c} for c in curve]
                base = {"model": m, "split": split, "protocol": which, "subset": "all",
                        "negative_type": "hard_active" if which == "hard_active" else "hard", "seed": -1,
                        "ci_low": None, "ci_high": None}
                rows += [{**base, "metric": "brier", "value": float(np.mean((p - y) ** 2))},
                         {**base, "metric": "brier_constant_base_rate", "value": float(y.mean() * (1 - y.mean()))},
                         {**base, "metric": "ece_10_bins", "value": float(ece)},
                         {**base, "metric": "mean_prediction", "value": float(p.mean())},
                         {**base, "metric": "positive_rate", "value": float(y.mean())}]
        # B. calibrated two-stage vs general LightGBM (and uncalibrated two-stage for reference)
        R = {m: raws(sp_, split, m, probs[m][split], PROTOS, cfg) for m in (CTS, "two_stage", GEN)}
        for i, m in enumerate((CTS, "two_stage", GEN)):
            rows += summary_rows(m if m == CTS else f"{m}|calibration_run", split, R[m], CELLS, cfg, n_boot, 121 + i)
        A = averaged(sp_, split, CTS, probs[CTS][split], PROTOS, cfg)
        B = averaged(sp_, split, GEN, probs[GEN][split], PROTOS, cfg)
        tests += test_rows(f"{CTS} - lightgbm", split, A, B, TESTS, cfg, n_boot, 131)
        for r in rows + tests:
            if r["split"] == split and r["seed"] == -1 and not r["metric"].endswith("_sd"):
                print(split, r["model"], r["protocol"], r["subset"], r["metric"], round(r["value"], 4), flush=True)
    calib = [r for r in rows if r["metric"] in ("brier", "brier_constant_base_rate", "ece_10_bins", "mean_prediction",
                                                 "positive_rate") or r["split"] == "train"]
    rest = [r for r in rows if r not in calib]
    write_results(calib, res / "calibration.csv", append=False)
    write_results(rest, res / "calibrated_two_stage.csv", append=False)
    write_results(tests, res / "calibrated_two_stage_tests.csv", append=False)
    pd.DataFrame(rel).to_csv(rep / "reliability.csv", index=False)
    for split in args.splits:
        figure(rep, res, fig, split)


def figure(rep: Path, res: Path, fig: Path, split: str):
    rel = pd.read_csv(rep / "reliability.csv")
    cal = pd.read_csv(res / "calibration.csv")
    f_, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, which in zip(axes, ("hard_active", "hard")):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        for j, m in enumerate((PF, GEN)):
            d = rel[(rel.model == m) & (rel.split == split) & (rel.set == which)]
            b = cal[(cal.model == m) & (cal.split == split) & (cal.protocol == which) & (cal.metric == "brier")].value.iloc[0]
            ax.plot(d.mean_pred, d.obs_rate, marker="o", markersize=3, color=SERIES[j],
                    label=f"{'pair-focused' if m == PF else 'general'} (Brier {b:.3f})")
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{which} pairs ({split})", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("mean predicted probability (bin)", fontsize=8, color=INK_2)
        ax.set_ylabel("observed positive rate", fontsize=8, color=INK_2)
        ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
        ax.grid(color=GRID)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    f_.suptitle("Reliability diagrams (seed-averaged probabilities, 10 bins). Evaluation sets are sampled ~1:1; "
                "not real-world collaboration probabilities.", x=0.01, ha="left", fontsize=9.5, color=INK)
    f_.tight_layout()
    f_.savefig(fig / f"calibration_{split}.png", dpi=150)
    plt.close(f_)


if __name__ == "__main__":
    main()
