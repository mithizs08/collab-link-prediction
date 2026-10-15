"""Phase 7, step 3: two-stage model  P(collaborate) = P(both active) x P(collaborate | both active).

  python scripts/two_stage.py --splits val          # validation first (rule 8)
  python scripts/two_stage.py --splits val test     # final

Stage 1, P(both authors active in the prediction year): LightGBM on the 27 pre-cutoff pair features (model
inputs never contain prediction-year information, rule 7). Training pairs = the train split's distance-2
non-collaborating pool (`train_pool_hard`, a uniform sample of the pairs `hard` draws its negatives from); label =
both authors have a work in the train window (2023). Grid = models.lightgbm.search (24 configs), seed 0, early
stopping and selection on validation: ROC-AUC for "both active in 2024" on the validation `neg_hard` pairs.
Stage 2, P(collaborate | both active): the pair-focused LightGBM (activity-matched training negatives,
distance-2 positives; main selected parameters). Two-stage score = product of the two probabilities, seed by seed
(stage-1 seed s with pair-focused seed s), 5 seeds.

Evaluated on the pooled `hard` protocol (which INCLUDES activity: its negatives are not activity-matched)
against the general LightGBM, plus the other headline protocols for the master table.

Outputs: results/two_stage_activity_search.csv (stage-1 grid; protocol `hard`, subset `activity_label`: the
         label is "both active", not "collaborate"), results/two_stage.csv, results/two_stage_tests.csv (raw p;
         Holm in final_comparison.py).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.features.build import features_for  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from phase7_common import averaged, n_boot_final, raws, summary_rows, test_rows  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

log = logging.getLogger("two_stage")
PF, TS, ACT = "lightgbm_pair_focused", "two_stage", "activity_lightgbm"
PROTOS = ("hard", "hard_active", "rank_hard", "rank_hard_active", "random")
CELLS = (("hard", "all", "roc_auc"), ("hard", "macro", "roc_auc"), ("hard_active", "macro", "roc_auc"),
         ("rank_hard_active", "macro", "query_auc"), ("rank_hard", "macro", "query_auc"), ("random", "all", "roc_auc"))
TESTS = (("hard", "all", "roc_auc"), ("hard", "macro", "roc_auc"))


def active_both(cfg, auth: pd.DataFrame, split_name: str, nodes: pd.DataFrame, u, v) -> np.ndarray:
    """Label only: both authors have a work in the split's prediction window (never a model input)."""
    s = cfg["splits"][split_name]
    act = set(auth.loc[auth["year"].between(s["cutoff"], s["window_end"]), "author_id"])
    a = nodes["author_id"].isin(act).values
    return (a[np.asarray(u)] & a[np.asarray(v)]).astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    n_boot = n_boot_final(cfg)
    E = Experiment(cfg)
    auth = pd.read_parquet(Path(cfg["paths"]["processed"]) / "authorships.parquet", columns=["work_id", "author_id"])
    works = pd.read_parquet(Path(cfg["paths"]["processed"]) / "works.parquet", columns=["work_id", "year"])
    auth = auth.merge(works, on="work_id", validate="many_to_one")

    # Stage 1 data: train pool (2023 labels) and validation neg_hard (2024 labels, selection / early stopping only).
    tr, va = E.splits["train"], E.splits["val"]
    pool = tr["train_pool_hard"]
    Xtr = features_for(E.feat["train"], pool["u_idx"].values, pool["v_idx"].values, ALL_FEATURES)
    ytr = active_both(cfg, auth, "train", tr["nodes"], pool["u_idx"], pool["v_idx"])
    Xva = E.X["val"]["neg_hard"][ALL_FEATURES]
    yva = active_both(cfg, auth, "val", va["nodes"], va["neg_hard"]["u_idx"], va["neg_hard"]["v_idx"])
    log.info("stage 1: train %d pairs (%.1f%% both active), val %d pairs (%.1f%%)", len(ytr), 100 * ytr.mean(),
             len(yva), 100 * yva.mean())

    lcfg = cfg["models"]["lightgbm"]
    default = {k: lcfg[k] for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
                                    "early_stopping_rounds")}
    g = lcfg["search"]
    grid_rows, best, best_auc = [], None, -1.0
    for nl, lr, mcs in itertools.product(g["num_leaves"], g["learning_rate"], g["min_child_samples"]):
        p = {**default, "num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        m = FM.fit_lightgbm(Xtr, ytr, Xva, yva, p, cfg["seed"])
        s = FM.predict(m, Xva)
        auc = roc_auc(s[yva == 1], s[yva == 0])
        base = {"model": f"{ACT}|leaves={nl}|lr={lr:g}|min_child={mcs}", "split": "val", "protocol": "hard",
                "subset": "activity_label", "negative_type": "hard", "seed": cfg["seed"], "ci_low": None, "ci_high": None}
        grid_rows += [{**base, "metric": "roc_auc", "value": float(auc)},
                      {**base, "metric": "n_trees", "value": float(m.best_iteration_ or m.n_estimators)}]
        if round(auc, 6) > round(best_auc, 6):
            best, best_auc = p, auc
    write_results(grid_rows, res / "two_stage_activity_search.csv", append=False)
    log.info("stage 1 selected %s (val activity ROC-AUC %.4f)", best, best_auc)

    pfp, gp = search_best(cfg, PF), search_best(cfg, "lightgbm")
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    sc = {TS: {s: [] for s in args.splits}, ACT: {s: [] for s in args.splits}, "lightgbm": {s: [] for s in args.splits},
          PF: {s: [] for s in args.splits}}
    act_rows = []
    for seed in cfg["seeds"]:
        ma = FM.fit_lightgbm(Xtr, ytr, Xva, yva, best, seed)
        mp = E.fit("lightgbm", ALL_FEATURES, lmix, seed, pfp, active=True, d2_only=True)
        mg = E.fit("lightgbm", ALL_FEATURES, lmix, seed, gp)
        for split in args.splits:
            Xs = E.Xsets(split, ALL_FEATURES)
            pa, pp = FM.score_sets(ma, Xs), FM.score_sets(mp, Xs)
            sc[ACT][split].append(pa)
            sc[PF][split].append(pp)
            sc[TS][split].append({k: pa[k] * pp[k] for k in pa})
            sc["lightgbm"][split].append(FM.score_sets(mg, Xs))
            # Stage-1 quality on this split's neg_hard pairs (label = both active in the prediction year).
            sp_ = E.splits[split]
            y = active_both(cfg, auth, split, sp_["nodes"], sp_["neg_hard"]["u_idx"], sp_["neg_hard"]["v_idx"])
            s_ = pa["neg_hard"]
            act_rows.append({"model": ACT, "split": split, "protocol": "hard", "subset": "activity_label",
                             "negative_type": "hard", "seed": int(seed), "metric": "roc_auc",
                             "value": float(roc_auc(s_[y == 1], s_[y == 0])), "ci_low": None, "ci_high": None})
        log.info("seed %d done", seed)

    rows, tests = list(act_rows), []
    for split in args.splits:
        sp_ = E.splits[split]
        R = {m: raws(sp_, split, m, sc[m][split], PROTOS, cfg) for m in (TS, "lightgbm", PF)}
        for i, m in enumerate((TS, "lightgbm", PF)):
            rows += summary_rows(m if m == TS else f"{m}|two_stage_run", split, R[m], CELLS, cfg, n_boot, 81 + i)
        A = averaged(sp_, split, TS, sc[TS][split], PROTOS, cfg)
        B = averaged(sp_, split, "lightgbm", sc["lightgbm"][split], PROTOS, cfg)
        tests += test_rows(f"{TS} - lightgbm", split, A, B, TESTS, cfg, n_boot, 91)
        for r in rows + tests:
            if r["split"] == split and r["seed"] == -1 and not r["metric"].endswith("_sd"):
                print(split, r["model"], r["protocol"], r["subset"], r["metric"], round(r["value"], 4), flush=True)
    write_results(rows, res / "two_stage.csv", append=False)
    write_results(tests, res / "two_stage_tests.csv", append=False)


if __name__ == "__main__":
    main()
