"""Diagnostic: how much of the `hard` gain is "both authors are still active"?

Every positive needs both authors to publish in the prediction year; a hard negative does not.
Activity features (recent works, years since last work) can therefore separate classes by
predicting who keeps publishing rather than who collaborates. This script re-scores the `hard`
protocol with negatives restricted to hard negatives whose two authors BOTH have at least one
work in the prediction year (subset `neg_both_active`).

The restriction uses label-window information to choose the evaluation subset only; no model
sees it. It is an analysis of what the models learned, not a deployable evaluation setting.

  python scripts/activity_diagnostic.py   ->  results/activity_diagnostic.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import bootstrap as bs  # noqa: E402
from linkpred.eval.metrics import average_precision, roc_auc  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES, load_feature_tables  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from run_feature_models import ABLATIONS, Experiment, cols_for, select_mix  # noqa: E402

MODELS = {"lightgbm": ABLATIONS["all"], "lightgbm[temporal]": ABLATIONS["temporal"],
          "lightgbm[structural]": ABLATIONS["structural"], "lightgbm[structural+text]": ABLATIONS["structural+text"]}


def chosen_settings(cfg):
    res = Path(cfg["paths"]["results"])
    grid = pd.read_csv(res / "negative_mix_grid.csv").to_dict("records")
    mix = select_mix(grid, "lightgbm")[0]
    s = pd.read_csv(res / "lgbm_search.csv")
    best = s[s.metric == "roc_auc"].assign(v=lambda d: d.value.round(6))
    best = best.sort_values("v", ascending=False, kind="stable")["model"].iloc[0]
    kv = dict(x.split("=") for x in best.split("|")[1:])
    lcfg = cfg["models"]["lightgbm"]
    params = {"learning_rate": float(kv["lr"]), "num_leaves": int(kv["leaves"]),
              "min_child_samples": int(kv["min_child"]), "subsample": lcfg["subsample"],
              "colsample_bytree": lcfg["colsample_bytree"]}
    return mix, params


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    E = Experiment(cfg)
    mix, params = chosen_settings(cfg)
    auth = load_feature_tables(cfg["paths"]["processed"])["auth"]
    if "--report-only" in sys.argv:
        return regenerate_report(cfg, mix, params)
    rows, paired = [], []
    for split in ("val", "test"):
        s = cfg["splits"][split]
        active = set(auth.loc[auth["year"].between(s["cutoff"], s["window_end"]), "author_id"])
        sp_ = E.splits[split]
        neg = sp_["neg_hard"]
        keep = neg["author_u"].isin(active).values & neg["author_v"].isin(active).values
        d2 = (sp_["positives"]["distance"] == 2).values
        scorers = {"adamic_adar": [(cfg["seed"], E.X[split]["positives"]["adamic_adar"].values,
                                    E.X[split]["neg_hard"]["adamic_adar"].values)]}
        for name, fams in MODELS.items():
            cols = cols_for(fams)
            scorers[name] = []
            for seed in cfg["seeds"]:
                m = E.fit("lightgbm", cols, mix, seed, params)
                Xs = E.Xsets(split, cols)
                scorers[name].append((seed, FM.predict(m, Xs["positives"]), FM.predict(m, Xs["neg_hard"])))
        for name, runs in scorers.items():
            for seed, pos, negs in runs:
                for subset, nmask in (("all", np.ones(len(neg), bool)), ("neg_both_active", keep)):
                    p, n = pos[d2], negs[nmask]
                    rng = np.random.default_rng([cfg["seed"], 55, 0 if subset == "all" else 1])
                    for metric, fn in (("roc_auc", roc_auc), ("average_precision", average_precision)):
                        lo, hi = bs.classification_ci(p, n, metric, rng, cfg["evaluation"]["n_bootstrap"])
                        rows.append({"model": name, "split": split, "protocol": "hard", "subset": subset,
                                     "negative_type": "hard", "seed": seed, "metric": metric,
                                     "value": fn(p, n), "ci_low": lo, "ci_high": hi})
                    rows.append({"model": name, "split": split, "protocol": "hard", "subset": subset,
                                 "negative_type": "hard", "seed": seed, "metric": "n_negatives",
                                 "value": float(nmask.sum())})
            # Paired: each model/seed minus Adamic-Adar on identical pairs (test and val).
            if name == "adamic_adar":
                continue
            _, ap, an = scorers["adamic_adar"][0]
            for seed, pos, negs in runs:
                for subset, nmask in (("all", np.ones(len(neg), bool)), ("neg_both_active", keep)):
                    rng = np.random.default_rng([cfg["seed"], 56, 0 if subset == "all" else 1])
                    r = bs.paired_classification(pos[d2], negs[nmask], ap[d2], an[nmask], "roc_auc", rng,
                                                 cfg["evaluation"]["n_bootstrap"])
                    base = {"model": f"{name} - adamic_adar", "split": split, "protocol": "hard", "subset": subset,
                            "negative_type": "hard", "seed": seed}
                    paired.append({**base, "metric": "roc_auc_diff", "value": r["diff"],
                                   "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                    paired.append({**base, "metric": "roc_auc_diff_p_value", "value": r["p_value"]})
    write_results(rows, Path(cfg["paths"]["results"]) / "activity_diagnostic.csv", append=False)
    write_results(paired, Path(cfg["paths"]["results"]) / "activity_diagnostic_paired.csv", append=False)
    regenerate_report(cfg, mix, params)


def regenerate_report(cfg, mix, params):
    from linkpred.eval import report
    from linkpred.eval.feature_report import write_feature_report

    res = Path(cfg["paths"]["results"])
    grid = pd.read_csv(res / "negative_mix_grid.csv").to_dict("records")
    base = pd.read_csv(res / "baselines.csv")
    val_best = {}
    for p in ("hard", "rank_hard", "rank_random", "random"):
        subset, metric = report.HEADLINE[p]
        d = base[(base.split == "val") & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)]
        val_best[p] = d.sort_values(["value", "model"], ascending=[False, True])["model"].iloc[0]
    write_feature_report(cfg, {"logreg": select_mix(grid, "logreg")[0], "lightgbm": mix}, params, val_best)


if __name__ == "__main__":
    main()
