"""Phase 7, step 1 (audit N8): extend the pair-focused LightGBM search one step beyond its selected corner.

  python scripts/grid_edge.py [--config configs/default.yaml]

Runs only the configs of `models.lightgbm.pair_focused_search` that are not yet in results/lgbm_search.csv, through
the same code path as run_feature_models.py (seed 0, train-split fit, validation pooled hard_active ROC-AUC), and
appends them. Validation only (rule 8): no test data is scored. Prints the selection before and after; if it
changes, the pair-focused model and everything built on it must be retrained (logged in docs/decisions.md).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from run_feature_models import Experiment, pf_search_config, select_mix  # noqa: E402

PF = "lightgbm_pair_focused"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    before = search_best(cfg, PF)
    done = set(pd.read_csv(res / "lgbm_search.csv").model)
    lcfg = cfg["models"]["lightgbm"]
    default = {k: lcfg[k] for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
                                    "early_stopping_rounds")}
    mix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    g = lcfg["pair_focused_search"]
    E = Experiment(cfg)
    rows = []
    for nl, lr, mcs in itertools.product(g["num_leaves"], g["learning_rate"], g["min_child_samples"]):
        if f"{PF}|leaves={nl}|lr={lr:g}|min_child={mcs}" in done:
            continue
        r, _ = pf_search_config(E, default, mix, cfg["seed"], PF, True, nl, lr, mcs)
        auc = next(x["value"] for x in r if x["metric"] == "roc_auc")
        print(f"leaves={nl} lr={lr:g} min_child={mcs}: val pooled hard_active ROC-AUC {auc:.4f}", flush=True)
        rows += r
    if rows:
        write_results(rows, res / "lgbm_search.csv", append=True)
    after = search_best(cfg, PF)
    keys = ("num_leaves", "learning_rate", "min_child_samples")
    print("selection before:", {k: before[k] for k in keys})
    print("selection after: ", {k: after[k] for k in keys}, "(changed)" if after != before else "(unchanged)")


if __name__ == "__main__":
    main()
