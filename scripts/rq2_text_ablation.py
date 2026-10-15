"""RQ2: do topical signals add predictive value beyond network structure? (docs/decisions.md, "RQ2 design")

  python scripts/rq2_text_ablation.py --splits val          # validation first (rule 8)
  python scripts/rq2_text_ablation.py --splits val test     # final

Pair-focused LightGBM (activity-matched 2023 negatives, distance-2 positives; main selected parameters and mix,
not re-tuned per feature set) on three feature sets -- structural, structural+text, all -- 5 seeds each.
Paired tests on seed-averaged scores (5,000 resamples): structural+text - structural on macro hard_active /
rank_hard_active and within each group; all - structural+text on the two macro metrics. Raw p only: Holm is applied
across the predefined family in final_comparison.py.

Outputs: results/rq2_text_ablation.csv, results/rq2_text_ablation_tests.csv.

  python scripts/rq2_text_ablation.py --block-order --splits val test

EXPLORATORY (Phase 8, step 2; not in the Holm family): the same model on structural+temporal (no text), so the order
of adding blocks is tested both ways: structural+temporal - structural, and all - structural+temporal (= text added
last). Outputs: results/rq2_block_order_exploratory.csv, results/rq2_block_order_exploratory_tests.csv (raw p only).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.features.pair_features import FAMILIES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from phase7_common import averaged, n_boot_final, raws, summary_rows, test_rows  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

log = logging.getLogger("rq2")
PF = "lightgbm_pair_focused"
SETS = {"structural": ["structural"], "structural+text": ["structural", "text"],
        "all": ["structural", "temporal", "text"]}
PROTOS = ("hard_active", "rank_hard_active")
HA_M, RK_M = ("hard_active", "macro", "roc_auc"), ("rank_hard_active", "macro", "query_auc")
GROUPS = (("hard_active", "established_matched", "roc_auc"), ("hard_active", "newcomer_matched", "roc_auc"),
          ("rank_hard_active", "established_matched", "query_auc"), ("rank_hard_active", "newcomer_matched", "query_auc"))
CELLS = (HA_M, RK_M) + GROUPS
TESTS = (("structural+text", "structural", (HA_M, RK_M) + GROUPS), ("all", "structural+text", (HA_M, RK_M)))


def name(label: str) -> str:
    return f"{PF}[{label}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    ap.add_argument("--block-order", action="store_true", help="exploratory: structural+temporal set (not in family)")
    args = ap.parse_args(argv)
    sets, tests_def, out = SETS, TESTS, "rq2_text_ablation"
    if args.block_order:
        sets = {"structural": ["structural"], "structural+temporal": ["structural", "temporal"],
                "all": ["structural", "temporal", "text"]}
        tests_def = (("structural+temporal", "structural", (HA_M, RK_M) + GROUPS),
                     ("all", "structural+temporal", (HA_M, RK_M) + GROUPS))
        out = "rq2_block_order_exploratory"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    n_boot = n_boot_final(cfg)
    E = Experiment(cfg)
    pfp = search_best(cfg, PF)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    sc = {lab: {s: [] for s in args.splits} for lab in sets}
    for lab, fams in sets.items():
        cols = [c for f in fams for c in FAMILIES[f]]
        for seed in cfg["seeds"]:
            m = E.fit("lightgbm", cols, lmix, seed, pfp, active=True, d2_only=True)
            for split in args.splits:
                sc[lab][split].append(FM.score_sets(m, E.Xsets(split, cols)))
        log.info("%s (%d features) done", lab, len(cols))
    rows, tests = [], []
    for split in args.splits:
        sp_ = E.splits[split]
        for i, lab in enumerate(sets):
            rows += summary_rows(name(lab), split, raws(sp_, split, name(lab), sc[lab][split], PROTOS, cfg), CELLS, cfg,
                                 n_boot, 101 + i)
        A = {lab: averaged(sp_, split, name(lab), sc[lab][split], PROTOS, cfg) for lab in sets}
        for k, (a, b, cells) in enumerate(tests_def):
            tests += test_rows(f"{name(a)} - {name(b)}", split, A[a], A[b], cells, cfg, n_boot, 111 + k)
        for r in rows + tests:
            if r["split"] == split and not r["metric"].endswith("_sd"):
                print(split, r["model"], r["protocol"], r["subset"], r["metric"], round(r["value"], 4), flush=True)
    write_results(rows, res / f"{out}.csv", append=False)
    write_results(tests, res / f"{out}_tests.csv", append=False)


if __name__ == "__main__":
    main()
