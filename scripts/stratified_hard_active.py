"""Diagnostic: is `hard_active` inflated by newcomer status?

Activity matching makes newcomers rare among negatives (test: 17% of hard_active negatives vs 35%
of distance-2 positives involve an author with <= 2 works before the cutoff), because for a
newcomer "published in the prediction year" is close to "co-authored a new link that year". The
existing cold_start / established strata restrict only the POSITIVES, so a model that simply scores
newcomers high looks very good on them. This script re-scores hard_active with BOTH classes
restricted to the same stratum:

  established_both  positives and negatives with no newcomer endpoint
  newcomer_both     positives and negatives with >= 1 newcomer endpoint

and paired tests against Adamic-Adar inside each stratum. Newcomer status is pre-cutoff
information. Writes results/hard_active_stratified.csv and results/hard_active_stratified_paired.csv.

  python scripts/stratified_hard_active.py
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
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from gnn_analysis import fit_hybrid, load_scores  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

GNN_DIR = Path("data/interim/gnn")
GNNS = ["gnn_pair_focused_sage_structural", "gnn_general_sage_text"]
HYBRID_GNN = "gnn_pair_focused_sage_structural"


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    E = Experiment(cfg)
    res = Path(cfg["paths"]["results"])
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    gen_params, pf_params = search_best(cfg, "lightgbm"), search_best(cfg, "lightgbm_pair_focused")
    max_w = cfg["splits"]["cold_start_max_works"]
    n_boot = cfg["evaluation"]["n_bootstrap"]
    rows, prow = [], []
    for split in ("val", "test"):
        sp_ = E.splits[split]
        nw = sp_["nodes"].set_index("node_idx")["n_works_before"]
        new = lambda df: (nw.loc[df["u_idx"]].values <= max_w) | (nw.loc[df["v_idx"]].values <= max_w)
        d2 = (sp_["positives"]["distance"] == 2).values
        pos_new, neg_new = new(sp_["positives"])[d2], new(sp_["neg_hard_active"])
        strata = {"all": (np.ones(d2.sum(), bool), np.ones(len(neg_new), bool)),
                  "established_both": (~pos_new, ~neg_new), "newcomer_both": (pos_new, neg_new)}
        runs = {h: [(cfg["seed"], E.X[split]["positives"][h].values, E.X[split]["neg_hard_active"][h].values)]
                for h in ("adamic_adar", "jaccard")}
        for name, kw in (("lightgbm", dict(params=gen_params, active=False)), ("lightgbm_pair_focused", dict(params=pf_params, active=True))):
            runs[name] = []
            for s in cfg["seeds"]:
                m = E.fit("lightgbm", ALL_FEATURES, lmix, s, kw["params"], active=kw["active"])
                Xs = E.Xsets(split, ALL_FEATURES)
                runs[name].append((s, FM.predict(m, Xs["positives"]), FM.predict(m, Xs["neg_hard_active"])))
        for g in GNNS:
            runs[g] = [(s, load_scores(GNN_DIR / "final" / g / f"seed{s}", split)["positives"],
                        load_scores(GNN_DIR / "final" / g / f"seed{s}", split)["neg_hard_active"]) for s in cfg["seeds"]]
        hyb = f"lightgbm_pair_focused+{HYBRID_GNN}"
        runs[hyb] = []
        for s in cfg["seeds"]:
            sc = fit_hybrid(E, GNN_DIR, s, HYBRID_GNN, pf_params, lmix)[split]
            runs[hyb].append((s, sc["positives"], sc["neg_hard_active"]))
        _, aa_p, aa_n = runs["adamic_adar"][0]
        for name, rr in runs.items():
            for s, p, n in rr:
                p = p[d2]
                for sub, (pm, nm) in strata.items():
                    rng = np.random.default_rng([cfg["seed"], 66, list(strata).index(sub)])
                    base = {"model": name, "split": split, "protocol": "hard_active", "subset": sub,
                            "negative_type": "hard_active", "seed": s}
                    for metric, fn in (("roc_auc", roc_auc), ("average_precision", average_precision)):
                        lo, hi = bs.classification_ci(p[pm], n[nm], metric, rng, n_boot)
                        rows.append({**base, "metric": metric, "value": fn(p[pm], n[nm]), "ci_low": lo, "ci_high": hi})
                    rows.append({**base, "metric": "n_positives", "value": float(pm.sum())})
                    rows.append({**base, "metric": "n_negatives", "value": float(nm.sum())})
                    if name != "adamic_adar":
                        r = bs.paired_classification(p[pm], n[nm], aa_p[d2][pm], aa_n[nm], "roc_auc",
                                                     np.random.default_rng([cfg["seed"], 67, list(strata).index(sub)]), n_boot)
                        pb = {**base, "model": f"{name} - adamic_adar"}
                        prow.append({**pb, "metric": "roc_auc_diff", "value": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                        prow.append({**pb, "metric": "roc_auc_diff_p_value", "value": r["p_value"]})
        print(split, "done", flush=True)
    write_results(rows, res / "hard_active_stratified.csv", append=False)
    write_results(prow, res / "hard_active_stratified_paired.csv", append=False)


if __name__ == "__main__":
    main()
