"""Diagnostic (audit M4 / N4): does the Procrustes alignment transfer a node2vec classifier from train to validation?

  python scripts/node2vec_alignment_check.py [--config configs/default.yaml]

Validation only (rule 8); no test data is read. Seed-0 embeddings at the selected p, q (from the cache written by
run_node2vec.py; trained here with the same settings if missing). Pairs: validation `random` protocol
(all positives vs random negatives). For each edge operator:
  transferred  logistic regression fitted on train-split pairs (Phase 4 mix: hard fraction 0, 1:3) with the train
               embedding, applied to the validation embedding aligned to train (what run_node2vec.py does);
  within_val   the same pipeline fitted and scored inside validation by 5-fold CV (unshuffled, so deterministic),
               with no transfer: an upper reference for what the operator can extract from the embedding.
If transferred ~= within_val, the alignment loses little; a low within_val means the operator itself is weak.

Output: results/node2vec_alignment_check.csv (model = node2vec_alignment|op=<op>|<transferred|within_val>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.splits import load_split  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.models import node2vec as N2V  # noqa: E402
from linkpred.models.feature_models import training_pairs  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from run_feature_models import select_mix  # noqa: E402

OPERATORS = ("hadamard", "l1", "l2")
MIX = (0.0, 3)


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def selected_pq(cfg: dict, res: Path) -> tuple[float, float]:
    """p, q of the reported node2vec model, by run_node2vec.py's validation rule applied to node2vec_grid.csv."""
    grid = pd.read_csv(res / "node2vec_grid.csv").to_dict("records")
    ch = []
    for p in cfg["models"]["node2vec"]["p_grid"]:
        for q in cfg["models"]["node2vec"]["q_grid"]:
            for op in cfg["models"]["node2vec"]["operators"]:
                kind = f"node2vec|p={p:g}|q={q:g}|op={op}"
                mix, _ = select_mix(grid, kind)
                name = f"{kind}|hard={mix[0]:g}|1:{mix[1]}"
                ha = next(x["value"] for x in grid if x["model"] == name and x["protocol"] == "hard_active"
                          and x["subset"] == "all" and x["metric"] == "roc_auc")
                ch.append({"p": p, "q": q, "v": round(ha, 6)})
    best = pd.DataFrame(ch).sort_values(["v", "p", "q"], ascending=[False, True, True], kind="stable").iloc[0]
    return float(best["p"]), float(best["q"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    seed = cfg["seed"]
    p, q = selected_pq(cfg, res)
    ncfg = cfg["models"]["node2vec"]
    sp = {s: load_split(Path(cfg["paths"]["processed"]) / "splits", s) for s in ("train", "val")}
    cache = Path(cfg["paths"]["interim"]) / "node2vec"
    emb = {s: N2V.cached_embeddings(cache, s, sp[s]["A"], p, q, seed, ncfg) for s in sp}
    aligned = N2V.align_to(emb["val"], sp["val"]["nodes"], emb["train"], sp["train"]["nodes"])

    v = sp["val"]
    U = np.r_[v["positives"].u_idx.values, v["neg_random"].u_idx.values]
    V = np.r_[v["positives"].v_idx.values, v["neg_random"].v_idx.values]
    y = np.r_[np.ones(len(v["positives"])), np.zeros(len(v["neg_random"]))]
    pairs, ytr = training_pairs(sp["train"], *MIX)

    rows = []
    for op in OPERATORS:
        lr = lr_model().fit(N2V.edge_features(emb["train"], pairs.u_idx.values, pairs.v_idx.values, op), ytr)
        s = lr.predict_proba(N2V.edge_features(aligned, U, V, op))[:, 1]
        transferred = roc_auc(s[y == 1], s[y == 0])
        within = cross_val_score(lr_model(), N2V.edge_features(emb["val"], U, V, op), y, cv=5, scoring="roc_auc").mean()
        for kind, val in (("transferred", transferred), ("within_val", within)):
            rows.append({"model": f"node2vec_alignment|op={op}|{kind}", "split": "val", "protocol": "random",
                         "subset": "all", "negative_type": "random", "seed": int(seed), "metric": "roc_auc",
                         "value": float(val), "ci_low": None, "ci_high": None})
        print(f"p={p:g} q={q:g} {op}: transferred {transferred:.3f}, within-val 5-fold {within:.3f}")
    write_results(rows, res / "node2vec_alignment_check.csv", append=False)


if __name__ == "__main__":
    main()
