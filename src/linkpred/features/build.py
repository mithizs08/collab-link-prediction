"""Compute and cache pair features for every pair set of every split.

data/processed/features/<split>.parquet   one row per unique unordered pair (u_idx < v_idx)
report/features/coverage.csv              embedding coverage per split
results/half_life_selection.csv           validation choice of the recency half-life

Selection rules (validation only, docs/decisions.md):
  katz_beta   reused from results/katz_beta_selection.csv (Phase 3)
  half_life   grid in config; maximises validation `hard` ROC-AUC of recency-weighted AA alone
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from linkpred.data.splits import SPLITS, build_input_graph, load_split
from linkpred.eval.metrics import roc_auc
from linkpred.features.embed import load_embeddings
from linkpred.features.pair_features import ALL_FEATURES, FeatureContext, load_feature_tables

log = logging.getLogger(__name__)

PAIR_SETS = {
    "train": [("positives", "u_idx", "v_idx"), ("train_pool_random", "u_idx", "v_idx"),
              ("train_pool_hard", "u_idx", "v_idx")],
    "eval": [("positives", "u_idx", "v_idx"), ("neg_random", "u_idx", "v_idx"), ("neg_hard", "u_idx", "v_idx"),
             ("ranking", "head_idx", "cand_idx"), ("ranking_hard", "head_idx", "cand_idx")],
}


def pair_sets(split_name: str):
    return PAIR_SETS["train" if split_name == "train" else "eval"]


def canonical(u, v):
    u, v = np.asarray(u), np.asarray(v)
    return np.minimum(u, v), np.maximum(u, v)


def orient_by_frequency(u, v):
    """Put the more frequent endpoint first, so per-source searches (distance, Katz) run from
    few distinct sources (ranking heads appear 100x). Features are symmetric, so this is safe."""
    cnt = np.bincount(np.r_[u, v])
    swap = cnt[v] > cnt[u]
    return np.where(swap, v, u), np.where(swap, u, v)


def katz_beta_from_selection(results_dir: Path) -> float:
    k = pd.read_csv(results_dir / "katz_beta_selection.csv")
    k = k[k["metric"] == "roc_auc"].copy()
    k["beta"] = k["model"].str.split("=").str[1].astype(float)
    k["v"] = k["value"].round(6)
    return float(k.sort_values(["v", "beta"], ascending=[False, True])["beta"].iloc[0])


def make_context(cfg, tables, split_name, split, emb, emb_ids, params) -> FeatureContext:
    return FeatureContext(tables, cfg["splits"][split_name]["cutoff"], split["nodes"], split["graph_edges"],
                          split["A"], emb, emb_ids, params)


def select_half_life(ctx: FeatureContext, val: dict, grid) -> tuple[float, list[dict]]:
    pos = val["positives"][val["positives"]["distance"] == 2]
    neg = val["neg_hard"]
    rows = []
    for h in grid:
        _, aa_p = ctx.rw_scores(pos["u_idx"].values, pos["v_idx"].values, h)
        _, aa_n = ctx.rw_scores(neg["u_idx"].values, neg["v_idx"].values, h)
        rows.append({"model": f"rw_adamic_adar_half_life={h:g}", "split": "val", "protocol": "hard",
                     "subset": "all", "negative_type": "hard", "seed": 0, "metric": "roc_auc",
                     "value": roc_auc(aa_p, aa_n)})
    best = max(grid, key=lambda h: (round(next(r["value"] for r in rows if r["model"].endswith(f"={h:g}")), 6), -h))
    return best, rows


def build_features(cfg: dict) -> dict:
    from linkpred.results import write_results

    t0 = time.time()
    root = Path(cfg["paths"]["splits"])
    out = Path(cfg["paths"]["processed"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    tables = load_feature_tables(cfg["paths"]["processed"])
    emb, emb_ids = load_embeddings(cfg)
    fcfg = cfg["features"]
    params = {"seed": cfg["seed"], "katz_beta": katz_beta_from_selection(Path(cfg["paths"]["results"])),
              "half_life": None}

    splits = {s: load_split(root, s) for s in SPLITS}
    val_ctx = make_context(cfg, tables, "val", splits["val"], emb, emb_ids, params)
    params["half_life"], hl_rows = select_half_life(val_ctx, splits["val"], fcfg["half_life_grid"])
    write_results(hl_rows, Path(cfg["paths"]["results"]) / "half_life_selection.csv", append=False)
    log.info("katz beta %g, half-life %g (validation)", params["katz_beta"], params["half_life"])

    coverage = []
    for name in SPLITS:
        ctx = val_ctx if name == "val" else make_context(cfg, tables, name, splits[name], emb, emb_ids, params)
        ctx.p = params
        us, vs = [], []
        for table, a, b in pair_sets(name):
            cu, cv = canonical(splits[name][table][a], splits[name][table][b])
            us.append(cu); vs.append(cv)
        pairs = pd.DataFrame({"u_idx": np.concatenate(us), "v_idx": np.concatenate(vs)}).drop_duplicates()
        su, sv = orient_by_frequency(pairs["u_idx"].values, pairs["v_idx"].values)
        feats = ctx.compute(su, sv)
        df = pd.concat([pairs.reset_index(drop=True), feats], axis=1)
        df.to_parquet(out / f"{name}.parquet", index=False)
        coverage.append({"split": name, **ctx.coverage()})
        log.info("%s: %d pairs, %.0fs", name, len(df), time.time() - t0)
        del ctx
    rep = Path(cfg["paths"]["report"]) / "features"
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(rep / "coverage.csv", index=False)
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    return params


def load_features(cfg: dict, split_name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg["paths"]["processed"]) / "features" / f"{split_name}.parquet")


def features_for(feat: pd.DataFrame, u, v, columns=ALL_FEATURES) -> pd.DataFrame:
    """Feature rows aligned with the pairs (u, v) in the given order."""
    cu, cv = canonical(u, v)
    key = pd.MultiIndex.from_arrays([feat["u_idx"].values, feat["v_idx"].values])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([cu, cv]))
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} pairs have no cached features")
    return feat[list(columns)].iloc[pos].reset_index(drop=True)
