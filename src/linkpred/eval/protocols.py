"""Evaluation protocols (decided 2026-10-14, see docs/decisions.md).

Every model is evaluated with the same five protocols on the same saved pairs:

hard                    distance-2 positives vs hard negatives (all at distance 2).  HEADLINE.
rank_random             each positive vs its 100 random ranking candidates: Hits@20/50/100, MRR.
rank_hard               each distance-2 positive vs up to 100 candidates at distance 2 from its
                        head. Queries with fewer candidates are reported as their own stratum.
random                  all positives vs random negatives (easy; for comparability with the literature).
diagnostic_all_vs_hard  all positives vs hard negatives. APPENDIX ONLY: distance >= 3 identifies
                        positives perfectly here, so this measures an artefact of the design.
hard_active             distance-2 positives vs hard negatives whose two authors both publish in the
                        prediction year (activity-matched). LEAD pair-level metric from Phase 5 on.
rank_hard_active        rank_hard with candidates restricted to authors active in the prediction year.

A scoring function has the signature score_fn(split, u_idx, v_idx) -> np.ndarray, higher =
more likely to collaborate, where `split` is the dict returned by splits.load_split and
indices are that split's node indices. It must only use the split's input graph / node table.

Bootstrap resamples depend on (boot_seed, protocol, split) but not on the model, so all models
are resampled identically and paired comparisons are possible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from linkpred.eval import bootstrap as bs
from linkpred.eval.metrics import average_precision, per_query_metrics, rank_counts, roc_auc

ScoreFn = Callable[[dict, np.ndarray, np.ndarray], np.ndarray]
# Appended protocols go at the END: bootstrap streams are keyed by position, so existing
# protocols keep identical resamples.
PROTOCOL_ORDER = ("hard", "rank_random", "rank_hard", "random", "diagnostic_all_vs_hard",
                  "hard_active", "rank_hard_active")
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
RANK_KS = (20, 50, 100)          # rank_random
RANK_KS_HARD = (10, 20, 50)      # rank_hard: Hits@100 is near-trivial with <= 101 items (Phase 3)


class PairScores:
    """Scores each saved pair set at most once per (model, split)."""

    SOURCES = {
        "positives": ("positives", "u_idx", "v_idx"),
        "neg_random": ("neg_random", "u_idx", "v_idx"),
        "neg_hard": ("neg_hard", "u_idx", "v_idx"),
        "ranking": ("ranking", "head_idx", "cand_idx"),
        "ranking_hard": ("ranking_hard", "head_idx", "cand_idx"),
        "neg_hard_active": ("neg_hard_active", "u_idx", "v_idx"),
        "ranking_hard_active": ("ranking_hard_active", "head_idx", "cand_idx"),
    }

    def __init__(self, score_fn: ScoreFn | None, split: dict, precomputed: dict | None = None):
        self.score_fn, self.split, self._cache = score_fn, split, dict(precomputed or {})
        for name, sc in self._cache.items():
            table = self.SOURCES[name][0]
            if len(sc) != len(split[table]) or not np.isfinite(sc).all():
                raise ValueError(f"precomputed scores for {name} do not match the split")

    def __call__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            if self.score_fn is None:
                raise KeyError(f"no scores for {name}")
            table, u, v = self.SOURCES[name]
            df = self.split[table]
            s = np.asarray(self.score_fn(self.split, df[u].values, df[v].values), dtype=np.float64)
            if s.shape != (len(df),) or not np.isfinite(s).all():
                raise ValueError(f"score_fn returned bad scores for {name}")
            self._cache[name] = s
        return self._cache[name]


def _scores(scorer, split) -> PairScores:
    return scorer if isinstance(scorer, PairScores) else PairScores(scorer, split)


def _rng(boot_seed: int, protocol: str, split_name: str) -> np.random.Generator:
    return np.random.default_rng([boot_seed, PROTOCOL_ORDER.index(protocol), SPLIT_INDEX[split_name]])


def _row(ctx, protocol, subset, neg_type, metric, value, ci=(None, None)):
    return {"model": ctx["model"], "split": ctx["split_name"], "protocol": protocol, "subset": subset,
            "negative_type": neg_type, "seed": ctx["seed"], "metric": metric, "value": float(value),
            "ci_low": None if ci[0] is None else float(ci[0]), "ci_high": None if ci[1] is None else float(ci[1])}


def _classification(ctx, protocol, neg_type, subsets: dict, neg: np.ndarray, n_boot: int):
    rows, raw = [], {}
    for subset, pos in subsets.items():
        if len(pos) == 0:
            continue
        rng = _rng(ctx["boot_seed"], protocol, ctx["split_name"])
        for metric, fn in (("roc_auc", roc_auc), ("average_precision", average_precision)):
            ci = bs.classification_ci(pos, neg, metric, rng, n_boot) if n_boot else (None, None)
            rows.append(_row(ctx, protocol, subset, neg_type, metric, fn(pos, neg), ci))
        rows.append(_row(ctx, protocol, subset, neg_type, "n_positives", len(pos)))
        rows.append(_row(ctx, protocol, subset, neg_type, "n_negatives", len(neg)))
        raw[subset] = {"pos": pos, "neg": neg}
    return rows, raw


def _ranking(ctx, protocol, neg_type, subsets: dict, n_boot: int, metrics_by_subset: dict):
    rows, raw = [], {}
    for subset, pq in subsets.items():
        if len(pq["mrr"]) == 0:
            continue
        rng = _rng(ctx["boot_seed"], protocol, ctx["split_name"])
        for metric in metrics_by_subset[subset]:
            vals = pq[metric]
            ci = bs.mean_ci(vals, rng, n_boot) if n_boot else (None, None)
            rows.append(_row(ctx, protocol, subset, neg_type, metric, vals.mean(), ci))
        rows.append(_row(ctx, protocol, subset, neg_type, "n_queries", len(pq["mrr"])))
        raw[subset] = pq
    return rows, raw


# ---------------------------------------------------------------- protocols

def protocol_hard(scorer, split, ctx, n_boot=1000):
    s = _scores(scorer, split)
    pos_df = split["positives"]
    d2 = (pos_df["distance"] == 2).values
    pos = s("positives")
    cold = pos_df["cold_start"].values
    subsets = {"all": pos[d2], "cold_start": pos[d2 & cold], "established": pos[d2 & ~cold]}
    return _classification(ctx, "hard", "hard", subsets, s("neg_hard"), n_boot)


def protocol_random(scorer, split, ctx, n_boot=1000):
    s = _scores(scorer, split)
    pos_df = split["positives"]
    d, cold, pos = pos_df["distance"].values, pos_df["cold_start"].values, s("positives")
    subsets = {
        "all": pos,
        "distance=2": pos[d == 2], "distance=3": pos[d == 3],
        "distance=4+": pos[(d >= 4) & np.isfinite(d)], "disconnected": pos[~np.isfinite(d)],
        "cold_start": pos[cold], "established": pos[~cold],
    }
    return _classification(ctx, "random", "random", subsets, s("neg_random"), n_boot)


def protocol_diagnostic_all_vs_hard(scorer, split, ctx, n_boot=1000):
    s = _scores(scorer, split)
    return _classification(ctx, "diagnostic_all_vs_hard", "hard", {"all": s("positives")}, s("neg_hard"), n_boot)


def _per_query(s, split, cand_table):
    rk = split[cand_table]
    q = rk["query_id"].values
    pos = s("positives")
    g, e, n = rank_counts(pos, s(cand_table), q, len(split["positives"]))
    return g, e, n


def protocol_rank_random(scorer, split, ctx, n_boot=1000):
    s = _scores(scorer, split)
    g, e, n = _per_query(s, split, "ranking")
    pq = per_query_metrics(g, e, n, RANK_KS)
    metrics = [f"hits@{k}" for k in RANK_KS] + ["mrr", "query_auc"]
    return _ranking(ctx, "rank_random", "random", {"all": pq}, n_boot, {"all": metrics})


def protocol_hard_active(scorer, split, ctx, n_boot=1000):
    """Distance-2 positives vs hard negatives whose authors BOTH publish in the prediction year.

    Removes the "is the author still active?" shortcut (Phase 4 finding). Activity selects the
    evaluation negatives only; no model input contains it.
    """
    s = _scores(scorer, split)
    pos_df = split["positives"]
    d2 = (pos_df["distance"] == 2).values
    pos, cold = s("positives"), pos_df["cold_start"].values
    subsets = {"all": pos[d2], "cold_start": pos[d2 & cold], "established": pos[d2 & ~cold]}
    return _classification(ctx, "hard_active", "hard_active", subsets, s("neg_hard_active"), n_boot)


def protocol_rank_hard_active(scorer, split, ctx, n_boot=1000, k_full=100):
    """rank_hard with candidates restricted to authors active in the prediction year."""
    return _rank_hard_like(scorer, split, ctx, n_boot, k_full, "ranking_hard_active", "rank_hard_active", "hard_active")


def protocol_rank_hard(scorer, split, ctx, n_boot=1000, k_full=100):
    return _rank_hard_like(scorer, split, ctx, n_boot, k_full, "ranking_hard", "rank_hard", "hard")


def _rank_hard_like(scorer, split, ctx, n_boot, k_full, table, protocol, neg_type):
    """Strata by candidate count; MRR / Hits@K are only comparable within a stratum.

    full_candidates     queries with exactly k_full candidates (headline rank_hard numbers)
    partial_candidates  1..k_full-1 candidates; reported separately (ranks are shorter)
    all                 every query with >= 1 candidate, query_auc only (count-independent)
    cold_start / established   all queries split by the positive's cold-start flag, query_auc only
    """
    s = _scores(scorer, split)
    g, e, n = _per_query(s, split, table)
    has = n > 0
    pq = per_query_metrics(g, e, np.maximum(n, 1), RANK_KS_HARD)
    take = lambda m: {k: v[m] for k, v in pq.items()}
    full, partial = has & (n == k_full), has & (n < k_full)
    cold = split["positives"]["cold_start"].values
    metrics = [f"hits@{k}" for k in RANK_KS_HARD] + ["mrr", "query_auc"]
    subsets = {"full_candidates": take(full), "partial_candidates": take(partial), "all": take(has),
               "cold_start": take(has & cold), "established": take(has & ~cold)}
    qa = ["query_auc"]
    return _ranking(ctx, protocol, neg_type, subsets, n_boot,
                    {"full_candidates": metrics, "partial_candidates": metrics, "all": qa,
                     "cold_start": qa, "established": qa})


PROTOCOLS = {
    "hard": protocol_hard,
    "rank_random": protocol_rank_random,
    "rank_hard": protocol_rank_hard,
    "random": protocol_random,
    "diagnostic_all_vs_hard": protocol_diagnostic_all_vs_hard,
    "hard_active": protocol_hard_active,
    "rank_hard_active": protocol_rank_hard_active,
}
RANK_HARD_LIKE = {"rank_hard", "rank_hard_active"}


def evaluate(score_fn: ScoreFn | None, split: dict, split_name: str, model: str, seed: int, boot_seed: int,
             n_boot: int = 1000, protocols=PROTOCOL_ORDER, k_full: int = 100, precomputed: dict | None = None):
    """Run protocols; return (result rows, raw per-protocol scores for paired tests).

    Either score_fn(split, u, v) or precomputed = {pair-set name: scores aligned with that table}.
    """
    ctx = {"model": model, "split_name": split_name, "seed": seed, "boot_seed": boot_seed}
    s = PairScores(score_fn, split, precomputed)
    rows, raw = [], {}
    for p in protocols:
        fn = PROTOCOLS[p]
        r, x = fn(s, split, ctx, n_boot, k_full) if p in RANK_HARD_LIKE else fn(s, split, ctx, n_boot)
        rows.extend(r)
        raw[p] = x
    return rows, raw
