"""Link-prediction metrics with explicit tie handling.

Heuristics give many pairs identical scores (e.g. CN = 0), so ties are frequent and the
choice matters. Rules used everywhere:

- Scores are first snapped to ~12 significant digits (`snap`), so values that are equal
  in exact arithmetic but differ by float rounding (e.g. Adamic-Adar sums taken in a
  different order) count as ties.
- ROC-AUC: Mann-Whitney statistic with tied (positive, negative) pairs counted as 1/2,
  i.e. average ranks. A model that scores everything 0 gets exactly 0.5.
- Average precision: tied scores form a single threshold (sklearn's convention), so the
  order *within* a tie never matters.
- Ranking (Hits@K, MRR) uses the expectation under uniformly random tie-breaking. With
  g candidates scored strictly higher than the positive and e tied with it, the positive's
  rank is uniform on {g+1, ..., g+e+1}; Hits@K and the reciprocal rank are averaged over
  that range. Optimistic ranking (rank g+1) would reward heuristics for giving everything 0;
  pessimistic ranking would punish them unfairly.
- query_auc = 1 - (g + e/2) / n_candidates: the fraction of a query's candidates the
  positive beats (ties 1/2). Unlike MRR it does not depend on the number of candidates, so
  it is comparable across queries with different candidate counts (0.5 = random).
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

_EULER = 0.5772156649015329


def snap(x: np.ndarray, bits: int = 40) -> np.ndarray:
    """Round to `bits` mantissa bits (relative precision ~1e-12)."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 2.0**bits) / 2.0**bits, e)


# ---------------------------------------------------------------- classification

def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = snap(pos), snap(neg)
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        raise ValueError("need at least one positive and one negative")
    r = rankdata(np.concatenate([pos, neg]), method="average")
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = snap(pos), snap(neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(average_precision_score(y, np.r_[pos, neg]))


# ---------------------------------------------------------------- ranking

def rank_counts(pos_scores: np.ndarray, cand_scores: np.ndarray, query_id: np.ndarray, n_queries: int):
    """Per query: g = #candidates > positive, e = #candidates == positive, n = #candidates."""
    pos_scores, cand_scores = snap(pos_scores), snap(cand_scores)
    ref = pos_scores[query_id]
    g = np.bincount(query_id, weights=(cand_scores > ref), minlength=n_queries)
    e = np.bincount(query_id, weights=(cand_scores == ref), minlength=n_queries)
    n = np.bincount(query_id, minlength=n_queries)
    return g.astype(np.int64), e.astype(np.int64), n.astype(np.int64)


def _harmonic(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return np.where(k > 0, digamma(k + 1) + _EULER, 0.0)


def expected_reciprocal_rank(g: np.ndarray, e: np.ndarray) -> np.ndarray:
    """E[1/rank] with rank uniform on {g+1, ..., g+e+1}."""
    return (_harmonic(g + e + 1) - _harmonic(g)) / (e + 1)


def expected_hits(g: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """P(rank <= k) with rank uniform on {g+1, ..., g+e+1}."""
    return np.clip(k - g, 0, e + 1) / (e + 1)


def query_auc(g: np.ndarray, e: np.ndarray, n: np.ndarray) -> np.ndarray:
    return 1.0 - (g + 0.5 * e) / n


def per_query_metrics(g, e, n, ks=(20, 50, 100)) -> dict[str, np.ndarray]:
    out = {f"hits@{k}": expected_hits(g, e, k) for k in ks}
    out["mrr"] = expected_reciprocal_rank(g, e)
    out["query_auc"] = query_auc(g, e, n)
    return out
