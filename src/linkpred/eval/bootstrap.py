"""Bootstrap confidence intervals and paired bootstrap tests.

- Classification protocols resample positives and negatives *separately* (stratified),
  keeping the class sizes fixed, so every resample has the same base rate.
- Ranking protocols resample queries (positives) and average the per-query metric.
- Paired tests apply the SAME resample indices to both models' scores on the same pairs,
  so the CI of the difference reflects only model differences, not pair sampling twice.
  The two-sided p-value is 2 * min(P(diff <= 0), P(diff >= 0)) over resamples, floored
  at 1 / n_boot (a resampled test cannot resolve smaller p-values).
- Intervals are percentile intervals (2.5%, 97.5%).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from linkpred.eval.metrics import average_precision, roc_auc

CLASSIFICATION_METRICS: dict[str, Callable] = {"roc_auc": roc_auc, "average_precision": average_precision}


def _indices(rng: np.random.Generator, n: int, n_boot: int) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def classification_ci(pos, neg, metric: str, rng: np.random.Generator, n_boot: int = 1000):
    fn = CLASSIFICATION_METRICS[metric]
    ip, ineg = _indices(rng, len(pos), n_boot), _indices(rng, len(neg), n_boot)
    vals = np.array([fn(pos[a], neg[b]) for a, b in zip(ip, ineg)])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def mean_ci(per_query: np.ndarray, rng: np.random.Generator, n_boot: int = 1000):
    idx = _indices(rng, len(per_query), n_boot)
    vals = per_query[idx].mean(axis=1)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _p_value(diffs: np.ndarray) -> float:
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(diffs))))


def paired_classification(pos_a, neg_a, pos_b, neg_b, metric: str, rng: np.random.Generator, n_boot: int = 1000):
    """Model A minus model B on the same positives / negatives (arrays aligned by pair)."""
    assert len(pos_a) == len(pos_b) and len(neg_a) == len(neg_b), "paired test needs the same pairs"
    fn = CLASSIFICATION_METRICS[metric]
    ip, ineg = _indices(rng, len(pos_a), n_boot), _indices(rng, len(neg_a), n_boot)
    diffs = np.array([fn(pos_a[a], neg_a[b]) - fn(pos_b[a], neg_b[b]) for a, b in zip(ip, ineg)])
    point = fn(pos_a, neg_a) - fn(pos_b, neg_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_value": _p_value(diffs)}


def paired_mean(vals_a, vals_b, rng: np.random.Generator, n_boot: int = 1000):
    """Per-query metric of A minus B on the same queries."""
    assert len(vals_a) == len(vals_b), "paired test needs the same queries"
    d = np.asarray(vals_a) - np.asarray(vals_b)
    idx = _indices(rng, len(d), n_boot)
    diffs = d[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": _p_value(diffs)}
