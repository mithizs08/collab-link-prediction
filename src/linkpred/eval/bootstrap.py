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


# ---------------------------------------------------------------- generic engine (audit M2 / M5)
# A "raw" is one of:
#   {"pos": array, "neg": array}                      pooled classification (ROC-AUC / AP)
#   {metric: per-query array, ...}                    ranking (mean of per-query values)
#   {"groups": [raw, raw, ...]}                       macro = unweighted mean of the groups' statistics
# Resampling is stratified the same way everywhere: positives and negatives separately, queries
# per group, each group independently. Group sizes depend only on the pairs, never on scores, so
# the same draw applies to every model and every seed on the same split.

def _draw(raw: dict, rng: np.random.Generator):
    if "groups" in raw:
        return [_draw(g, rng) for g in raw["groups"]]
    if "pos" in raw:
        return rng.integers(0, len(raw["pos"]), len(raw["pos"])), rng.integers(0, len(raw["neg"]), len(raw["neg"]))
    n = len(next(iter(raw.values())))
    return rng.integers(0, n, n)


def raw_stat(raw: dict, metric: str, idx=None) -> float:
    if "groups" in raw:
        return float(np.mean([raw_stat(g, metric, None if idx is None else idx[k]) for k, g in enumerate(raw["groups"])]))
    if "pos" in raw:
        fn = CLASSIFICATION_METRICS[metric]
        return fn(raw["pos"], raw["neg"]) if idx is None else fn(raw["pos"][idx[0]], raw["neg"][idx[1]])
    v = raw[metric]
    return float(v.mean() if idx is None else v[idx].mean())


def raw_ci(raw: dict, metric: str, rng: np.random.Generator, n_boot: int = 1000):
    vals = [raw_stat(raw, metric, _draw(raw, rng)) for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def seed_mean_ci(raws: list[dict], metric: str, rng: np.random.Generator, n_boot: int = 1000):
    """CI of the MEAN over seeds of a metric: each resample is applied to every seed's scores and
    the seeds' statistics are averaged (the audit's recommended interval for learned models)."""
    vals = []
    for _ in range(n_boot):
        d = _draw(raws[0], rng)
        vals.append(np.mean([raw_stat(r, metric, d) for r in raws]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def paired_raw(raw_a: dict, raw_b: dict, metric: str, rng: np.random.Generator, n_boot: int = 1000):
    """A minus B on the same pairs / queries / groups, identical resamples for both."""
    diffs = []
    for _ in range(n_boot):
        d = _draw(raw_a, rng)
        diffs.append(raw_stat(raw_a, metric, d) - raw_stat(raw_b, metric, d))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": raw_stat(raw_a, metric) - raw_stat(raw_b, metric), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": _p_value(diffs)}


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
