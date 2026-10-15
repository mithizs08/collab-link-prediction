"""Shared helpers for the Phase 7 robustness scripts (disambiguation sensitivity, two-stage model).

Same conventions as scripts/final_comparison.py: learned models are summarised by the mean over seeds with a
seed-mean bootstrap CI; tests are one paired bootstrap on the seed-AVERAGED scores. Raw p-values are written
here; Holm correction is applied in final_comparison.py across the whole predefined family (docs/decisions.md).
"""

from __future__ import annotations

import numpy as np

from linkpred.eval import bootstrap as bs
from linkpred.eval.protocols import PairScores, evaluate, paired_test

NEG_TYPE = {"hard": "hard", "rank_hard": "hard", "hard_active": "hard_active", "rank_hard_active": "hard_active",
            "random": "random", "rank_random": "random"}


def n_boot_final(cfg: dict) -> int:
    """Resamples for final reporting (Phase 7): evaluation.n_bootstrap_final."""
    return int(cfg["evaluation"]["n_bootstrap_final"])


def raws(split: dict, split_name: str, model: str, score_list: list[dict], protos, cfg) -> list[dict]:
    """Per-seed raw evaluation structures (no bootstrap) for each seed's scores."""
    k_full = cfg["negatives"]["ranking_candidates"]
    return [evaluate(None, split, split_name, model, 0, cfg["seed"], 0, protocols=protos, k_full=k_full,
                     precomputed=sc)[1] for sc in score_list]


def averaged(split: dict, split_name: str, model: str, score_list: list[dict], protos, cfg) -> dict:
    """Raw evaluation structure of the seed-AVERAGED scores (per pair: mean over seeds)."""
    avg = {k: np.mean([sc[k] for sc in score_list], axis=0) for k in score_list[0]}
    return raws(split, split_name, model, [avg], protos, cfg)[0]


def summary_rows(model: str, split_name: str, R: list[dict], cells, cfg, n_boot: int, rng_tag: int) -> list[dict]:
    """Mean over seeds + seed-mean bootstrap CI (+ seed sd) for each (protocol, subset, metric) cell."""
    rows = []
    for k, (p, sub, metric) in enumerate(cells):
        vals = [bs.raw_stat(r[p][sub], metric) for r in R]
        rng = np.random.default_rng([cfg["seed"], rng_tag, k, 0 if split_name == "val" else 1])
        lo, hi = bs.seed_mean_ci([r[p][sub] for r in R], metric, rng, n_boot)
        seed = -1 if len(vals) > 1 else cfg["seed"]
        base = {"model": model, "split": split_name, "protocol": p, "subset": sub, "negative_type": NEG_TYPE[p],
                "seed": seed}
        rows.append({**base, "metric": metric, "value": float(np.mean(vals)), "ci_low": lo, "ci_high": hi})
        rows.append({**base, "metric": f"{metric}_seed_sd", "value": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                     "ci_low": None, "ci_high": None})
    return rows


def test_rows(name: str, split_name: str, A: dict, B: dict, cells, cfg, n_boot: int, rng_tag: int) -> list[dict]:
    """One paired bootstrap per cell on seed-averaged raws: diff [CI] and raw p-value (Holm is applied later)."""
    out = []
    for j, (p, sub, metric) in enumerate(cells):
        rng = np.random.default_rng([cfg["seed"], rng_tag, j, 0 if split_name == "val" else 1])
        r = paired_test(A, B, p, sub, metric, rng, n_boot)
        base = {"model": name, "split": split_name, "protocol": p, "subset": sub, "negative_type": NEG_TYPE[p], "seed": -1}
        out += [{**base, "metric": f"{metric}_diff", "value": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]},
                {**base, "metric": f"{metric}_diff_p_value", "value": r["p_value"], "ci_low": None, "ci_high": None}]
    return out


def score_dict(split: dict, fn) -> dict:
    """{pair set: fn(table rows)} over every evaluation pair set of a split."""
    return {k: fn(split[t][a].values, split[t][b].values) for k, (t, a, b) in PairScores.SOURCES.items()}
