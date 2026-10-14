"""Protocols and bootstrap on a hand-made split."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from linkpred.eval import bootstrap as bs
from linkpred.eval.protocols import PROTOCOL_ORDER, evaluate
from linkpred.results import validate_row


@pytest.fixture
def split():
    n = 40          # candidates / negatives use indices >= n so they never equal a positive
    rng = np.random.default_rng(0)
    dist = np.array([2, 2, 2, 3, 4, np.inf, 2, 3] * 3, dtype=float)
    P = len(dist)
    pos = pd.DataFrame({"u_idx": np.arange(P), "v_idx": np.arange(P) + 1, "distance": dist,
                        "cold_start": np.arange(P) % 3 == 0})
    neg = lambda k: pd.DataFrame({"u_idx": rng.integers(0, n, k), "v_idx": rng.integers(0, n, k) + n})
    rk = pd.DataFrame({"query_id": np.repeat(np.arange(P), 10), "head_idx": np.repeat(pos["u_idx"], 10),
                       "cand_idx": rng.integers(n, 2 * n, P * 10)})
    d2 = np.flatnonzero(dist == 2)
    counts = np.where(np.arange(len(d2)) % 2 == 0, 10, 4)       # half full, half partial
    rkh = pd.DataFrame({"query_id": np.repeat(d2, counts), "head_idx": np.repeat(pos["u_idx"].values[d2], counts),
                        "cand_idx": rng.integers(n, 2 * n, counts.sum())})
    return {"A": sp.csr_matrix((n, n)), "positives": pos, "neg_random": neg(30), "neg_hard": neg(30),
            "neg_hard_active": neg(12), "ranking": rk, "ranking_hard": rkh, "ranking_hard_active": rkh.copy()}


POS_TAG = 1000.0


def oracle(split):
    """Scores the true positive pairs above everything else."""
    pos = set(zip(split["positives"]["u_idx"], split["positives"]["v_idx"]))

    def f(_, u, v):
        return np.array([POS_TAG if (a, b) in pos else 0.0 for a, b in zip(u, v)])
    return f


def run(score_fn, split, k_full=10):
    import linkpred.eval.protocols as P
    rows, raw = [], {}
    ctx = {"model": "m", "split_name": "val", "seed": 0, "boot_seed": 0}
    s = P.PairScores(score_fn, split)
    for p in PROTOCOL_ORDER:
        fn = P.PROTOCOLS[p]
        r, x = fn(s, split, ctx, 50, k_full) if p in P.RANK_HARD_LIKE else fn(s, split, ctx, 50)
        rows += r
        raw[p] = x
    return pd.DataFrame(rows), raw


def _val(df, protocol, subset, metric):
    x = df[(df.protocol == protocol) & (df.subset == subset) & (df.metric == metric)]
    assert len(x) == 1
    return x.value.iloc[0]


def test_rows_follow_schema(split):
    df, _ = run(oracle(split), split)
    for r in df.to_dict("records"):
        validate_row({**r, "ci_low": None if pd.isna(r["ci_low"]) else r["ci_low"],
                      "ci_high": None if pd.isna(r["ci_high"]) else r["ci_high"]})
    assert set(df.protocol) == set(PROTOCOL_ORDER)


def test_hard_protocol_uses_only_distance_two_positives(split):
    df, _ = run(oracle(split), split)
    n_d2 = int((split["positives"]["distance"] == 2).sum())
    assert _val(df, "hard", "all", "n_positives") == n_d2
    assert _val(df, "diagnostic_all_vs_hard", "all", "n_positives") == len(split["positives"])
    assert _val(df, "random", "distance=3", "n_positives") == 6
    assert _val(df, "random", "disconnected", "n_positives") == 3


def test_oracle_is_perfect(split):
    df, _ = run(oracle(split), split)
    for p in ("hard", "random", "diagnostic_all_vs_hard"):
        assert _val(df, p, "all", "roc_auc") == 1.0
    assert _val(df, "rank_random", "all", "mrr") == 1.0
    assert _val(df, "rank_hard", "full_candidates", "hits@20") == 1.0


def test_constant_scorer_is_chance(split):
    df, _ = run(lambda _, u, v: np.zeros(len(u)), split)
    assert _val(df, "hard", "all", "roc_auc") == 0.5
    assert _val(df, "rank_random", "all", "mrr") == pytest.approx(np.mean(1 / np.arange(1, 12)))
    assert _val(df, "rank_hard", "all", "query_auc") == 0.5
    # partial queries have 4 candidates -> rank uniform on 1..5
    assert _val(df, "rank_hard", "partial_candidates", "mrr") == pytest.approx(np.mean(1 / np.arange(1, 6)))


def test_active_protocols(split):
    df, _ = run(oracle(split), split)
    assert _val(df, "hard_active", "all", "roc_auc") == 1.0
    assert _val(df, "hard_active", "all", "n_negatives") == 12
    assert _val(df, "rank_hard_active", "full_candidates", "mrr") == 1.0
    assert set(df[df.protocol == "hard_active"].negative_type) == {"hard_active"}


def test_rank_hard_strata_counts(split):
    df, _ = run(oracle(split), split)
    n_d2 = int((split["positives"]["distance"] == 2).sum())
    full = _val(df, "rank_hard", "full_candidates", "n_queries")
    part = _val(df, "rank_hard", "partial_candidates", "n_queries")
    assert full + part == n_d2 == _val(df, "rank_hard", "all", "n_queries")
    assert not ((df.protocol == "rank_hard") & (df.subset == "all") & (df.metric == "mrr")).any()


def test_scores_computed_once_per_pair_set(split):
    calls = []

    def f(_, u, v):
        calls.append(len(u))
        return np.zeros(len(u))
    evaluate(f, split, "val", "m", 0, 0, n_boot=0)
    assert len(calls) == 7          # positives, 3 negative sets, ranking, ranking_hard, ranking_hard_active


def test_bad_scores_rejected(split):
    with pytest.raises(ValueError):
        evaluate(lambda _, u, v: np.full(len(u), np.nan), split, "val", "m", 0, 0, n_boot=0)


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_ci_contains_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 300), rng.normal(0, 1, 300)
    a = bs.classification_ci(pos, neg, "roc_auc", np.random.default_rng(5), 300)
    b = bs.classification_ci(pos, neg, "roc_auc", np.random.default_rng(5), 300)
    from linkpred.eval.metrics import roc_auc
    assert a == b and a[0] < roc_auc(pos, neg) < a[1]


def test_paired_identical_models():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, 200)
    r = bs.paired_classification(pos, neg, pos, neg, "roc_auc", np.random.default_rng(1), 200)
    assert r["diff"] == 0 and r["p_value"] == 1.0


def test_paired_detects_clear_difference():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.5, 1, 400), rng.normal(0, 1, 400)
    noise_p, noise_n = rng.normal(0, 3, 400), rng.normal(0, 3, 400)
    r = bs.paired_classification(pos, neg, pos + noise_p, neg + noise_n, "roc_auc", np.random.default_rng(1), 500)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_value"] <= 0.01
    m = bs.paired_mean(np.ones(50), np.zeros(50), np.random.default_rng(2), 200)
    assert m["diff"] == 1 and m["p_value"] == pytest.approx(1 / 200)
