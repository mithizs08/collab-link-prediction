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


# ---------------------------------------------------------------- group-matched (within-group) evaluation (audit M2)

@pytest.fixture
def gsplit():
    """Nodes 0-19 established (10 works), 20-39 newcomers (1 work). Positives / negatives mixed."""
    rng = np.random.default_rng(3)
    n = 40
    nodes = pd.DataFrame({"node_idx": np.arange(n), "n_works_before": np.r_[np.full(20, 10), np.full(20, 1)]})
    P = 30
    pu, pv = rng.integers(0, n, P), rng.integers(0, n, P)
    pos = pd.DataFrame({"u_idx": pu, "v_idx": pv, "distance": 2.0, "cold_start": False})
    neg = pd.DataFrame({"u_idx": rng.integers(0, n, 40), "v_idx": rng.integers(0, n, 40)})
    rk = pd.DataFrame({"query_id": np.repeat(np.arange(P), 8), "head_idx": np.repeat(pu, 8),
                       "cand_idx": rng.integers(0, n, P * 8)})
    return {"A": sp.csr_matrix((n, n)), "nodes": nodes, "stats": {"cold_start_max_works": 2},
            "positives": pos, "neg_random": neg, "neg_hard": neg, "neg_hard_active": neg,
            "ranking": rk, "ranking_hard": rk, "ranking_hard_active": rk}


def test_group_matched_rows_and_macro(gsplit):
    from linkpred.eval.protocols import newcomer_pairs
    rng = np.random.default_rng(0)
    score = lambda _, u, v: rng.normal(size=len(u))
    rows, raw = evaluate(score, gsplit, "val", "m", 0, 0, n_boot=30, protocols=("hard_active", "rank_hard_active"))
    df = pd.DataFrame(rows)
    pos, neg = gsplit["positives"], gsplit["neg_hard_active"]
    pn = newcomer_pairs(gsplit, pos.u_idx.values, pos.v_idx.values)
    nn = newcomer_pairs(gsplit, neg.u_idx.values, neg.v_idx.values)
    g = lambda sub, m: df[(df.protocol == "hard_active") & (df.subset == sub) & (df.metric == m)].value.iloc[0]
    assert g("established_matched", "n_positives") == (~pn).sum() and g("established_matched", "n_negatives") == (~nn).sum()
    assert g("newcomer_matched", "n_negatives") == nn.sum()
    assert g("macro", "roc_auc") == pytest.approx((g("established_matched", "roc_auc") + g("newcomer_matched", "roc_auc")) / 2)
    # ranking: every kept candidate has the query's group
    a = raw["rank_hard_active"]
    assert len(a["established_matched"]["query_auc"]) + len(a["newcomer_matched"]["query_auc"]) <= len(pos)


def test_prefer_newer_is_chance_within_groups(gsplit):
    """With a constant number of prior works inside a group (this fixture), a scorer that only knows
    seniority separates nothing once both classes come from that group. (On real data prior works vary
    within groups, so prefer_newer is not 0.5 there.)"""
    from linkpred.models.heuristics import prefer_newer
    rows, _ = evaluate(prefer_newer, gsplit, "val", "pn", 0, 0, n_boot=0, protocols=("hard_active",))
    df = pd.DataFrame(rows)
    est = df[(df.subset == "established_matched") & (df.metric == "roc_auc")].value.iloc[0]
    assert est == 0.5      # all established pairs score -10: all tied


def test_holm_and_seed_mean_ci():
    from linkpred.eval import bootstrap as bs
    adj = bs.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert bs.holm({"a": 0.6, "b": 0.5})["a"] == 1.0
    rng = np.random.default_rng(0)
    raw = {"pos": rng.normal(1, 1, 200), "neg": rng.normal(0, 1, 200)}
    one = bs.seed_mean_ci([raw], "roc_auc", np.random.default_rng(1), 200)
    ref = bs.raw_ci(raw, "roc_auc", np.random.default_rng(1), 200)
    assert one == ref                                    # a single seed reduces to the ordinary CI
    macro = {"groups": [raw, {"pos": rng.normal(2, 1, 50), "neg": rng.normal(0, 1, 80)}]}
    r = bs.paired_raw(macro, macro, "roc_auc", np.random.default_rng(2), 100)
    assert r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
