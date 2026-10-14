"""Metrics on tiny hand-computed examples, including ties."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from linkpred.eval.metrics import (average_precision, expected_hits, expected_reciprocal_rank,
                                   per_query_metrics, query_auc, rank_counts, roc_auc, snap)


# ---------------------------------------------------------------- AUC

def test_auc_perfect_and_reversed():
    assert roc_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert roc_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


def test_auc_hand_computed_with_ties():
    # pos = [2, 1], neg = [1, 0]. Pairs: (2>1)=1, (2>0)=1, (1=1)=0.5, (1>0)=1 -> 3.5 / 4
    assert roc_auc(np.array([2.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.875)


def test_auc_all_tied_is_half():
    assert roc_auc(np.zeros(5), np.zeros(7)) == 0.5


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 50).astype(float), rng.integers(0, 4, 80).astype(float)
    y = np.r_[np.ones(50), np.zeros(80)]
    assert roc_auc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))


def test_float_noise_counts_as_tie():
    a = 0.1 + 0.2          # 0.30000000000000004
    assert roc_auc(np.array([a]), np.array([0.3])) == 0.5
    assert snap(np.array([a]))[0] == snap(np.array([0.3]))[0]


# ---------------------------------------------------------------- AP

def test_ap_hand_computed():
    # Ranking: pos(0.9), neg(0.8), pos(0.7): precision at the positives = 1, 2/3 -> AP = 5/6
    assert average_precision(np.array([0.9, 0.7]), np.array([0.8])) == pytest.approx(5 / 6)


def test_ap_ties_form_one_threshold():
    # All tied: one threshold, precision = base rate = 2/5, recall jumps 0 -> 1.
    assert average_precision(np.zeros(2), np.zeros(3)) == pytest.approx(0.4)


# ---------------------------------------------------------------- ranking

def test_rank_counts():
    pos = np.array([5.0, 0.0])
    cand = np.array([6.0, 5.0, 1.0, 0.0, 0.0, 0.0])
    q = np.array([0, 0, 0, 1, 1, 1])
    g, e, n = rank_counts(pos, cand, q, 2)
    assert g.tolist() == [1, 0] and e.tolist() == [1, 3] and n.tolist() == [3, 3]


def test_expected_rr_and_hits_hand_computed():
    # g=1, e=1: rank uniform on {2, 3}. E[1/r] = (1/2 + 1/3) / 2; P(r<=2) = 1/2
    assert expected_reciprocal_rank(np.array([1]), np.array([1]))[0] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert expected_hits(np.array([1]), np.array([1]), 2)[0] == pytest.approx(0.5)
    # No ties reduces to the usual definitions.
    assert expected_reciprocal_rank(np.array([0]), np.array([0]))[0] == 1.0
    assert expected_hits(np.array([4]), np.array([0]), 5)[0] == 1.0
    assert expected_hits(np.array([5]), np.array([0]), 5)[0] == 0.0


def test_all_tied_query_equals_random_ranker():
    # Positive tied with all 100 candidates: rank uniform on 1..101.
    g, e = np.array([0]), np.array([100])
    assert expected_reciprocal_rank(g, e)[0] == pytest.approx(np.mean(1 / np.arange(1, 102)))
    assert expected_hits(g, e, 20)[0] == pytest.approx(20 / 101)
    assert query_auc(g, e, np.array([100]))[0] == 0.5


def test_expected_metrics_match_random_tiebreak_simulation():
    rng = np.random.default_rng(1)
    pos = rng.integers(0, 3, 30).astype(float)
    cand = rng.integers(0, 3, 30 * 10).astype(float)
    q = np.repeat(np.arange(30), 10)
    g, e, n = rank_counts(pos, cand, q, 30)
    m = per_query_metrics(g, e, n, ks=(3,))
    sims_rr, sims_h = [], []
    for _ in range(4000):
        jit_p, jit_c = rng.random(30), rng.random(300)          # random tie-breaking
        better = ((cand > pos[q]) | ((cand == pos[q]) & (jit_c > jit_p[q])))
        rank = 1 + np.bincount(q, weights=better, minlength=30)
        sims_rr.append((1 / rank).mean())
        sims_h.append((rank <= 3).mean())
    assert m["mrr"].mean() == pytest.approx(np.mean(sims_rr), abs=0.005)
    assert m["hits@3"].mean() == pytest.approx(np.mean(sims_h), abs=0.005)
