"""Heuristics against networkx / brute force on small graphs."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from linkpred.models import heuristics as H


def _split_from_graph(G: nx.Graph, n: int, weighted: bool = True) -> dict:
    rows, cols, w = [], [], []
    for a, b in G.edges():
        wt = 3.0 if weighted else 1.0          # weights must be ignored by the heuristics
        rows += [a, b]; cols += [b, a]; w += [wt, wt]
    return {"A": sp.csr_matrix((w, (rows, cols)), shape=(n, n))}


@pytest.fixture(scope="module")
def graph():
    G = nx.gnm_random_graph(60, 180, seed=3)
    G.add_node(60)                             # isolated node
    n = 61
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if not G.has_edge(a, b)]
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.choice(len(pairs), 300, replace=False)] + [(0, 60), (5, 60)]
    u, v = np.array(pairs).T
    return G, _split_from_graph(G, n), u, v


def test_common_neighbors(graph):
    G, split, u, v = graph
    exp = [len(list(nx.common_neighbors(G, a, b))) for a, b in zip(u, v)]
    assert np.allclose(H.score("common_neighbors", split, u, v), exp)


@pytest.mark.parametrize("name,fn", [
    ("jaccard", nx.jaccard_coefficient),
    ("adamic_adar", nx.adamic_adar_index),
    ("resource_allocation", nx.resource_allocation_index),
    ("preferential_attachment", nx.preferential_attachment),
])
def test_matches_networkx(graph, name, fn):
    G, split, u, v = graph
    pairs = list(zip(u.tolist(), v.tolist()))
    if name == "jaccard":   # networkx raises on 0/0 for isolated pairs; we define it as 0
        exp = [0.0 if G.degree(a) + G.degree(b) == 0 else s for a, b, s in
               ((a, b, next(iter(fn(G, [(a, b)])))[2] if G.degree(a) + G.degree(b) else 0.0) for a, b in pairs)]
    else:
        exp = [s for _, _, s in fn(G, pairs)]
    assert np.allclose(H.score(name, split, u, v), exp)


def test_katz_walk_counts_match_matrix_powers(graph):
    G, split, u, v = graph
    A = nx.to_numpy_array(G, nodelist=range(61))
    A2, A3 = A @ A, A @ A @ A
    counts = H.katz_walk_counts(split, u, v, batch=7)   # small batch exercises batching
    assert np.allclose(counts[:, 0], A[u, v])
    assert np.allclose(counts[:, 1], A2[u, v])
    assert np.allclose(counts[:, 2], A3[u, v])
    beta = 0.05
    assert np.allclose(H.katz_from_counts(counts, beta), beta * A[u, v] + beta**2 * A2[u, v] + beta**3 * A3[u, v])


def test_katz_scorer_caches_and_respects_beta(graph):
    _, split, u, v = graph
    k = H.KatzScorer(0.01)
    a = k(split, u, v)
    k.beta = 0.1
    b = k(split, u, v)
    assert len(k._cache) == 1 and not np.allclose(a, b)


def test_isolated_node_scores_zero(graph):
    _, split, _, _ = graph
    u, v = np.array([0]), np.array([60])
    for name in H.HEURISTICS:
        assert H.score(name, split, u, v)[0] == 0.0


def test_context_cache_never_stale():
    """Audit m8: the cache is checked by object identity, so a different adjacency is never served stale."""
    A1 = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float))
    A2 = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    s1, s2 = {"A": A1}, {"A": A2}
    assert H.context(s1).deg.tolist() == [2, 1, 1]
    assert H.context(s2).deg.tolist() == [1, 2, 1]
    assert H.context(s1).deg.tolist() == [2, 1, 1]


def test_prefer_newer():
    import pandas as pd
    split = {"nodes": pd.DataFrame({"n_works_before": [10, 1, 5]})}
    assert H.prefer_newer(split, np.array([0, 0]), np.array([1, 2])).tolist() == [-1.0, -5.0]
