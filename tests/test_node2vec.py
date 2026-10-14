"""node2vec helpers: operators, alignment, determinism; embeddings depend only on the adjacency."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import ortho_group

from linkpred.models import node2vec as N2V

CFG = dict(embedding_dim=8, walk_length=5, context_size=3, walks_per_node=2, num_negative_samples=1,
           epochs=1, lr=0.05, batch_size=16)


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(0)
    n = 40
    r, c = rng.integers(0, n, 120), rng.integers(0, n, 120)
    keep = r != c
    A = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    return A


def test_operators():
    E = np.array([[1.0, 2.0], [3.0, -1.0]])
    u, v = np.array([0]), np.array([1])
    assert N2V.edge_features(E, u, v, "hadamard").tolist() == [[3.0, -2.0]]
    assert N2V.edge_features(E, u, v, "average").tolist() == [[2.0, 0.5]]
    assert N2V.edge_features(E, u, v, "l1").tolist() == [[2.0, 3.0]]
    assert N2V.edge_features(E, u, v, "l2").tolist() == [[4.0, 9.0]]
    assert N2V.cosine(E, u, v)[0] == pytest.approx(1 / (np.sqrt(5) * np.sqrt(10)))
    with pytest.raises(ValueError):
        N2V.edge_features(E, u, v, "max")


def test_alignment_recovers_rotation():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(30, 6))
    R = ortho_group.rvs(6, random_state=2)
    nodes_ref = pd.DataFrame({"author_id": [f"A{i}" for i in range(30)], "node_idx": range(30), "degree": 1})
    # The later graph has the same authors in a different order plus two new ones.
    perm = np.r_[rng.permutation(30), 30, 31]
    emb = np.vstack([ref, rng.normal(size=(2, 6))])[perm] @ R.T
    nodes = pd.DataFrame({"author_id": [f"A{i}" for i in perm], "node_idx": range(32), "degree": 1})
    aligned = N2V.align_to(emb, nodes, ref, nodes_ref)
    back = pd.Series(range(32), index=perm)
    assert np.allclose(aligned[back.loc[range(30)].values], ref, atol=1e-8)


@pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])   # biased walks use C rand()
def test_same_seed_same_embeddings(graph, p, q):
    a = N2V.train_embeddings(graph, p, q, 3, CFG)
    b = N2V.train_embeddings(graph, p, q, 3, CFG)
    c = N2V.train_embeddings(graph, p, q, 4, CFG)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (graph.shape[0], 8)


def test_embeddings_depend_only_on_adjacency(graph):
    """train_embeddings receives nothing but the adjacency; equal adjacency -> equal embeddings,
    so the input-graph invariance in tests/test_leakage.py carries over to node2vec."""
    import inspect
    params = list(inspect.signature(N2V.train_embeddings).parameters)
    assert params[0] == "A" and "split" not in params and "tables" not in params
    a = N2V.train_embeddings(graph, 0.5, 2.0, 0, CFG)
    b = N2V.train_embeddings(graph.copy(), 0.5, 2.0, 0, CFG)
    assert np.array_equal(a, b)
