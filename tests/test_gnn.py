"""GNN model, decoder, training loop and the supervision-pair leakage assertion.

Runs on CPU on a fresh clone (synthetic graphs), and checks the real train split when present.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from linkpred.models import gnn as G


def _graph(n=60, m=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    r, c = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = r != c
    A = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    x = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    return A, G.GraphData(x, G.edge_index_from_adjacency(A))


@pytest.mark.parametrize("encoder,layers", [("sage", 2), ("sage", 3), ("gat", 2)])
@pytest.mark.parametrize("decoder", ["mlp", "dot"])
def test_forward_shapes_and_symmetry(encoder, layers, decoder):
    _, g = _graph()
    torch.manual_seed(0)
    m = G.LinkPredictor(encoder, decoder, g.x.shape[1], 16, layers, 0.3, heads=4)
    u, v = np.array([0, 1, 2]), np.array([5, 6, 7])
    s_uv, s_vu = m.score(g, u, v), m.score(g, v, u)
    assert s_uv.shape == (3,) and np.isfinite(s_uv).all()
    assert np.allclose(s_uv, s_vu, atol=1e-6)          # Hadamard / dot decoders are symmetric


def test_dot_decoder_is_inner_product():
    dec = G.Decoder("dot", 4, 0.0)
    a, b = torch.tensor([[1.0, 2.0, 0.0, -1.0]]), torch.tensor([[3.0, 1.0, 5.0, 2.0]])
    assert float(dec(a, b)) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        G.Decoder("bilinear", 4, 0.0)


def test_encoder_rejects_unknown_kind():
    with pytest.raises(ValueError):
        G.Encoder("gcn", 4, 8, 2, 0.0)


def test_eval_mode_is_deterministic():
    _, g = _graph()
    m = G.LinkPredictor("sage", "mlp", g.x.shape[1], 16, 2, 0.5)
    u, v = np.arange(10), np.arange(10, 20)
    assert np.array_equal(m.score(g, u, v), m.score(g, u, v))   # dropout off in score()


def test_same_seed_same_training_on_cpu():
    A, g = _graph()
    u, v = np.array([0, 1, 2, 3, 4, 5]), np.array([30, 31, 32, 33, 34, 35])
    keep = np.asarray(A[u, v]).ravel() == 0
    u, v = u[keep], v[keep]
    y = (np.arange(len(u)) % 2).astype(float)

    def run(seed):
        G.set_determinism(seed)
        m = G.LinkPredictor("sage", "mlp", g.x.shape[1], 8, 2, 0.3)
        r = G.train(m, g, u, v, y, g, (u, v), y, metric="roc_auc", lr=0.01, weight_decay=0.0,
                    epochs=5, patience=5, device="cpu")
        return r.model.score(g, u, v), r.curve
    (a, ca), (b, cb) = run(1), run(1)
    assert np.array_equal(a, b) and [c["train_loss"] for c in ca] == [c["train_loss"] for c in cb]


def test_training_learns_a_separable_signal():
    """Positives = pairs whose features match; the model must beat chance on held-out pairs."""
    rng = np.random.default_rng(0)
    n = 200
    A, _ = _graph(n=n, m=400)
    group = rng.integers(0, 2, n)
    x = torch.from_numpy(np.c_[group, 1 - group, rng.normal(size=(n, 2))].astype(np.float32))
    g = G.GraphData(x, G.edge_index_from_adjacency(A))
    u, v = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    keep = (u != v) & (np.asarray(A[u, v]).ravel() == 0)
    u, v = u[keep], v[keep]
    y = (group[u] == group[v]).astype(float)
    tr, va = slice(0, len(u) // 2), slice(len(u) // 2, None)
    G.set_determinism(0)
    m = G.LinkPredictor("sage", "mlp", 4, 16, 2, 0.0)
    r = G.train(m, g, u[tr], v[tr], y[tr], g, (u[va], v[va]), y[va], metric="roc_auc", lr=0.01,
                weight_decay=0.0, epochs=60, patience=60, device="cpu")
    assert r.best_val > 0.9


def test_nonfinite_loss_raises():
    _, g = _graph()
    g.x[0, 0] = float("nan")
    m = G.LinkPredictor("sage", "mlp", g.x.shape[1], 8, 2, 0.0)
    u, v, y = np.array([0, 1]), np.array([2, 3]), np.array([1.0, 0.0])
    with pytest.raises(FloatingPointError):
        G.train(m, g, u, v, y, g, (u, v), y, metric="roc_auc", lr=0.01, weight_decay=0.0, epochs=2, patience=2,
                device="cpu")


# ---------------------------------------------------------------- leakage

def test_supervision_assertion_catches_graph_edges():
    A, _ = _graph()
    r, c = A.nonzero()
    with pytest.raises(AssertionError):
        G.assert_supervision_not_in_graph(A, r[:3], c[:3])
    nonedge = [(a, b) for a in range(60) for b in range(a + 1, 60) if A[a, b] == 0][:5]
    u, v = np.array(nonedge).T
    G.assert_supervision_not_in_graph(A, u, v)        # does not raise


def test_node_features_standardised_with_given_stats():
    s = np.array([[1.0, 10.0], [3.0, 10.0]])
    out = G.make_node_features(s, None, mean=np.array([2.0, 10.0]), std=np.array([1.0, 0.0]))
    assert out.tolist() == [[-1.0, 0.0], [1.0, 0.0]]          # zero std -> centred, not divided
    t = np.ones((2, 3), dtype=np.float32)
    assert G.make_node_features(s, t, np.zeros(2), np.ones(2)).shape == (2, 5)


SPLITS_DIR = Path("data/processed/splits/train")


@pytest.mark.skipif(not (SPLITS_DIR / "train_pool_hard_active.parquet").exists(), reason="splits not built")
@pytest.mark.parametrize("objective", ["general", "pair_focused"])
def test_real_supervision_pairs_are_not_input_edges(objective):
    """Train-split supervision pairs (positives + both training-negative mixes) are never
    message-passing edges of the train input graph."""
    from linkpred.config import load_config
    from linkpred.data.splits import load_split
    from linkpred.models.feature_models import training_pairs

    cfg = load_config()
    tr = load_split("data/processed/splits", "train")
    g = cfg["models"]["gnn"]
    if objective == "general":
        pairs, y = training_pairs(tr, g["general_mix"][0], g["general_mix"][1])
    else:
        pairs, y = training_pairs(tr, 0.0, g["pair_focused_ratio"], active=True)
    assert y.sum() == len(tr["positives"])
    G.assert_supervision_not_in_graph(tr["A"], pairs["u_idx"].values, pairs["v_idx"].values)


@pytest.mark.skipif(not Path("data/processed/node_features/train_structural.npy").exists(),
                    reason="node features not built")
def test_node_features_match_split_nodes():
    from linkpred.data.splits import load_split
    from linkpred.features.node_features import load_node_features

    for s in ("train", "val", "test"):
        nodes = load_split("data/processed/splits", s)["nodes"]
        st, tx = load_node_features("data/processed", s, "structural"), load_node_features("data/processed", s, "text")
        assert st.shape == (len(nodes), 4) and tx.shape[0] == len(nodes)
        assert np.allclose(st[:, 0], np.log1p(nodes["degree"].values), atol=1e-5)   # log degree = input-graph degree
        assert np.array_equal(st[:, 3].astype(bool), nodes["is_uts"].values)        # pre-cutoff UTS flag
