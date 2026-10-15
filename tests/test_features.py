"""Pair features: leakage (future-perturbation invariance), hand-computed values, alignment."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from linkpred.data.build import build
from linkpred.data.splits import build_input_graph, build_splits, load_split, load_tables
from linkpred.features.build import canonical, features_for, orient_by_frequency
from linkpred.features.pair_features import ALL_FEATURES, NO_SHARED, FeatureContext, load_feature_tables
from linkpred.models.feature_models import training_pairs, training_set

from test_leakage import _synthetic_cfg

PARAMS = {"seed": 0, "katz_beta": 0.005, "half_life": 2.0}


# ---------------------------------------------------------------- leakage: future perturbation

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("feat")
    cfg = _synthetic_cfg(tmp)
    cfg["negatives"] = {**cfg["negatives"], "train_pool_ratio": 2}
    build(cfg)
    build_splits(cfg)
    proc = cfg["paths"]["processed"]
    ft, st = load_feature_tables(proc), load_tables(proc)
    rng = np.random.default_rng(0)
    emb_ids = pd.DataFrame({"work_id": ft["works"]["work_id"], "has_text": True, "has_abstract": True})
    emb = rng.normal(size=(len(emb_ids), 8)).astype(np.float32)
    return {"cfg": cfg, "ft": ft, "st": st, "emb": emb, "emb_ids": emb_ids}


def _features(ft, st, cutoff, emb, emb_ids, u, v):
    nodes, g, A = build_input_graph(st, cutoff)
    ctx = FeatureContext(ft, cutoff, nodes, g, A, emb, emb_ids, PARAMS)
    return nodes, ctx.compute(u, v)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_features_ignore_works_in_or_after_cutoff(synth, split):
    cfg = synth["cfg"]
    cutoff = cfg["splits"][split]["cutoff"]
    sp_ = load_split(cfg["paths"]["splits"], split)
    pairs = pd.concat([sp_["positives"][["u_idx", "v_idx"]], sp_["neg_hard"][["u_idx", "v_idx"]],
                       sp_["neg_random"][["u_idx", "v_idx"]]]).drop_duplicates()
    u, v = pairs["u_idx"].values, pairs["v_idx"].values
    nodes_full, full = _features(synth["ft"], synth["st"], cutoff, synth["emb"], synth["emb_ids"], u, v)

    # Delete every work in or after the cutoff and scramble the embeddings of future works.
    past = {k: (df[df["year"] < cutoff] if "year" in df.columns else df) for k, df in synth["ft"].items()}
    st_past = {k: (df[df["year"] < cutoff] if "year" in df.columns else df) for k, df in synth["st"].items()}
    future = synth["ft"]["works"]["year"].values >= cutoff
    emb2 = synth["emb"].copy()
    emb2[future] = np.random.default_rng(1).normal(size=(future.sum(), emb2.shape[1]))
    nodes_past, past_f = _features(past, st_past, cutoff, emb2, synth["emb_ids"], u, v)

    assert nodes_full["author_id"].equals(nodes_past["author_id"])
    pd.testing.assert_frame_equal(full, past_f)
    assert list(full.columns) == ALL_FEATURES


def test_features_symmetric(synth):
    cfg = synth["cfg"]
    sp_ = load_split(cfg["paths"]["splits"], "val")
    u, v = sp_["neg_random"]["u_idx"].values, sp_["neg_random"]["v_idx"].values
    _, a = _features(synth["ft"], synth["st"], cfg["splits"]["val"]["cutoff"], synth["emb"], synth["emb_ids"], u, v)
    _, b = _features(synth["ft"], synth["st"], cfg["splits"]["val"]["cutoff"], synth["emb"], synth["emb_ids"], v, u)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- hand-computed values

@pytest.fixture
def tiny():
    """Path 0-1-2-3, plus 1-4 and 2-4; node 5 isolated. Cutoff 2023.

    last collaboration years: 0-1: 2022, 1-2: 2020, 2-3: 2021, 1-4: 2022, 2-4: 2018.
    UTS: 0, 1, 2, 3 ; non-UTS: 4, 5.
    """
    authors = [f"A{i}" for i in range(6)]
    edges = [(0, 1, 2022), (1, 2, 2020), (2, 3, 2021), (1, 4, 2022), (2, 4, 2018)]
    nodes = pd.DataFrame({"node_idx": range(6), "author_id": authors,
                          "is_uts": [True, True, True, True, False, False], "n_works_before": 1, "degree": 0})
    g = pd.DataFrame({"u_idx": [a for a, _, _ in edges], "v_idx": [b for _, b, _ in edges],
                      "author_u": [authors[a] for a, _, _ in edges], "author_v": [authors[b] for _, b, _ in edges],
                      "weight": 1, "first_year": [y for *_, y in edges], "last_year": [y for *_, y in edges]})
    n = 6
    A = sp.csr_matrix((np.ones(2 * len(edges)), (np.r_[g.u_idx, g.v_idx], np.r_[g.v_idx, g.u_idx])), shape=(n, n))
    # Works: one per edge in its last year, plus a 2024 work (future) for A0 and A3, plus a solo A5 work.
    rows = []
    for k, (a, b, y) in enumerate(edges):
        for x in (a, b):
            rows.append((f"W{k}", authors[x], x < 4, ["I1"] + (["I9"] if x in (0, 3) else []), y))
    rows += [("W9", "A0", True, ["I1"], 2024), ("W9", "A3", True, ["I1"], 2024), ("W8", "A5", False, ["I2"], 2016)]
    auth = pd.DataFrame(rows, columns=["work_id", "author_id", "is_uts", "institution_ids", "year"])
    works = auth[["work_id", "year"]].drop_duplicates()
    emb_ids = pd.DataFrame({"work_id": works["work_id"].values, "has_text": True, "has_abstract": False})
    emb = np.eye(len(works), 4, dtype=np.float32)
    emb[works["work_id"].values == "W9"] = 1.0   # future work: must not matter
    ctx = FeatureContext({"works": works, "auth": auth}, 2023, nodes, g, A, emb, emb_ids, PARAMS)
    return ctx


def test_hand_computed(tiny):
    f = tiny.compute(np.array([0, 0, 1, 0]), np.array([2, 3, 3, 5]))
    # (0,2): common neighbour 1 only; (0,3): distance 3; (1,3): common neighbour 2; (0,5): disconnected
    assert f["common_neighbors"].tolist() == [1, 0, 1, 0]
    assert f["distance"].tolist() == [2, 3, 2, 5]
    assert f["disconnected"].tolist() == [0, 0, 0, 1]
    assert f["cn_uts"].tolist() == [1, 0, 1, 0] and f["cn_non_uts"].tolist() == [0, 0, 0, 0]
    # recency weights, h = 2, c = 2023: w(0,1) = .5**(1/2), w(2,1) = .5**(3/2)
    assert f["rw_cn"][0] == pytest.approx(0.5 ** 0.5 * 0.5 ** 1.5)
    # most recent shared collaboration for (0,2) via 1: min(2022, 2020) = 2020 -> 3 years
    assert f["years_since_shared"].tolist()[:2] == [3, NO_SHARED]
    # A0 and A3 share I9 and I1 (A0's 2024 work is ignored)
    assert f["shared_institutions"][1] == 2
    # A0: first work 2022 -> career age 1; A3: 2021 -> 2
    assert (f["career_age_min"][1], f["career_age_max"][1]) == (1, 2)


def test_cn_split_by_uts(tiny):
    f = tiny.compute(np.array([1]), np.array([2]))     # connected pair; common neighbour 4 is non-UTS
    assert f["cn_uts"][0] == 0 and f["cn_non_uts"][0] == 1


# ---------------------------------------------------------------- alignment and training sets

def test_features_for_alignment():
    feat = pd.DataFrame({"u_idx": [1, 2, 3], "v_idx": [5, 6, 7], "x": [10.0, 20.0, 30.0]})
    out = features_for(feat, np.array([6, 1, 7]), np.array([2, 5, 3]), ["x"])
    assert out["x"].tolist() == [20.0, 10.0, 30.0]
    with pytest.raises(KeyError):
        features_for(feat, np.array([1]), np.array([2]), ["x"])


def test_orient_and_canonical():
    u, v = np.array([5, 1, 1, 1]), np.array([1, 7, 8, 9])
    s, t = orient_by_frequency(u, v)
    assert (s == 1).all() and sorted(t.tolist()) == [5, 7, 8, 9]
    cu, cv = canonical(v, u)
    assert (cu <= cv).all()


def test_training_set_mix_counts(synth):
    cfg = synth["cfg"]
    train = load_split(cfg["paths"]["splits"], "train")
    allp = pd.concat([train[k][["u_idx", "v_idx"]] for k in ("positives", "train_pool_random", "train_pool_hard")])
    feat = allp.drop_duplicates().assign(x=1.0)
    P = len(train["positives"])
    X, y = training_set(train, feat, ["x"], 0.5, 2)
    assert y.sum() == P and (y == 0).sum() == 2 * P
    Xs, ys = training_set(train, feat, ["x"], 0.0, 1, specialist=True)
    assert ys.sum() == (train["positives"]["distance"] == 2).sum()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_gnn_node_features_ignore_works_in_or_after_cutoff(synth, split):
    """GNN node features (degree, career age, recent works, UTS flag, text vector) are built by the
    same FeatureContext and are unchanged when every work in/after the cutoff is deleted."""
    from linkpred.features.node_features import node_feature_arrays

    cutoff = synth["cfg"]["splits"][split]["cutoff"]

    def arrays(ft, st, emb):
        nodes, g, A = build_input_graph(st, cutoff)
        ctx = FeatureContext(ft, cutoff, nodes, g, A, emb, synth["emb_ids"], PARAMS, communities=False)
        return node_feature_arrays(ctx)

    s1, t1 = arrays(synth["ft"], synth["st"], synth["emb"])
    past = {k: (df[df["year"] < cutoff] if "year" in df.columns else df) for k, df in synth["ft"].items()}
    st_past = {k: (df[df["year"] < cutoff] if "year" in df.columns else df) for k, df in synth["st"].items()}
    emb2 = synth["emb"].copy()
    future = synth["ft"]["works"]["year"].values >= cutoff
    emb2[future] = np.random.default_rng(7).normal(size=(future.sum(), emb2.shape[1]))
    s2, t2 = arrays(past, st_past, emb2)
    assert np.array_equal(s1, s2) and np.array_equal(t1, t2)


def test_d2_only_training_pairs(synth):
    """Audit M1: the pair-focused training set uses distance-2 positives only, so distance cannot separate classes."""
    cfg = synth["cfg"]
    train = load_split(cfg["paths"]["splits"], "train")
    pairs, y = training_pairs(train, 0.0, 1, active=True, d2_only=True)
    pos = train["positives"]
    d2 = set(zip(pos.u_idx[pos.distance == 2], pos.v_idx[pos.distance == 2]))
    got = set(zip(pairs.u_idx[y == 1], pairs.v_idx[y == 1]))
    assert got == d2 and (y == 0).sum() == len(d2)
