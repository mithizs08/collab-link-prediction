"""Leakage tests for the temporal splits.

Every check recomputes what it needs (UTS status, window collaborations, the input
graph) directly from the Phase 1 tables instead of trusting columns written by
splits.py. Checks run on a synthetic OpenAlex-like snapshot pushed through the real
build() + build_splits() (always), and on data/processed/splits when present.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linkpred.config import load_config
from linkpred.data.build import build
from linkpred.data.splits import SPLITS, all_uts_coauthor_pairs, build_splits, load_split, load_tables, make_split

UTS, EXT = "I114017466", "I999"
OA = "https://openalex.org/"


# ---------------------------------------------------------------- synthetic fixture

def _synthetic_works(seed: int = 0) -> list[dict]:
    """~440 works, 2015-2025, four communities of UTS authors plus external authors.

    A070-A079 are UTS authors who only appear from 2024: they must not be UTS
    (or even nodes) in the train split. A080-A089 publish with UTS only as externals
    until 2022, then with a UTS affiliation.
    """
    rnd = random.Random(seed)
    uts_core = [f"A{i:03d}" for i in range(70)]
    late_uts = [f"A{i:03d}" for i in range(70, 80)]
    switchers = [f"A{i:03d}" for i in range(80, 90)]
    external = [f"A{i:03d}" for i in range(100, 160)]
    comm = {a: i % 4 for i, a in enumerate(uts_core + late_uts + switchers + external)}

    def auth(a, year):
        is_uts = a in uts_core or a in late_uts or (a in switchers and year >= 2023)
        inst = UTS if is_uts else EXT
        return {"author_position": "middle", "author": {"id": OA + a, "display_name": f"Person {a}"},
                "institutions": [{"id": OA + inst, "lineage": [OA + inst]}]}

    works, wid = [], 0
    for year in range(2015, 2026):
        pool = uts_core + switchers + external + (late_uts if year >= 2024 else [])
        for _ in range(40):
            c = rnd.randrange(4)
            members = [a for a in pool if comm[a] == c]
            k = rnd.randint(2, 5)
            team = rnd.sample(members, k - 1) + [rnd.choice(pool)]   # one possibly cross-community
            if not any(a in uts_core or a in late_uts or (a in switchers and year >= 2023) for a in team):
                team.append(rnd.choice(uts_core))
            wid += 1
            works.append({"id": f"{OA}W{wid}", "publication_year": year, "type": "article",
                          "title": f"Synthetic study {wid} on collaboration graphs",
                          "authorships": [auth(a, year) for a in dict.fromkeys(team)],
                          "abstract_inverted_index": None, "primary_location": {}})
    # Hyper-authored works: no edges, but they count as joint works.
    for year in (2016, 2023):
        wid += 1
        team = rnd.sample(uts_core, 20) + rnd.sample(external, 10)
        works.append({"id": f"{OA}W{wid}", "publication_year": year, "type": "article",
                      "title": f"Large consortium paper {wid} with many authors",
                      "authorships": [auth(a, year) for a in team],
                      "abstract_inverted_index": None, "primary_location": {}})
    return works


def _synthetic_cfg(tmp: Path) -> dict:
    snap = tmp / "snap.json"
    snap.write_text(json.dumps(_synthetic_works()))
    cfg = load_config()
    cfg["data"] = {**cfg["data"], "snapshot_path": str(snap)}
    cfg["paths"] = {**cfg["paths"], "processed": str(tmp / "processed"), "report": str(tmp / "report"),
                    "splits": str(tmp / "processed" / "splits")}
    cfg["negatives"] = {**cfg["negatives"], "ranking_candidates": 5}
    return cfg


def _bundle(cfg: dict) -> dict:
    processed = Path(cfg["paths"]["processed"])
    root = Path(cfg["paths"]["splits"])
    return {"cfg": cfg, "tables": load_tables(processed),
            "splits": {name: load_split(root, name) for name in SPLITS}}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("leak")
    cfg = _synthetic_cfg(tmp)
    build(cfg)
    build_splits(cfg)
    return _bundle(cfg)


@pytest.fixture(scope="module")
def real():
    cfg = load_config()
    if not (Path(cfg["paths"]["splits"]) / "test" / "meta.json").exists():
        pytest.skip("splits not built (run `make data splits`)")
    return _bundle(cfg)


@pytest.fixture(params=["synthetic", "real"])
def bundle(request):
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------- independent recomputation

def _cut(b, name):
    s = b["cfg"]["splits"][name]
    return s["cutoff"], s["window_end"]


def _uts_before(t, cutoff) -> set:
    a = t["auth"]
    return set(a.loc[(a["year"] < cutoff) & a["is_uts"], "author_id"])


def _authors_before(t, cutoff) -> set:
    a = t["auth"]
    return set(a.loc[a["year"] < cutoff, "author_id"])


def _window_collab(t, cutoff, window_end) -> set:
    """Pairs with ANY joint work (any size) in the window."""
    a = t["auth"][t["auth"]["year"].between(cutoff, window_end)][["work_id", "author_id"]].drop_duplicates()
    m = a.merge(a, on="work_id")
    m = m[m["author_id_x"] < m["author_id_y"]]
    return set(zip(m["author_id_x"], m["author_id_y"]))


def _pairs(df, u="author_u", v="author_v") -> set:
    return set(zip(df[u], df[v]))


def _graph_pairs(sp_) -> set:
    return _pairs(sp_["graph_edges"])


def _ordered(rk: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"author_u": rk[["head", "cand"]].min(axis=1), "author_v": rk[["head", "cand"]].max(axis=1)})


def _all_negatives(sp_) -> pd.DataFrame:
    pools = [sp_[k][["author_u", "author_v"]] for k in ("train_pool_random", "train_pool_hard") if k in sp_]
    return pd.concat([sp_["neg_random"][["author_u", "author_v"]], sp_["neg_hard"][["author_u", "author_v"]],
                      _ordered(sp_["ranking"]), _ordered(sp_["ranking_hard"]), *pools])


def _neighbours(sp_) -> dict:
    g = sp_["graph_edges"]
    nbrs: dict[str, set] = {}
    for u, v in zip(g["author_u"], g["author_v"]):
        nbrs.setdefault(u, set()).add(v)
        nbrs.setdefault(v, set()).add(u)
    return nbrs


# ---------------------------------------------------------------- the required leakage checks

@pytest.mark.parametrize("name", SPLITS)
def test_input_graph_uses_only_pre_cutoff_works(bundle, name):
    t, sp_ = bundle["tables"], bundle["splits"][name]
    cutoff, _ = _cut(bundle, name)
    g = sp_["graph_edges"]
    assert (g["last_year"] < cutoff).all()
    # Exact equality with a recomputation from pre-cutoff edges: any future edge or
    # any future joint work inflating a weight would break this.
    e = t["edges"][t["edges"]["year"] < cutoff]
    expected = e.groupby(["author_u", "author_v"])["work_id"].nunique()
    got = g.set_index(["author_u", "author_v"])["weight"]
    assert len(got) == len(expected)
    assert got.sort_index().equals(expected.sort_index().rename("weight"))
    # Nodes are exactly the authors with a pre-cutoff work; node attributes are pre-cutoff.
    nodes = sp_["nodes"]
    assert set(nodes["author_id"]) == _authors_before(t, cutoff)
    assert set(nodes.loc[nodes["is_uts"], "author_id"]) == _uts_before(t, cutoff)


@pytest.mark.parametrize("name", SPLITS)
def test_positives_not_connected_in_input_graph(bundle, name):
    sp_ = bundle["splits"][name]
    assert not (_pairs(sp_["positives"]) & _graph_pairs(sp_))


@pytest.mark.parametrize("name", SPLITS)
def test_positives_are_first_coauthorships_in_window(bundle, name):
    t, sp_ = bundle["tables"], bundle["splits"][name]
    cutoff, window_end = _cut(bundle, name)
    first = all_uts_coauthor_pairs(t["auth"]).groupby(["author_u", "author_v"])["year"].min()
    yrs = first.loc[list(_pairs(sp_["positives"]))]
    assert yrs.between(cutoff, window_end).all()


@pytest.mark.parametrize("name", SPLITS)
def test_all_pairs_are_uts_authors_in_input_graph(bundle, name):
    t, sp_ = bundle["tables"], bundle["splits"][name]
    cutoff, _ = _cut(bundle, name)
    uts = _uts_before(t, cutoff)
    in_graph = _authors_before(t, cutoff)
    allowed = uts & in_graph
    for key in ("positives", "neg_random", "neg_hard", "train_pool_random", "train_pool_hard"):
        if key not in sp_:
            continue
        ids = set(sp_[key]["author_u"]) | set(sp_[key]["author_v"])
        assert ids <= allowed, f"{key} uses non-UTS or out-of-graph authors"
    for key in ("ranking", "ranking_hard"):
        rk = sp_[key]
        assert set(rk["head"]) | set(rk["cand"]) <= allowed, f"{key} uses non-UTS or out-of-graph authors"


@pytest.mark.parametrize("name", SPLITS)
def test_negatives_unconnected_and_not_window_collaborations(bundle, name):
    t, sp_ = bundle["tables"], bundle["splits"][name]
    cutoff, window_end = _cut(bundle, name)
    neg = _all_negatives(sp_)
    neg_pairs = _pairs(neg)
    assert not (neg_pairs & _graph_pairs(sp_)), "a negative is connected in the input graph"
    assert not (neg_pairs & _window_collab(t, cutoff, window_end)), "a negative collaborated in the window"
    assert not (neg_pairs & _pairs(sp_["positives"])), "a negative is a positive"
    assert not (neg["author_u"] == neg["author_v"]).any()


@pytest.mark.parametrize("name", SPLITS)
def test_hard_negatives_at_distance_exactly_two(bundle, name):
    sp_ = bundle["splits"][name]
    nbrs = _neighbours(sp_)
    hard = pd.concat([sp_[k] for k in ("neg_hard", "train_pool_hard") if k in sp_])
    for u, v in zip(hard["author_u"], hard["author_v"]):
        nu, nv = nbrs.get(u, set()), nbrs.get(v, set())
        assert v not in nu                      # not distance 1
        assert nu & nv                          # a common neighbour -> distance exactly 2


@pytest.mark.parametrize("later", ["val", "test"])
def test_later_positives_absent_from_training_graph(bundle, later):
    train_edges = _graph_pairs(bundle["splits"]["train"])
    assert not (_pairs(bundle["splits"][later]["positives"]) & train_edges)


def test_ranking_candidates_shape(bundle):
    for name in SPLITS:
        sp_ = bundle["splits"][name]
        rk, pos = sp_["ranking"], sp_["positives"]
        k = bundle["cfg"]["negatives"]["ranking_candidates"]
        assert len(rk) == k * len(pos)
        assert (rk.groupby("query_id")["cand"].nunique() == k).all()
        assert (rk["head_idx"].values == pos["u_idx"].values[rk["query_id"]]).all()


def test_negative_counts_one_to_one(bundle):
    for name in SPLITS:
        sp_ = bundle["splits"][name]
        n = len(sp_["positives"])
        assert n > 0
        assert len(sp_["neg_random"]) == n and len(sp_["neg_hard"]) == n
        assert not sp_["neg_random"].duplicated(["author_u", "author_v"]).any()
        assert not sp_["neg_hard"].duplicated(["author_u", "author_v"]).any()


@pytest.mark.parametrize("name", SPLITS)
def test_positive_distance_column(bundle, name):
    """distance == 2 exactly when the pair shares a neighbour in the input graph (protocols rely on it)."""
    sp_ = bundle["splits"][name]
    nbrs = _neighbours(sp_)
    pos = sp_["positives"]
    shares = np.array([bool(nbrs.get(u, set()) & nbrs.get(v, set())) for u, v in zip(pos["author_u"], pos["author_v"])])
    assert ((pos["distance"] == 2).values == shares).all()
    assert (pos["distance"] >= 2).all()


@pytest.mark.parametrize("name", SPLITS)
def test_rank_hard_candidates(bundle, name):
    t, sp_ = bundle["tables"], bundle["splits"][name]
    cutoff, window_end = _cut(bundle, name)
    rk, pos = sp_["ranking_hard"], sp_["positives"]
    k = bundle["cfg"]["negatives"]["ranking_candidates"]
    nbrs = _neighbours(sp_)
    window = _window_collab(t, cutoff, window_end)
    # Queries are exactly the distance-2 positives (minus any with an empty pool), head = author_u.
    d2 = set(np.flatnonzero((pos["distance"] == 2).values))
    q = set(rk["query_id"])
    assert q <= d2
    assert len(d2 - q) == sp_["stats"]["rank_hard_no_candidates"]
    assert (rk["head"].values == pos["author_u"].values[rk["query_id"]]).all()
    per_q = rk.groupby("query_id")["cand"]
    assert (per_q.size() <= k).all() and (per_q.nunique() == per_q.size()).all()
    for h, w in zip(rk["head"], rk["cand"]):
        assert h != w
        assert w not in nbrs.get(h, set())                       # not connected
        assert nbrs.get(h, set()) & nbrs.get(w, set())           # distance exactly 2
        assert (min(h, w), max(h, w)) not in window              # no joint work in the window


# ---------------------------------------------------------------- reproducibility

def _run_train(cfg, t, seed):
    s, neg = cfg["splits"]["train"], cfg["negatives"]
    return make_split(t, all_uts_coauthor_pairs(t["auth"]), s["cutoff"], s["window_end"], seed, 0,
                      neg["ratios"], neg["ranking_candidates"], cfg["splits"]["cold_start_max_works"],
                      pool_ratio=neg["train_pool_ratio"])


def test_same_seed_reproduces_synthetic(synthetic):
    cfg, t = synthetic["cfg"], synthetic["tables"]
    a, b = _run_train(cfg, t, 7), _run_train(cfg, t, 7)
    for key in ("positives", "neg_random", "neg_hard", "ranking", "ranking_hard"):
        pd.testing.assert_frame_equal(a[key], b[key])
    c = _run_train(cfg, t, 8)
    assert not (a["neg_random"].equals(c["neg_random"]) and a["ranking"].equals(c["ranking"]))
    assert not a["ranking_hard"].equals(c["ranking_hard"])


def test_saved_splits_reproduce_from_config_seed(bundle):
    """Rebuilding the train split with the config seed gives exactly the saved files."""
    cfg, t = bundle["cfg"], bundle["tables"]
    fresh = _run_train(cfg, t, cfg["seed"])
    saved = bundle["splits"]["train"]
    for key in ("positives", "neg_random", "neg_hard", "ranking", "ranking_hard", "train_pool_random", "train_pool_hard"):
        pd.testing.assert_frame_equal(fresh[key].reset_index(drop=True), saved[key].reset_index(drop=True),
                                      check_dtype=False)


# ---------------------------------------------------------------- the fixture exercises the hard cases

def test_synthetic_fixture_covers_edge_cases(synthetic):
    t = synthetic["tables"]
    train_nodes = set(synthetic["splits"]["train"]["nodes"]["author_id"])
    train_uts = set(synthetic["splits"]["train"]["nodes"].query("is_uts")["author_id"])
    assert not ({f"A{i:03d}" for i in range(70, 80)} & train_nodes)       # late arrivals absent
    switchers = {f"A{i:03d}" for i in range(80, 90)}
    assert switchers & train_nodes and not (switchers & train_uts)         # present but not yet UTS
    test_uts = set(synthetic["splits"]["test"]["nodes"].query("is_uts")["author_id"])
    assert switchers & test_uts
    assert (t["works"]["n_authors"] > 25).sum() == 2
    tr = synthetic["splits"]["train"]                # pools exist (train only) and are non-trivial
    assert len(tr["train_pool_hard"]) > 0 and len(tr["train_pool_random"]) > 0
    assert "train_pool_hard" not in synthetic["splits"]["val"]
    for name in SPLITS:                       # rank_hard tests must not pass vacuously
        st = synthetic["splits"][name]["stats"]
        assert st["rank_hard_queries"] > 0 and st["rank_hard_rows"] > 0
