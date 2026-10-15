"""Invariants of the processed co-authorship tables.

Every invariant runs on a small synthetic snapshot pushed through the real
`build()` (always), and on data/processed when it exists (skipped otherwise,
since data/ is not in git).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from linkpred.config import load_config
from linkpred.data.build import build, deduplicate, extract_doi, normalize_text, reconstruct_abstract

UTS = "I114017466"
OA = "https://openalex.org/"
PROCESSED = Path("data/processed")


def _auth(aid, inst=UTS, name=None, pos="middle"):
    return {
        "author_position": pos,
        "author": {"id": f"{OA}{aid}" if aid else None, "display_name": name or f"Name {aid}"},
        "institutions": [{"id": f"{OA}{inst}", "lineage": [f"{OA}{inst}"]}] if inst else [],
    }


def _work(wid, year, authors, title=None, doi=None, wtype="article", abstract=None):
    return {
        "id": f"{OA}{wid}", "publication_year": year, "type": wtype,
        "title": title or f"A distinctive study number {wid} of graphs",
        "authorships": authors,
        "abstract_inverted_index": abstract,
        "primary_location": {"id": f"doi:{doi}", "landing_page_url": f"https://doi.org/{doi}"} if doi else {},
    }


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("snap")
    works = [
        _work("W1", 2018, [_auth("A1"), _auth("A2"), _auth("A3", inst="I999")], doi="10.1234/a",
              abstract={"hello": [0], "world": [1]}),
        _work("W2", 2019, [_auth("A2"), _auth("A1"), _auth("A2")]),          # repeated author -> self-loop
        _work("W3", None, [_auth("A1"), _auth("A4")]),                         # missing year
        _work("W4", 2019, [_auth("A1"), _auth("A2")], doi="10.1234/A"),          # DOI duplicate of W1 (case)
        _work("W5", 2020, [_auth(f"A{i}") for i in range(10, 40)]),           # 30 authors -> hyper
        _work("W6", 2020, [_auth("A1"), _auth(None), _auth("A5")]),           # author without ID
        _work("W7", 2021, [_auth("A5"), _auth("A6")], wtype="preprint",
              title="Same long title for a preprint and article"),
        _work("W8", 2021, [_auth("A5"), _auth("A7")],
              title="Same long title for a preprint and article"),            # title+year dup, shares A5
        _work("W9", 2021, [_auth("A8"), _auth("A9")], title="Editorial"),
        _work("W10", 2021, [_auth("A8"), _auth("A6")], title="Editorial"),    # short title: NOT a dup
        _work("W11", 2021, [_auth("A1", name="Jane Smith"), _auth("A20", name="Jane  SMITH")]),
        _work("W12", 2021, [_auth("A1"), _auth("A99")], wtype="erratum"),  # excluded type
    ]
    snap = tmp / "snap.json"
    snap.write_text(json.dumps(works))
    cfg = load_config()
    cfg["data"] = {**cfg["data"], "snapshot_path": str(snap), "start_year": 2015, "end_year": 2025}
    cfg["paths"] = {**cfg["paths"], "processed": str(tmp / "processed"), "report": str(tmp / "report")}
    stats = build(cfg)
    p = tmp / "processed"
    return {
        "stats": stats,
        "works": pd.read_parquet(p / "works.parquet"),
        "authors": pd.read_parquet(p / "authors.parquet"),
        "edges": pd.read_parquet(p / "edges.parquet"),
        "authorships": pd.read_parquet(p / "authorships.parquet"),
        "max_authors": cfg["data"]["max_authors_per_work"],
    }


@pytest.fixture(scope="module")
def real():
    if not (PROCESSED / "edges.parquet").exists():
        pytest.skip("data/processed not built (run `make data`)")
    return {
        "works": pd.read_parquet(PROCESSED / "works.parquet", columns=["work_id", "year", "n_authors"]),
        "authors": pd.read_parquet(PROCESSED / "authors.parquet", columns=["author_id"]),
        "edges": pd.read_parquet(PROCESSED / "edges.parquet"),
        "max_authors": load_config()["data"]["max_authors_per_work"],
    }


@pytest.fixture(params=["synthetic", "real"])
def tables(request):
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------- invariants (both datasets)

def test_edges_are_ordered(tables):
    e = tables["edges"]
    assert (e["author_u"] < e["author_v"]).all()


def test_no_self_loops(tables):
    e = tables["edges"]
    assert not (e["author_u"] == e["author_v"]).any()


def test_no_duplicate_pair_work_rows(tables):
    assert not tables["edges"].duplicated(["author_u", "author_v", "work_id"]).any()


def test_edge_authors_exist(tables):
    e, known = tables["edges"], set(tables["authors"]["author_id"])
    assert set(e["author_u"]).issubset(known) and set(e["author_v"]).issubset(known)


def test_edge_year_matches_work_year(tables):
    m = tables["edges"].merge(tables["works"][["work_id", "year"]], on="work_id",
                              suffixes=("", "_work"), how="left", validate="many_to_one")
    assert m["year_work"].notna().all()
    assert (m["year"] == m["year_work"]).all()


def test_hyper_authored_works_produce_no_edges(tables):
    w = tables["works"]
    hyper = set(w.loc[w["n_authors"] > tables["max_authors"], "work_id"])
    assert hyper, "fixture/data should contain at least one hyper-authored work"
    assert not tables["edges"]["work_id"].isin(hyper).any()


# ---------------------------------------------------------------- cleaning rules (synthetic)

def test_cleaning_counts(synthetic):
    s = synthetic["stats"]
    assert s["missing_year_dropped"] == 1
    assert s["duplicate_doi"] == 1
    assert s["duplicate_title_year"] == 1
    assert s["authorships_without_author_id_dropped"] == 1
    assert s["hyper_works"] == 1
    assert s["hyper_would_be_edges"] == 30 * 29 // 2
    assert s["self_loops_removed"] == 1
    assert s["duplicate_pair_work_rows_removed"] == 1   # (A1,A2) twice in W2


def test_excluded_types_dropped(synthetic):
    assert synthetic["stats"]["excluded_type_dropped"] == 1
    assert "W12" not in set(synthetic["works"]["work_id"])
    assert not synthetic["edges"]["work_id"].eq("W12").any()
    assert "A99" not in set(synthetic["authors"]["author_id"])


def test_dedup_keeps_preferred_copy(synthetic):
    ids = set(synthetic["works"]["work_id"])
    assert "W1" in ids and "W4" not in ids        # DOI dup: W1 has abstract
    assert "W8" in ids and "W7" not in ids        # article preferred over preprint
    assert {"W9", "W10"} <= ids                   # generic short title kept apart


def test_hyper_work_kept_in_works(synthetic):
    assert "W5" in set(synthetic["works"]["work_id"])


def test_disambiguation_flag(synthetic):
    a = synthetic["authors"].set_index("author_id")
    assert a.loc["A1", "disambiguation_flag"] and a.loc["A20", "disambiguation_flag"]
    assert synthetic["stats"]["disambig_groups"] == 1


def test_is_uts_and_counts(synthetic):
    au = synthetic["authorships"]
    assert not au.loc[au["author_id"] == "A3", "is_uts"].any()
    a = synthetic["authors"].set_index("author_id")
    assert (a["n_uts_works"] <= a["n_works"]).all()
    assert a.loc["A2", "n_works"] == 2            # W1, W2 (W2 lists A2 twice)


# ---------------------------------------------------------------- helpers

def test_helpers():
    assert reconstruct_abstract({"b": [1], "a": [0, 2]}) == "a b a"
    assert reconstruct_abstract(None) == ""
    assert extract_doi({"primary_location": {"landing_page_url": "https://doi.org/10.1002/ANIE.1"}}) == "10.1002/anie.1"
    assert extract_doi({"primary_location": {"id": "pmid:123"}}) is None
    assert normalize_text("Évaluation: Graph-Based  Methods!") == "evaluation graph based methods"


def test_deduplicate_requires_shared_author():
    works = pd.DataFrame({
        "work_id": ["W1", "W2"], "year": [2020, 2020], "doi": [None, None], "type": ["article"] * 2,
        "title": ["One long identical title for two papers"] * 2, "has_abstract": [True] * 2, "n_authors": [2, 2],
    })
    auth = pd.DataFrame({"work_id": ["W1", "W1", "W2", "W2"], "author_id": ["A", "B", "C", "D"]})
    kept, stats = deduplicate(works, auth)
    assert len(kept) == 2 and stats["duplicate_title_year"] == 0


def test_missing_env_snapshot_is_an_error(tmp_path, monkeypatch):
    """Audit N6: an explicitly set LINKPRED_SNAPSHOT_PATH that does not exist must not fall back silently."""
    monkeypatch.setenv("LINKPRED_SNAPSHOT_PATH", str(tmp_path / "missing.json"))
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "raw": str(tmp_path / "raw"), "processed": str(tmp_path / "p")}
    (tmp_path / "raw" / "openalex_pages").mkdir(parents=True)        # a stale fallback that must NOT be used
    with pytest.raises(FileNotFoundError, match="LINKPRED_SNAPSHOT_PATH"):
        build(cfg)
