"""Build the cleaned co-authorship tables from raw OpenAlex works.

Outputs (data/processed):
  works.parquet         one row per kept work (incl. hyper-authored works)
  authorships.parquet   one row per (work, identified author)
  authors.parquet       one row per author ID  -- DESCRIPTIVE ONLY, see below
  edges.parquet         one row per co-author pair per work, u < v
  disambiguation_flags.parquet  same normalised name, >1 ID, same institution
and a cleaning log (report/data_audit/cleaning_log.json) with a count for every rule.

LEAKAGE WARNING: authors.parquet aggregates over ALL years (first/last year,
n_works, primary institution). It must never be used as a model feature;
features for period T are recomputed from works with year < T's cutoff.

Input: either a single JSON array of works (the snapshot from the earlier
project) or a directory of cached API pages from fetch_openalex.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterator

import ijson
import pandas as pd

log = logging.getLogger(__name__)

OA_PREFIX = "https://openalex.org/"
DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)
# Preprints are dropped in favour of the published version when deduplicating.
TYPE_PRIORITY = {"preprint": 1}
# Title+year matching only for titles with at least this many words: short generic
# titles ("Editorial", "Introduction", "Letter from the editors") recur across
# genuinely different works by the same editor. Chosen after inspecting the matches
# (see docs/decisions.md).
TITLE_MIN_WORDS = 5


# ---------------------------------------------------------------- parsing helpers

def strip_oa(x: str | None) -> str | None:
    if not x:
        return None
    return x.rsplit("/", 1)[-1] if x.startswith(OA_PREFIX) else x


def normalize_doi(x: str | None) -> str | None:
    if not x:
        return None
    m = DOI_RE.search(x)
    return m.group(1).lower().rstrip(".") if m else None


def extract_doi(w: dict) -> str | None:
    """Work DOI, falling back to the primary location.

    The earlier snapshot did not select `doi`; for those works the primary
    location's id ("doi:10...") or landing page (https://doi.org/10...) carries it.
    """
    if w.get("doi"):
        return normalize_doi(w["doi"])
    pl = w.get("primary_location") or {}
    for field in ("id", "landing_page_url"):
        v = pl.get(field) or ""
        if v.startswith("doi:") or "doi.org/" in v:
            return normalize_doi(v)
    return None


def normalize_text(s: str | None) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {p: word for word, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def iter_raw_works(source: Path) -> Iterator[dict]:
    """Stream works so the ~0.9 GB snapshot never has to sit in memory at once."""
    if source.is_dir():
        for page in sorted(source.glob("page_*.json")):
            yield from json.loads(page.read_text())["results"]
    else:
        with open(source, "rb") as f:
            yield from ijson.items(f, "item", use_float=True)


def parse_work(w: dict, uts_id: str) -> tuple[dict, list[dict]]:
    auths = w.get("authorships") or []
    work_id = strip_oa(w["id"])
    work = {
        "work_id": work_id,
        "year": w.get("publication_year"),
        "title": w.get("title") or "",
        "doi": extract_doi(w),
        "type": w.get("type"),
        # Byline length INCLUDING authors without an OpenAlex ID: the author cap
        # is about how the paper was written, not about what OpenAlex resolved.
        # The list API caps authorships at 100, so n_authors == 100 means ">= 100".
        "n_authors": len(auths),
    }
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    work["has_abstract"] = bool(abstract)
    work["abstract"] = abstract

    rows = []
    for idx, a in enumerate(auths):
        author = a.get("author") or {}
        insts = a.get("institutions") or []
        inst_ids = [strip_oa(i.get("id")) for i in insts if i.get("id")]
        lineage = {strip_oa(l) for i in insts for l in (i.get("lineage") or [])}
        rows.append({
            "work_id": work_id,
            "author_id": strip_oa(author.get("id")),
            "display_name": author.get("display_name") or a.get("raw_author_name"),
            "author_position": a.get("author_position"),
            "author_index": idx,
            "institution_ids": inst_ids,
            # Lineage catches UTS sub-units should OpenAlex ever split them out.
            "is_uts": uts_id in inst_ids or uts_id in lineage,
        })
    return work, rows


# ---------------------------------------------------------------- cleaning steps

def _canonical_order(works: pd.DataFrame) -> pd.DataFrame:
    """Sort so the preferred copy of a duplicate comes first."""
    return works.assign(
        _type_rank=works["type"].map(TYPE_PRIORITY).fillna(0),
    ).sort_values(
        ["_type_rank", "has_abstract", "n_authors", "work_id"],
        ascending=[True, False, False, True],
    )


def deduplicate(works: pd.DataFrame, auth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate works: same DOI, or same normalised title + year sharing an author.

    Two guards keep distinct works with generic titles apart: the title must have
    at least TITLE_MIN_WORDS words, and the copies must share an author ID.
    Typical true matches: preprint vs. published version, Angewandte's German and
    English editions, consensus statements co-published in several journals.
    """
    stats = {}
    ordered = _canonical_order(works)

    has_doi = ordered["doi"].notna()
    doi_dup = has_doi & ordered.duplicated("doi", keep="first")
    stats["duplicate_doi"] = int(doi_dup.sum())
    ordered = ordered[~doi_dup]

    ordered = ordered.assign(_ntitle=ordered["title"].map(normalize_text))
    authors_of = (
        auth.dropna(subset=["author_id"]).groupby("work_id")["author_id"].agg(set).to_dict()
    )
    title_dups: set[str] = set()
    candidates = ordered[ordered["_ntitle"].str.split().str.len() >= TITLE_MIN_WORDS]
    for _, grp in candidates.groupby(["_ntitle", "year"], sort=False):
        if len(grp) < 2:
            continue
        kept: list[set] = []
        for wid in grp["work_id"]:  # already in canonical order
            a = authors_of.get(wid, set())
            if any(a & k for k in kept):
                title_dups.add(wid)
            else:
                kept.append(a)
    stats["duplicate_title_year"] = len(title_dups)
    ordered = ordered[~ordered["work_id"].isin(title_dups)]

    out = ordered.drop(columns=["_type_rank", "_ntitle"]).sort_values("work_id")
    return out.reset_index(drop=True), stats


def flag_disambiguation(auth: pd.DataFrame) -> pd.DataFrame:
    """(normalised name, institution) groups that map to more than one author ID."""
    a = auth[["author_id", "display_name", "institution_ids"]].explode("institution_ids")
    a = a.dropna(subset=["institution_ids"]).rename(columns={"institution_ids": "institution_id"})
    a["norm_name"] = a["display_name"].map(normalize_text)
    a = a[a["norm_name"] != ""]
    g = a.groupby(["norm_name", "institution_id"])["author_id"].agg(lambda s: sorted(set(s)))
    g = g[g.map(len) > 1]
    return g.rename("author_ids").reset_index().assign(n_ids=lambda d: d["author_ids"].map(len))


def build_edges(works: pd.DataFrame, auth: pd.DataFrame, max_authors: int) -> tuple[pd.DataFrame, dict]:
    stats = {}
    hyper = works["n_authors"] > max_authors
    stats["hyper_works"] = int(hyper.sum())
    stats["hyper_works_truncated_at_100"] = int((works["n_authors"] >= 100).sum())

    ids_by_work = auth.groupby("work_id")["author_id"].agg(list)
    hyper_ids = set(works.loc[hyper, "work_id"])
    # Would-be edges use distinct identified authors; for works capped at 100
    # authorships this is a lower bound.
    stats["hyper_would_be_edges"] = int(sum(
        len(set(ids)) * (len(set(ids)) - 1) // 2
        for wid, ids in ids_by_work.items() if wid in hyper_ids
    ))

    year_of = works.set_index("work_id")["year"]
    rows, self_loops = [], 0
    for wid, ids in ids_by_work.items():
        if wid in hyper_ids or wid not in year_of.index:
            continue
        y = int(year_of[wid])
        for a, b in combinations(ids, 2):
            if a == b:  # same author listed twice on one byline
                self_loops += 1
                continue
            u, v = (a, b) if a < b else (b, a)
            rows.append((u, v, y, wid))
    edges = pd.DataFrame(rows, columns=["author_u", "author_v", "year", "work_id"])
    stats["self_loops_removed"] = self_loops
    dup = edges.duplicated(["author_u", "author_v", "work_id"])
    stats["duplicate_pair_work_rows_removed"] = int(dup.sum())
    edges = edges[~dup].sort_values(["year", "work_id", "author_u", "author_v"])
    edges["year"] = edges["year"].astype("int16")
    return edges.reset_index(drop=True), stats


def build_authors(auth: pd.DataFrame, works: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    a = auth.merge(works[["work_id", "year"]], on="work_id")
    names = a.groupby("author_id")["display_name"].agg(lambda s: s.value_counts().index[0])
    agg = a.groupby("author_id").agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_works=("work_id", "nunique"),
    )
    # Count distinct works, not rows: an author can appear twice on one byline.
    agg["n_uts_works"] = a[a["is_uts"]].groupby("author_id")["work_id"].nunique()
    agg["n_uts_works"] = agg["n_uts_works"].fillna(0).astype(int)
    # Primary institution: most frequent; ties go to the most recently used one.
    inst = a[["author_id", "institution_ids", "year"]].explode("institution_ids").dropna()
    inst = (
        inst.groupby(["author_id", "institution_ids"])["year"].agg(["size", "max"])
        .reset_index().sort_values(["author_id", "size", "max"], ascending=[True, False, False])
        .drop_duplicates("author_id").set_index("author_id")["institution_ids"]
    )
    out = agg.assign(display_name=names, primary_institution=inst)
    flagged_ids = {i for ids in flagged["author_ids"] for i in ids}
    out["disambiguation_flag"] = out.index.isin(flagged_ids)
    cols = ["display_name", "first_year", "last_year", "n_works", "primary_institution",
            "n_uts_works", "disambiguation_flag"]
    return out[cols].reset_index()


# ---------------------------------------------------------------- pipeline

def build(cfg: dict) -> dict:
    d, paths = cfg["data"], cfg["paths"]
    # Snapshot location: env LINKPRED_SNAPSHOT_PATH > config data.snapshot_path > cached API pages
    # (data/raw/openalex_pages, filled by `python -m linkpred.data.fetch_openalex`). See README.
    env_snap = os.environ.get("LINKPRED_SNAPSHOT_PATH")
    if env_snap and not Path(env_snap).exists():
        # An explicitly requested snapshot must exist: never fall back silently (audit N6).
        raise FileNotFoundError(f"LINKPRED_SNAPSHOT_PATH={env_snap!r} does not exist")
    snap = env_snap or d.get("snapshot_path")
    if snap and Path(snap).exists():
        source = Path(snap)
    else:
        source = Path(paths["raw"]) / "openalex_pages"
        if snap:
            log.warning("configured snapshot %s not found; using downloaded pages in %s", snap, source)
    if not source.exists():
        raise FileNotFoundError(
            f"No OpenAlex input found (snapshot {snap!r} missing and {source} absent). Set LINKPRED_SNAPSHOT_PATH "
            "to a works JSON snapshot, or run `python -m linkpred.data.fetch_openalex` (needs OPENALEX_MAILTO, "
            "optionally OPENALEX_API_KEY) to download the pages; see README 'Rebuilding the data'.")
    uts_id, max_authors = d["institution_id"], d["max_authors_per_work"]
    out = Path(paths["processed"])
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"source": str(source)}

    works_rows, auth_rows = [], []
    for w in iter_raw_works(source):
        work, rows = parse_work(w, uts_id)
        works_rows.append(work)
        auth_rows.extend(rows)
    works = pd.DataFrame(works_rows)
    auth = pd.DataFrame(auth_rows)
    stats["raw_works"] = len(works)
    stats["raw_authorships"] = len(auth)
    stats["duplicate_work_ids"] = int(works.duplicated("work_id").sum())
    works = works.drop_duplicates("work_id")

    # 1. publication year
    missing_year = works["year"].isna()
    stats["missing_year_dropped"] = int(missing_year.sum())
    works = works[~missing_year].astype({"year": "int16"})
    if d.get("start_year") and d.get("end_year"):
        in_range = works["year"].between(d["start_year"], d["end_year"])
        stats["out_of_year_range_dropped"] = int((~in_range).sum())
        works = works[in_range]

    # 1b. non-collaboration work types (before dedup, so they can never be the kept copy)
    excluded = works["type"].isin(d.get("exclude_types") or [])
    stats["excluded_type_dropped"] = int(excluded.sum())
    stats["excluded_type_counts"] = works.loc[excluded, "type"].value_counts().to_dict()
    works = works[~excluded]

    # 2. duplicates
    works, dup_stats = deduplicate(works, auth[auth["work_id"].isin(works["work_id"])])
    stats.update(dup_stats)
    auth = auth[auth["work_id"].isin(works["work_id"])]

    # 3. authorships without an author ID
    no_id = auth["author_id"].isna()
    stats["authorships_without_author_id_dropped"] = int(no_id.sum())
    stats["works_affected_by_missing_author_id"] = int(auth.loc[no_id, "work_id"].nunique())
    auth = auth[~no_id].reset_index(drop=True)
    stats["works_without_uts_authorship"] = int(
        len(set(works["work_id"]) - set(auth.loc[auth["is_uts"], "work_id"]))
    )

    # 4. disambiguation (flag only, never merged)
    flagged = flag_disambiguation(auth)
    stats["disambig_groups"] = len(flagged)
    stats["disambig_author_ids"] = int(len({i for ids in flagged["author_ids"] for i in ids}))
    uts_flagged = flagged[flagged["institution_id"] == uts_id]
    stats["disambig_groups_uts"] = len(uts_flagged)
    stats["disambig_author_ids_uts"] = int(len({i for ids in uts_flagged["author_ids"] for i in ids}))

    # 5. edges (hyper-authored works excluded, self-loops and duplicate rows removed)
    edges, edge_stats = build_edges(works, auth, max_authors)
    stats.update(edge_stats)

    authors = build_authors(auth, works, flagged)

    stats.update(
        works=len(works),
        authorships=len(auth),
        authors=len(authors),
        uts_authors=int((authors["n_uts_works"] > 0).sum()),
        edge_rows=len(edges),
        distinct_pairs=int(len(edges.drop_duplicates(["author_u", "author_v"]))),
        max_authors_per_work=max_authors,
    )

    works.to_parquet(out / "works.parquet", index=False)
    auth[["work_id", "author_id", "author_position", "author_index", "institution_ids", "is_uts"]] \
        .to_parquet(out / "authorships.parquet", index=False)
    authors.to_parquet(out / "authors.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    flagged.to_parquet(out / "disambiguation_flags.parquet", index=False)

    log_dir = Path(paths["report"]) / "data_audit"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "cleaning_log.json").write_text(json.dumps(stats, indent=2) + "\n")
    for k, v in stats.items():
        log.info("%s: %s", k, v)
    return stats


def main(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(cfg)
