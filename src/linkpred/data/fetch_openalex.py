"""Re-pull UTS works from the OpenAlex API, caching every raw page to disk.

Not needed for the current dataset (built from an existing snapshot, see
docs/decisions.md), but kept so the dataset can be rebuilt from scratch.

Politeness / robustness:
- cursor paging (the only way to page past 10k results), 200 results per page
- `mailto` (env OPENALEX_MAILTO) and optional `api_key` (env OPENALEX_API_KEY)
- retries with exponential backoff on 429 / 5xx / connection errors, honouring Retry-After
- each page is written atomically to data/raw/openalex_pages/page_NNNNN.json together
  with the cursor that produced it and the next cursor, so a rerun (or a resumed,
  interrupted run) reads cached pages instead of re-downloading them
- a manifest stores the query; rerunning with a different query refuses to mix pages

Usage:  python -m linkpred.data.fetch_openalex --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.openalex.org/works"
# Verified 2026-10-14 via /institutions/I114017466 and ror:03f0f6041 (see decisions.md).
UTS_INSTITUTION_ID = "I114017466"
SELECT_FIELDS = (
    "id,doi,title,publication_year,publication_date,type,"
    "authorships,abstract_inverted_index,primary_location"
)
RETRY_STATUS = {429, 500, 502, 503, 504}


def build_params(institution_id: str, start_year: int, end_year: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "filter": f"institutions.id:{institution_id},publication_year:{start_year}-{end_year}",
        "select": SELECT_FIELDS,
        "per-page": 200,
    }
    if os.getenv("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    return params


def get_with_retries(
    session: requests.Session,
    params: dict[str, Any],
    max_retries: int = 5,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> dict:
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=60)
        except requests.RequestException:
            if attempt == max_retries:
                raise
            sleep(backoff * 2**attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            sleep(float(retry_after) if retry_after else backoff * 2**attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")


def _atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    tmp.replace(path)


def fetch(
    out_dir: Path,
    institution_id: str,
    start_year: int,
    end_year: int,
    session: requests.Session | None = None,
    delay: float = 0.1,
    sleep=time.sleep,
) -> int:
    """Fetch all pages into out_dir; return the number of works across all pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    params = build_params(institution_id, start_year, end_year)

    # Credentials are excluded from the manifest so they never land on disk.
    query = {k: v for k, v in params.items() if k not in ("mailto", "api_key")}
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != query:
            raise RuntimeError(f"{out_dir} holds pages from a different query; use a new directory")
    else:
        _atomic_write_json(query, manifest)

    cursor, page_no, n_works = "*", 1, 0
    while cursor:
        page_path = out_dir / f"page_{page_no:05d}.json"
        if page_path.exists():
            page = json.loads(page_path.read_text())
        else:
            data = get_with_retries(session, {**params, "cursor": cursor}, sleep=sleep)
            page = {
                "cursor": cursor,
                "next_cursor": data["meta"].get("next_cursor"),
                "count": data["meta"].get("count"),
                "results": data["results"],
            }
            _atomic_write_json(page, page_path)
            sleep(delay)
        n_works += len(page["results"])
        # An empty page ends the stream even if a cursor is still returned.
        cursor = page["next_cursor"] if page["results"] else None
        page_no += 1
    return n_works


def main(argv: list[str] | None = None) -> None:
    from linkpred.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["data"]
    out = Path(cfg["paths"]["raw"]) / "openalex_pages"
    n = fetch(out, d["institution_id"], d["start_year"], d["end_year"])
    print(f"{n} works in {out}")


if __name__ == "__main__":
    main()
