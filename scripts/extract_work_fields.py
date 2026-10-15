"""Phase 8, step 5 input: the OpenAlex field of each work (for the cross-discipline error analysis).

  python scripts/extract_work_fields.py

data/processed/works.parquet does not keep OpenAlex topics, so this streams the raw snapshot (read-only; same source
resolution as linkpred.data.build: env LINKPRED_SNAPSHOT_PATH > config data.snapshot_path) and writes, for every work in
works.parquet, its PRIMARY topic's subfield / field / domain. The primary topic is OpenAlex's first `topics` entry
(the list is ordered by score). No existing table is modified.

Outputs: data/processed/work_fields.parquet (work_id, field_id, field, subfield_id, subfield, domain, topic_score),
         report/fields/coverage.csv (share of works with a field, by year).
Author field (defined in scripts/phase8_errors.py, docs/decisions.md): the modal field of the author's works published
BEFORE the split's cutoff (rule 1).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkpred.config import load_config  # noqa: E402
from linkpred.data.build import iter_raw_works, strip_oa  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed"])
    env = os.environ.get("LINKPRED_SNAPSHOT_PATH")
    if env and not Path(env).exists():
        raise FileNotFoundError(f"LINKPRED_SNAPSHOT_PATH={env!r} does not exist")
    src = Path(env or cfg["data"]["snapshot_path"])
    keep = set(pd.read_parquet(processed / "works.parquet", columns=["work_id"]).work_id)
    rows, seen = [], 0
    for w in iter_raw_works(src):
        seen += 1
        wid = strip_oa(w["id"])
        if wid not in keep:
            continue
        t = (w.get("topics") or [None])[0]
        if not t:
            rows.append({"work_id": wid})
            continue
        rows.append({"work_id": wid, "field_id": strip_oa(t["field"]["id"]), "field": t["field"]["display_name"],
                     "subfield_id": strip_oa(t["subfield"]["id"]), "subfield": t["subfield"]["display_name"],
                     "domain": t["domain"]["display_name"], "topic_score": float(t.get("score") or 0.0)})
    df = pd.DataFrame(rows).drop_duplicates("work_id")
    df.to_parquet(processed / "work_fields.parquet", index=False)
    works = pd.read_parquet(processed / "works.parquet", columns=["work_id", "year"]).merge(df, on="work_id", how="left")
    rep = Path(cfg["paths"]["report"]) / "fields"
    rep.mkdir(parents=True, exist_ok=True)
    cov = works.groupby("year").field_id.apply(lambda s: s.notna().mean()).rename("share_with_field").reset_index()
    cov.to_csv(rep / "coverage.csv", index=False)
    print(f"raw works streamed {seen:,}; kept {len(df):,}; with field {df.field_id.notna().mean():.3f}; "
          f"distinct fields {df.field_id.nunique()}")


if __name__ == "__main__":
    main()
