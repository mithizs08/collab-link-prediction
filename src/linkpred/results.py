"""The only sanctioned way to write results.

Every number reported anywhere must come from a CSV written through
`write_results`, which enforces the fixed long-format schema.

Schema (extended 2026-10-14 for evaluation protocols, see docs/decisions.md):
  model, split, protocol, subset, negative_type, seed, metric, value, ci_low, ci_high
- protocol: one of PROTOCOLS
- subset: "all" or a named stratum (e.g. "distance=3", "cold_start", "full_candidates")
- negative_type: the negatives the protocol compares against ("random" / "hard")
- seed: training seed for learned models; deterministic models record the config seed
  (the seed that fixed the evaluation pairs)
- ci_low / ci_high: bootstrap 95% interval, empty when not computed
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

RESULT_COLUMNS = (
    "model", "split", "protocol", "subset", "negative_type", "seed", "metric", "value", "ci_low", "ci_high",
)
VALID_SPLITS = {"train", "val", "test"}
PROTOCOLS = {"hard", "rank_random", "rank_hard", "random", "diagnostic_all_vs_hard"}
NEGATIVE_TYPES = {"random", "hard"}
OPTIONAL = {"ci_low", "ci_high"}


def _is_missing(x) -> bool:
    return x is None or x == "" or (isinstance(x, float) and math.isnan(x))


def validate_row(row: Mapping) -> None:
    required = set(RESULT_COLUMNS) - OPTIONAL
    missing = required - set(row)
    extra = set(row) - set(RESULT_COLUMNS)
    if missing or extra:
        raise ValueError(f"Bad result row: missing={sorted(missing)} extra={sorted(extra)}")
    if row["split"] not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {row['split']!r}")
    if row["protocol"] not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}, got {row['protocol']!r}")
    if row["negative_type"] not in NEGATIVE_TYPES:
        raise ValueError(f"negative_type must be one of {sorted(NEGATIVE_TYPES)}, got {row['negative_type']!r}")
    if not isinstance(row["seed"], int):
        raise ValueError(f"seed must be int, got {row['seed']!r}")
    if math.isnan(float(row["value"])):
        raise ValueError(f"NaN value for {row['model']}/{row['metric']}")
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if _is_missing(lo) != _is_missing(hi):
        raise ValueError("ci_low and ci_high must both be set or both be empty")
    if not _is_missing(lo) and not (float(lo) <= float(hi)):
        raise ValueError(f"ci_low > ci_high for {row['model']}/{row['metric']}")


def write_results(rows: Iterable[Mapping], path: str | Path, append: bool = True) -> Path:
    """Validate and write result rows. Appends by default so stages can add models."""
    rows = list(rows)
    for row in rows:
        validate_row(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and path.exists() and path.stat().st_size > 0)
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(
            {k: ("" if _is_missing(row.get(k)) else row.get(k)) for k in RESULT_COLUMNS} for row in rows
        )
    return path
