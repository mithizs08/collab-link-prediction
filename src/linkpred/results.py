"""The only sanctioned way to write results.

Every number reported anywhere must come from a CSV written through
`write_results`, which enforces the fixed long-format schema.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

RESULT_COLUMNS = ("model", "split", "negative_type", "seed", "metric", "value")
VALID_SPLITS = {"train", "val", "test"}


def validate_row(row: Mapping) -> None:
    missing = set(RESULT_COLUMNS) - set(row)
    extra = set(row) - set(RESULT_COLUMNS)
    if missing or extra:
        raise ValueError(f"Bad result row: missing={sorted(missing)} extra={sorted(extra)}")
    if row["split"] not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {row['split']!r}")
    if not isinstance(row["seed"], int):
        raise ValueError(f"seed must be int, got {row['seed']!r}")
    value = float(row["value"])
    if math.isnan(value):
        raise ValueError(f"NaN value for {row['model']}/{row['metric']}")


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
        writer.writerows({k: row[k] for k in RESULT_COLUMNS} for row in rows)
    return path
