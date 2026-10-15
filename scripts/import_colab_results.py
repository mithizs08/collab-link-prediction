"""Import Phase 6 results downloaded from Google Drive into the repo.

  python scripts/import_colab_results.py <path/to/gnn_results> [--results-dir results] [--scores-dir data/interim/gnn]

Validates every row of gnn_grid.csv, gnn.csv and gnn_determinism.csv against the results schema
(linkpred.results.validate_row), checks that every completed final run has its score files, and then:
  results/gnn_grid.csv, results/gnn.csv, results/gnn_determinism.csv   (replaced, validated)
  results/gnn_versions.json                                             (library / GPU versions used)
  report/gnn/curves/<run>.csv                                           (training curves)
  <scores-dir>/                                                         (pair scores + OOF scores, git-ignored)
Nothing is written unless validation passes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkpred.results import RESULT_COLUMNS, validate_row, write_results  # noqa: E402

CSVS = ("gnn_grid.csv", "gnn.csv", "gnn_determinism.csv")


def _clean(r: dict) -> dict:
    out = {}
    for k in RESULT_COLUMNS:
        v = r.get(k)
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    out["seed"] = int(out["seed"])
    return out


def load_validated(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    if list(df.columns) != list(RESULT_COLUMNS):
        raise ValueError(f"{path.name}: columns {list(df.columns)} != schema {list(RESULT_COLUMNS)}")
    rows = [_clean(r) for r in df.to_dict("records")]
    for i, r in enumerate(rows):
        try:
            validate_row(r)
        except ValueError as e:
            raise ValueError(f"{path.name} row {i}: {e}") from e
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--report-dir", type=Path, default=Path("report/gnn"))
    ap.add_argument("--scores-dir", type=Path, default=Path("data/interim/gnn"))
    args = ap.parse_args(argv)
    src = args.src

    tables = {}
    for name in CSVS:
        if not (src / name).exists():
            raise SystemExit(f"missing {src / name}")
        tables[name] = load_validated(src / name)
    finals = sorted(src.glob("final/*/seed*/meta.json")) + sorted(src.glob("ablation/*/seed*/meta.json"))
    for m in finals:
        for sp in ("val", "test"):
            if not (m.parent / f"scores_{sp}.npz").exists():
                raise SystemExit(f"{m.parent} is complete but has no scores_{sp}.npz")
    if not (src / "versions.json").exists():
        raise SystemExit("missing versions.json")
    print("validated:", {k: len(v) for k, v in tables.items()}, f"{len(finals)} final/ablation runs")

    for name, rows in tables.items():
        write_results(rows, args.results_dir / name, append=False)
    shutil.copy(src / "versions.json", args.results_dir / "gnn_versions.json")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if (src / "run.log").exists():          # audit m3: keep the training log traceable in the repo
        shutil.copy(src / "run.log", args.report_dir / "run.log")
    curves = args.report_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    for c in src.glob("**/curve.csv"):
        shutil.copy(c, curves / ("__".join(c.parent.relative_to(src).parts) + ".csv"))
    for sub in ("final", "ablation", "oof", "determinism"):
        if (src / sub).exists():
            shutil.copytree(src / sub, args.scores_dir / sub, dirs_exist_ok=True)
    print(f"imported into {args.results_dir}, curves -> {curves}, scores -> {args.scores_dir}")
    print("versions:", json.loads((src / "versions.json").read_text()))


if __name__ == "__main__":
    main()
