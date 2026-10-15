"""Colab import (audit N7): every import refreshes results/gnn_colab.csv, which gnn_analysis.py reads."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from linkpred.results import RESULT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _importer():
    spec = importlib.util.spec_from_file_location("import_colab_results", ROOT / "scripts" / "import_colab_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["import_colab_results"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(model, value):
    return {"model": model, "split": "val", "protocol": "hard_active", "subset": "macro", "negative_type": "hard_active",
            "seed": 0, "metric": "roc_auc", "value": value, "ci_low": None, "ci_high": None}


def _bundle(d: Path, value: float):
    d.mkdir(parents=True)
    for name in ("gnn_grid.csv", "gnn.csv", "gnn_determinism.csv"):
        pd.DataFrame([_row("gnn_x", value)], columns=list(RESULT_COLUMNS)).to_csv(d / name, index=False)
    (d / "versions.json").write_text(json.dumps({"torch": "x"}))


def test_reimport_refreshes_gnn_colab(tmp_path):
    imp = _importer()
    res = tmp_path / "results"
    args = ["--results-dir", str(res), "--report-dir", str(tmp_path / "rep"), "--scores-dir", str(tmp_path / "sc")]
    _bundle(tmp_path / "a", 0.6)
    imp.main([str(tmp_path / "a")] + args)
    assert pd.read_csv(res / "gnn_colab.csv").value.tolist() == [0.6]
    pd.DataFrame([_row("gnn_x", 0.1)]).to_csv(res / "gnn.csv", index=False)     # gnn_analysis re-evaluated
    _bundle(tmp_path / "b", 0.7)
    imp.main([str(tmp_path / "b")] + args)
    assert pd.read_csv(res / "gnn_colab.csv").value.tolist() == [0.7]
    assert pd.read_csv(res / "gnn.csv").value.tolist() == [0.7]
