import csv

import pytest

from linkpred.results import RESULT_COLUMNS, write_results


def _row(**overrides):
    row = dict(model="cn", split="val", protocol="hard", subset="all", negative_type="hard",
               seed=0, metric="roc_auc", value=0.8, ci_low=0.78, ci_high=0.82)
    row.update(overrides)
    return row


def test_write_and_append(tmp_path):
    path = tmp_path / "r.csv"
    write_results([_row()], path)
    write_results([_row(seed=1, ci_low=None, ci_high=None)], path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == RESULT_COLUMNS
    assert [r["seed"] for r in rows] == ["0", "1"]
    assert rows[1]["ci_low"] == "" and rows[1]["ci_high"] == ""


@pytest.mark.parametrize(
    "bad",
    [
        {"split": "holdout"},
        {"protocol": "all_vs_all"},
        {"negative_type": "degree"},
        {"seed": "0"},
        {"value": float("nan")},
        {"ci_low": 0.9, "ci_high": 0.8},
        {"ci_low": 0.7, "ci_high": None},
    ],
)
def test_rejects_invalid_rows(tmp_path, bad):
    with pytest.raises(ValueError):
        write_results([_row(**bad)], tmp_path / "r.csv")


def test_rejects_missing_column(tmp_path):
    row = _row()
    del row["protocol"]
    with pytest.raises(ValueError):
        write_results([row], tmp_path / "r.csv")
