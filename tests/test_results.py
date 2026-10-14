import csv

import pytest

from linkpred.results import RESULT_COLUMNS, write_results


def _row(**overrides):
    row = dict(model="cn", split="val", negative_type="random", seed=0, metric="roc_auc", value=0.8)
    row.update(overrides)
    return row


def test_write_and_append(tmp_path):
    path = tmp_path / "r.csv"
    write_results([_row()], path)
    write_results([_row(seed=1)], path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == RESULT_COLUMNS
    assert [r["seed"] for r in rows] == ["0", "1"]


@pytest.mark.parametrize(
    "bad",
    [
        {"split": "holdout"},
        {"seed": "0"},
        {"value": float("nan")},
    ],
)
def test_rejects_invalid_rows(tmp_path, bad):
    with pytest.raises(ValueError):
        write_results([_row(**bad)], tmp_path / "r.csv")


def test_rejects_missing_column(tmp_path):
    row = _row()
    del row["negative_type"]
    with pytest.raises(ValueError):
        write_results([row], tmp_path / "r.csv")
