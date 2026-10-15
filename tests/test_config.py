import random

from linkpred.config import load_config, set_seed


def test_default_config_has_required_sections():
    cfg = load_config("configs/default.yaml")
    for key in ("seed", "seeds", "paths", "data", "splits", "negatives", "models"):
        assert key in cfg
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["data"]["max_authors_per_work"] == 25
    assert set(cfg["splits"]) >= {"train", "val", "test"}


def test_split_cutoffs_are_ordered_when_set():
    # Guards against a val/test window that overlaps or precedes training.
    cfg = load_config("configs/default.yaml")
    s = cfg["splits"]
    cutoffs = [s[p]["cutoff"] for p in ("train", "val", "test")]
    if all(c is not None for c in cutoffs):
        assert cutoffs == sorted(cutoffs) and len(set(cutoffs)) == 3
        for p in ("train", "val", "test"):
            assert s[p]["window_end"] >= s[p]["cutoff"]
        # A label window must end before the next period's cutoff, otherwise the
        # same new edges would be labels in one period and graph input in the next.
        assert s["train"]["window_end"] < s["val"]["cutoff"]
        assert s["val"]["window_end"] < s["test"]["cutoff"]


def test_set_seed_is_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_decided_splits_and_target():
    cfg = load_config("configs/default.yaml")
    s = cfg["splits"]
    assert [(s[p]["cutoff"], s[p]["window_end"]) for p in ("train", "val", "test")] == [
        (2023, 2023), (2024, 2024), (2025, 2025)]
    assert set(cfg["data"]["exclude_types"]) == {"erratum", "peer-review", "paratext"}


def test_every_config_key_is_read_by_code():
    """Audit m5: no decorative config keys. Every leaf key must be read by name somewhere in src/ or scripts/."""
    import pathlib
    import re
    cfg = load_config("configs/default.yaml")
    code = "\n".join(p.read_text() for p in list(pathlib.Path("src").rglob("*.py")) + list(pathlib.Path("scripts").glob("*.py")))

    def leaves(d, path=()):
        for k, v in d.items():
            if isinstance(v, dict) and k not in ("sage", "gat"):
                yield from leaves(v, path + (k,))
            else:
                yield path + (k,)
    unused = [".".join(map(str, p)) for p in leaves(cfg) if not re.search(rf"""["']{re.escape(str(p[-1]))}["']""", code)]
    assert not unused, f"config keys never read: {unused}"
