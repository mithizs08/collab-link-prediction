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
