"""Config loading and the single entry point for seeding randomness."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return cfg


def set_seed(seed: int) -> None:
    """Seed every RNG the project uses.

    All code must call this (with cfg["seed"] or one of cfg["seeds"]) rather than
    seeding libraries ad hoc, so a run is reproducible from the config alone.
    numpy/torch are imported lazily so lightweight stages and tests don't need them.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
