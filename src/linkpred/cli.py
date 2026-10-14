"""Pipeline entry point used by the Makefile: `python -m linkpred.cli --config X <stage>`.

Stages are registered here as they are implemented; unimplemented stages fail
loudly instead of silently producing nothing.
"""

from __future__ import annotations

import argparse
import sys

from linkpred.config import load_config, set_seed

STAGES = ("data", "splits", "baselines", "features", "train", "evaluate", "report")

def _data(cfg: dict) -> None:
    from linkpred.data import audit, build

    build.main(cfg)
    audit.main(cfg)


def _splits(cfg: dict) -> None:
    from linkpred.data import splits

    splits.main(cfg)


def _features(cfg: dict) -> None:
    import logging

    from linkpred.features.build import build_features
    from linkpred.features.embed import build_embeddings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build_embeddings(cfg)
    build_features(cfg)


# stage name -> callable(cfg). Filled in as phases are built.
IMPLEMENTED: dict = {"data": _data, "splits": _splits, "features": _features}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linkpred")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    fn = IMPLEMENTED.get(args.stage)
    if fn is None:
        print(f"Stage '{args.stage}' is not implemented yet.", file=sys.stderr)
        return 1
    fn(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
