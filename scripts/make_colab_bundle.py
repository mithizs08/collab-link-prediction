"""Build colab_bundle.tar.gz for Phase 6 (GNN training on Colab). Not committed (see .gitignore).

Contents: code (src/, configs/, scripts/, tests/, pyproject.toml, requirements.txt) and ONLY the data
Phase 6 needs:
  data/processed/splits/<split>/*          input graphs (nodes, edges, adjacency) and pair sets
  data/processed/node_features/*           structural node features + author text vectors
plus MANIFEST.json with the size and sha256 of every data file.

  python scripts/make_colab_bundle.py [--out colab_bundle.tar.gz]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["src", "configs", "scripts", "tests", "pyproject.toml", "requirements.txt"]
DATA = ["data/processed/splits", "data/processed/node_features"]
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"}


def files_under(rel: str):
    p = ROOT / rel
    if p.is_file():
        yield p
        return
    for f in sorted(p.rglob("*")):
        if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP) and f.suffix != ".pyc":
            yield f


def sha256(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "colab_bundle.tar.gz"))
    args = ap.parse_args(argv)
    for d in DATA:
        if not (ROOT / d).exists():
            raise SystemExit(f"missing {d}: run `make data splits features` and build node features first")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    raw = 0
    with tarfile.open(args.out, "w:gz", compresslevel=6) as tar:
        for rel in CODE + DATA:
            for f in files_under(rel):
                arc = str(f.relative_to(ROOT))
                tar.add(f, arcname=f"collab-link-prediction/{arc}")
                raw += f.stat().st_size
                if arc.startswith("data/"):
                    manifest["files"][arc] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("collab-link-prediction/MANIFEST.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    size = Path(args.out).stat().st_size
    print(f"{args.out}: {size / 1e6:.1f} MB compressed ({raw / 1e6:.1f} MB uncompressed), "
          f"{len(manifest['files'])} data files")


if __name__ == "__main__":
    main()
