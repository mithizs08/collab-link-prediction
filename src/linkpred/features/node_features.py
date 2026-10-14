"""Node features for the GNNs, per split, from works published before the split's cutoff.

Built by the SAME FeatureContext code as the Phase 4 pair features (never recomputed
differently), so the leakage guarantees of tests/test_features.py carry over:

  structural  [log1p(degree), career_age, log1p(works in the last 2 years), is_uts]
  text        author text vector (unit mean of the author's pre-cutoff work embeddings;
              zeros if the author has no text), 384 dims

Saved to data/processed/node_features/<split>_{structural,text}.npy, rows in node_idx order.
Text vectors are stored as float16 (halves the Colab bundle; cosine error < 1e-3).
Standardisation of the structural block uses TRAIN-split statistics (gnn.py), which are
pre-cutoff for every split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from linkpred.data.splits import SPLITS, load_split
from linkpred.features.embed import load_embeddings
from linkpred.features.pair_features import FeatureContext, load_feature_tables

log = logging.getLogger(__name__)
STRUCTURAL_NAMES = ["log1p_degree", "career_age", "log1p_recent_works", "is_uts"]


def node_feature_arrays(ctx: FeatureContext) -> tuple[np.ndarray, np.ndarray]:
    structural = np.column_stack([np.log1p(ctx.g.deg), ctx.career, np.log1p(ctx.recent), ctx.is_uts]).astype(np.float32)
    return structural, ctx.U.astype(np.float32)


def build_node_features(cfg: dict) -> Path:
    out = Path(cfg["paths"]["processed"]) / "node_features"
    out.mkdir(parents=True, exist_ok=True)
    tables = load_feature_tables(cfg["paths"]["processed"])
    emb, emb_ids = load_embeddings(cfg)
    params = json.loads((Path(cfg["paths"]["processed"]) / "features" / "params.json").read_text())
    meta = {}
    for name in SPLITS:
        sp_ = load_split(cfg["paths"]["splits"], name)
        ctx = FeatureContext(tables, cfg["splits"][name]["cutoff"], sp_["nodes"], sp_["graph_edges"], sp_["A"],
                             emb, emb_ids, params, communities=False)
        s, t = node_feature_arrays(ctx)
        np.save(out / f"{name}_structural.npy", s)
        np.save(out / f"{name}_text.npy", t.astype(np.float16))
        meta[name] = {"nodes": int(len(s)), "share_with_text": float(ctx.has_text.mean())}
        log.info("%s node features: %s", name, meta[name])
    (out / "meta.json").write_text(json.dumps({"structural": STRUCTURAL_NAMES, "text_dim": int(emb.shape[1]),
                                               "splits": meta}, indent=2) + "\n")
    return out


def load_node_features(processed: Path | str, split: str, kind: str) -> np.ndarray:
    """kind: 'structural' or 'text' (float32 either way)."""
    return np.load(Path(processed) / "node_features" / f"{split}_{kind}.npy").astype(np.float32)
