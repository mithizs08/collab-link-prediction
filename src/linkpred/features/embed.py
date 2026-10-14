"""Sentence embeddings of every work, computed once and cached.

Text = title + ". " + abstract when an abstract exists, title only otherwise (HTML tags
stripped). all-MiniLM-L6-v2 truncates at 256 word pieces. Embeddings are L2-normalised.

Leakage: a work's embedding depends only on its own text, so it carries no information
from other works or later years. Author vectors are built per split from works published
before the cutoff only (linkpred.features.pair_features). Caveat (docs/data_audit.md,
Limitations): the pretrained model was trained on text that may postdate the cutoffs.

Cache (data/interim/embeddings/): work_embeddings.npy (float32, n_works x 384),
work_ids.parquet (row order), meta.json (model, text rule, counts).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAG_RE = re.compile(r"<[^>]+>")
log = logging.getLogger(__name__)


def work_text(title: str | None, abstract: str | None) -> str:
    t = TAG_RE.sub("", title or "").strip()
    a = TAG_RE.sub("", abstract or "").strip()
    return f"{t}. {a}" if a else t


def cache_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "embeddings"


def build_embeddings(cfg: dict, batch_size: int = 64, device: str | None = None) -> Path:
    out = cache_dir(cfg)
    if (out / "work_embeddings.npy").exists():
        log.info("embeddings cached in %s", out)
        return out
    import torch
    from sentence_transformers import SentenceTransformer

    works = pd.read_parquet(Path(cfg["paths"]["processed"]) / "works.parquet",
                            columns=["work_id", "title", "abstract", "has_abstract"])
    texts = [work_text(t, a) for t, a in zip(works["title"], works["abstract"])]
    has_text = np.array([bool(s.strip(". ")) for s in texts])
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    emb[~has_text] = 0.0          # empty text: no vector (excluded from author means)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "work_embeddings.npy", emb)
    pd.DataFrame({"work_id": works["work_id"], "has_text": has_text,
                  "has_abstract": works["has_abstract"]}).to_parquet(out / "work_ids.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({
        "model": MODEL, "dim": int(emb.shape[1]), "device": device, "n_works": int(len(works)),
        "n_with_text": int(has_text.sum()), "n_with_abstract": int(works["has_abstract"].sum()),
        "text_rule": "title + '. ' + abstract if abstract else title; HTML tags stripped; max 256 tokens",
    }, indent=2) + "\n")
    return out


def load_embeddings(cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    d = cache_dir(cfg)
    return np.load(d / "work_embeddings.npy"), pd.read_parquet(d / "work_ids.parquet")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from linkpred.config import load_config

    logging.basicConfig(level=logging.INFO)
    build_embeddings(load_config())
