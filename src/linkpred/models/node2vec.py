"""node2vec embeddings of a split's input graph, alignment across splits, and edge operators.

Inputs are ONLY the split's pre-cutoff adjacency (binary); no window information exists in it
(tests/test_leakage.py: model inputs are identical when all works in/after the cutoff are deleted).

Cross-split alignment. An embedding is only defined up to rotation, and each split's graph is
embedded separately, so a classifier fitted on train-split pairs cannot be applied to raw
val/test embeddings. Val and test embeddings are mapped into the train embedding's space by
orthogonal Procrustes, fitted on authors present with >= 1 co-author in both graphs (both graphs
are pre-cutoff, so no leakage). The alignment-free alternative -- cosine similarity of the raw
embeddings within a split -- is reported as `node2vec_cosine` as a check on the alignment.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.linalg import orthogonal_procrustes

log = logging.getLogger(__name__)
_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
OPERATORS = ("hadamard", "average", "l1", "l2")


def train_embeddings(A: sp.csr_matrix, p: float, q: float, seed: int, cfg: dict, epochs: int | None = None) -> np.ndarray:
    from torch_geometric.nn import Node2Vec

    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch-cluster's BIASED walks (p or q != 1) draw from the C library's rand(), which
    # torch.manual_seed does not touch: without this, p/q != 1 runs are not reproducible
    # (verified 2026-10-14; unbiased walks use torch's generator and were already deterministic).
    _LIBC.srand(seed)
    B = A.tocoo()
    edge_index = torch.tensor(np.vstack([B.row, B.col]), dtype=torch.long)
    model = Node2Vec(edge_index, embedding_dim=cfg["embedding_dim"], walk_length=cfg["walk_length"],
                     context_size=cfg["context_size"], walks_per_node=cfg["walks_per_node"],
                     num_negative_samples=cfg["num_negative_samples"], p=p, q=q, sparse=True,
                     num_nodes=A.shape[0])
    g = torch.Generator().manual_seed(seed)
    loader = model.loader(batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=g)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=cfg["lr"])
    model.train()
    for ep in range(epochs or cfg["epochs"]):
        total = 0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            opt.step()
            total += float(loss)
        log.debug("epoch %d loss %.4f", ep, total / max(len(loader), 1))
    return model.embedding.weight.detach().cpu().numpy().astype(np.float32)


def align_to(emb: np.ndarray, nodes: pd.DataFrame, ref_emb: np.ndarray, ref_nodes: pd.DataFrame) -> np.ndarray:
    """Rotate `emb` into `ref_emb`'s space using authors with degree > 0 in both graphs."""
    a = nodes.loc[nodes["degree"] > 0, ["author_id", "node_idx"]]
    b = ref_nodes.loc[ref_nodes["degree"] > 0, ["author_id", "node_idx"]]
    m = a.merge(b, on="author_id", suffixes=("", "_ref"))
    R, _ = orthogonal_procrustes(emb[m["node_idx"].values], ref_emb[m["node_idx_ref"].values])
    return emb @ R


def edge_features(emb: np.ndarray, u: np.ndarray, v: np.ndarray, op: str) -> np.ndarray:
    a, b = emb[u], emb[v]
    if op == "hadamard":
        return a * b
    if op == "average":
        return (a + b) / 2
    if op == "l1":
        return np.abs(a - b)
    if op == "l2":
        return (a - b) ** 2
    raise ValueError(op)


def cosine(emb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    a, b = emb[u], emb[v]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), den, out=np.zeros(len(u)), where=den > 0)


def cached_embeddings(cache: Path, split_name: str, A, p, q, seed, cfg) -> np.ndarray:
    # Every training hyperparameter is in the name, so a config change never reuses stale embeddings.
    f = cache / (f"{split_name}_p{p:g}_q{q:g}_s{seed}_d{cfg['embedding_dim']}_wl{cfg['walk_length']}"
                 f"_c{cfg['context_size']}_w{cfg['walks_per_node']}_lr{cfg['lr']:g}_e{cfg['epochs']}"
                 f"_b{cfg['batch_size']}.npy")
    if f.exists():
        return np.load(f)
    cache.mkdir(parents=True, exist_ok=True)
    emb = train_embeddings(A, p, q, seed, cfg)
    np.save(f, emb)
    return emb
