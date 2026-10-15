"""Neighbourhood and path heuristics on a split's input graph.

All heuristics use the BINARY adjacency of the input graph (an edge = at least one joint
work before the cutoff); collaboration counts are left for the feature-based models.
Degrees are binary degrees in the input graph, so everything is pre-cutoff by construction.

Pairs are scored in chunks with sparse row products (no Python loop over pairs).
Truncated Katz counts walks of length 1..3:  beta*A + beta^2*A^2 + beta^3*A^3.
For the unconnected pairs we evaluate, the A term is 0 and the ranking depends only on
beta (A^2 + beta*A^3), i.e. beta trades common neighbours against length-3 walks.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import scipy.sparse as sp

HEURISTICS = ("common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment")
CHUNK = 50_000


class GraphContext:
    """Binary adjacency and degree-derived weights for one split, computed once."""

    def __init__(self, A: sp.csr_matrix):
        B = A.copy().tocsr()
        B.data[:] = 1.0
        B.eliminate_zeros()
        self.A = B
        self.deg = np.asarray(B.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            # A common neighbour has degree >= 2, so log(deg) > 0 wherever it is used.
            self.w_aa = np.where(self.deg > 1, 1.0 / np.log(np.maximum(self.deg, 2)), 0.0)
            self.w_ra = np.where(self.deg > 0, 1.0 / np.maximum(self.deg, 1), 0.0)


_CACHE: dict = {"A": None, "ctx": None}


def context(split: dict) -> GraphContext:
    """GraphContext for the split's adjacency, cached for ONE adjacency object at a time.

    The cache holds a reference to the adjacency it was built from and is checked by identity
    (`is`), so it can never return a stale context for a different matrix. (An earlier version
    keyed the cache on id(A), which a new object can reuse after garbage collection; audit m8.)
    """
    if _CACHE["A"] is not split["A"]:
        _CACHE["A"], _CACHE["ctx"] = split["A"], GraphContext(split["A"])
    return _CACHE["ctx"]


def _common(ctx: GraphContext, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    return ctx.A[u].multiply(ctx.A[v]).tocsr()


def _chunked(fn, ctx, u, v):
    out = np.empty(len(u), dtype=np.float64)
    for s in range(0, len(u), CHUNK):
        out[s:s + CHUNK] = fn(ctx, u[s:s + CHUNK], v[s:s + CHUNK])
    return out


def _cn(ctx, u, v):
    return np.asarray(_common(ctx, u, v).sum(axis=1)).ravel()


def _jaccard(ctx, u, v):
    cn = _cn(ctx, u, v)
    union = ctx.deg[u] + ctx.deg[v] - cn
    return np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)


def _aa(ctx, u, v):
    return _common(ctx, u, v) @ ctx.w_aa


def _ra(ctx, u, v):
    return _common(ctx, u, v) @ ctx.w_ra


def _pa(ctx, u, v):
    return ctx.deg[u] * ctx.deg[v]


_FNS = {"common_neighbors": _cn, "jaccard": _jaccard, "adamic_adar": _aa,
        "resource_allocation": _ra, "preferential_attachment": _pa}


def score(name: str, split: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ctx = context(split)
    return _chunked(_FNS[name], ctx, np.asarray(u), np.asarray(v))


def scorer(name: str):
    """score_fn(split, u, v) for the evaluation protocols."""
    if name == "prefer_newer":
        return prefer_newer
    return partial(score, name)


def prefer_newer(split: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference scorer (audit M2): prefer pairs with a newer author, -min(works before the cutoff).

    It uses no pair information at all; it measures how much a protocol rewards newcomer status.
    Pre-cutoff by construction (`n_works_before` in the split's node table)."""
    nw = split["nodes"]["n_works_before"].values.astype(np.float64)
    return -np.minimum(nw[np.asarray(u)], nw[np.asarray(v)])


# ---------------------------------------------------------------- truncated Katz

def katz_walk_counts(split: dict, u: np.ndarray, v: np.ndarray, batch: int = 128) -> np.ndarray:
    """(n_pairs, 3) array of walk counts of length 1, 2, 3 between u and v.

    Grouped by head node: for a batch of heads H, A^2 rows = A[H] @ A (sparse) and
    A^3 rows = (A @ (A^2 rows)^T)^T using the symmetry of A, computed densely per batch.
    """
    A = context(split).A
    u, v = np.asarray(u), np.asarray(v)
    out = np.zeros((len(u), 3))
    out[:, 0] = np.asarray(A[u, v]).ravel()
    heads, inv = np.unique(u, return_inverse=True)
    for s in range(0, len(heads), batch):
        hb = heads[s:s + batch]
        A2 = (A[hb] @ A).toarray().astype(np.float64)        # (b, N)
        A3 = np.asarray(A @ A2.T).T                           # (b, N)
        sel = np.flatnonzero((inv >= s) & (inv < s + len(hb)))
        r = inv[sel] - s
        out[sel, 1] = A2[r, v[sel]]
        out[sel, 2] = A3[r, v[sel]]
    return out


def katz_from_counts(counts: np.ndarray, beta: float) -> np.ndarray:
    return counts @ np.array([beta, beta**2, beta**3])


class KatzScorer:
    """Caches walk counts per pair set so a beta grid costs one computation."""

    def __init__(self, beta: float):
        self.beta = beta
        self._cache: dict = {}

    def __call__(self, split: dict, u, v) -> np.ndarray:
        key = (id(split["A"]), len(u), hash(np.asarray(u).tobytes()), hash(np.asarray(v).tobytes()))
        if key not in self._cache:
            self._cache[key] = katz_walk_counts(split, u, v)
        return katz_from_counts(self._cache[key], self.beta)
