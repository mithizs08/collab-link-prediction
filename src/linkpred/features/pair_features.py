"""Pair features for one split, computed only from works published before its cutoff.

`FeatureContext(tables, cutoff, graph, emb, params)` filters every table to year < cutoff
itself (it is handed the FULL tables on purpose: tests/test_features.py perturbs works in
and after the cutoff and requires identical features). The input graph passed in must be
the split's pre-cutoff graph (splits.build_input_graph).

All features are symmetric in (u, v) (per-author quantities enter as min / max), so a pair's
features do not depend on which endpoint is the ranking head.

Families
  structural  Phase 3 heuristics (CN, Jaccard, AA, RA, PA, truncated Katz), degree min/max,
              shortest-path distance capped at 5 + disconnected flag (different component),
              same Louvain community (input graph, seeded), shared institutions, common
              neighbours split into UTS / non-UTS
  temporal    career age, works in the last 2 years, years since last work (min/max each),
              recency-weighted CN and AA (each shared co-author w weighted by
              0.5**((c - last_uw)/h) * 0.5**((c - last_vw)/h), half-life h chosen on
              validation), years since the most recent shared collaboration
              (c - max_w min(last_uw, last_vw); 11 when there is no common neighbour)
  text        cosine of author vectors (mean embedding of the author's pre-cutoff works),
              similarity of each author to the mean vector of the OTHER author's co-authors
              (min / max), and a flag that both authors have text
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from linkpred.models import heuristics as H

log = logging.getLogger(__name__)

STRUCTURAL = ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment",
              "katz", "deg_min", "deg_max", "distance", "disconnected", "same_community",
              "shared_institutions", "cn_uts", "cn_non_uts"]
TEMPORAL = ["career_age_min", "career_age_max", "recent_works_min", "recent_works_max",
            "years_since_last_min", "years_since_last_max", "rw_cn", "rw_aa", "years_since_shared"]
TEXT = ["text_cos", "nbr_text_sim_min", "nbr_text_sim_max", "text_both"]
FAMILIES = {"structural": STRUCTURAL, "temporal": TEMPORAL, "text": TEXT}
ALL_FEATURES = STRUCTURAL + TEMPORAL + TEXT
# Non-negative heavy-tailed counts: log1p before standardising for logistic regression.
LOG1P = {"common_neighbors", "adamic_adar", "resource_allocation", "preferential_attachment", "katz",
         "deg_min", "deg_max", "shared_institutions", "cn_uts", "cn_non_uts", "recent_works_min",
         "recent_works_max", "rw_cn", "rw_aa"}
DIST_CAP = 5
NO_SHARED = 11  # > any possible gap (data start 2015, latest cutoff 2025)


def load_feature_tables(processed) -> dict[str, pd.DataFrame]:
    from pathlib import Path

    processed = Path(processed)
    works = pd.read_parquet(processed / "works.parquet", columns=["work_id", "year"])
    auth = pd.read_parquet(processed / "authorships.parquet",
                           columns=["work_id", "author_id", "is_uts", "institution_ids"])
    auth = auth.merge(works, on="work_id", validate="many_to_one")
    return {"works": works, "auth": auth}


class FeatureContext:
    def __init__(self, tables: dict, cutoff: int, nodes: pd.DataFrame, graph_edges: pd.DataFrame,
                 A: sp.csr_matrix, emb: np.ndarray | None, emb_ids: pd.DataFrame | None, params: dict,
                 communities: bool = True):
        self.c = cutoff
        self.p = params
        self.n = len(nodes)
        idx = pd.Series(nodes["node_idx"].values, index=nodes["author_id"].values)
        auth = tables["auth"][tables["auth"]["year"] < cutoff]          # <- the only time filter
        auth = auth[auth["author_id"].isin(idx.index)]
        a_idx = idx.loc[auth["author_id"]].values

        self.g = H.GraphContext(A)
        self.A = self.g.A
        self.is_uts = nodes["is_uts"].values.astype(float)

        # Edge attributes as sparse matrices with the adjacency pattern.
        u, v = graph_edges["u_idx"].values, graph_edges["v_idx"].values
        last = graph_edges["last_year"].values.astype(float)
        self.L = sp.csr_matrix((np.r_[last, last], (np.r_[u, v], np.r_[v, u])), shape=(self.n, self.n))

        # Components (disconnected flag) and Louvain communities on the weighted input graph.
        _, self.comp = connected_components(self.A, directed=False)
        # Louvain is only needed for pair features; node features (GNN) skip it.
        self.community = self._louvain(graph_edges, params["seed"]) if communities else None

        # Per-author activity before the cutoff.
        aw = pd.DataFrame({"a": a_idx, "w": auth["work_id"].values, "y": auth["year"].values}).drop_duplicates(["a", "w"])
        first = aw.groupby("a")["y"].min()
        lasty = aw.groupby("a")["y"].max()
        recent = aw[aw["y"] >= cutoff - 2].groupby("a").size()
        self.career = np.zeros(self.n); self.career[first.index] = cutoff - first.values
        self.since_last = np.full(self.n, float(NO_SHARED)); self.since_last[lasty.index] = cutoff - lasty.values
        self.recent = np.zeros(self.n); self.recent[recent.index] = recent.values

        # Author x institution incidence.
        ex = pd.DataFrame({"a": a_idx, "i": auth["institution_ids"].values}).explode("i").dropna().drop_duplicates()
        inst = pd.Index(ex["i"].unique())
        self.I = sp.csr_matrix((np.ones(len(ex)), (ex["a"].values, inst.get_indexer(ex["i"]))),
                               shape=(self.n, max(len(inst), 1)))

        # Text: author vector = mean embedding of pre-cutoff works with text.
        self.has_text = np.zeros(self.n, dtype=bool)
        self.has_abstract = np.zeros(self.n, dtype=bool)
        self.U = self.Nm = None
        if emb is not None:
            row = pd.Series(np.arange(len(emb_ids)), index=emb_ids["work_id"].values)
            aw_t = aw[aw["w"].isin(row.index)]
            r = row.loc[aw_t["w"]].values
            keep = emb_ids["has_text"].values[r]
            P = sp.csr_matrix((np.ones(keep.sum()), (aw_t["a"].values[keep], r[keep])), shape=(self.n, len(emb_ids)))
            cnt = np.asarray(P.sum(axis=1)).ravel()
            self.has_text = cnt > 0
            abs_rows = emb_ids["has_abstract"].values[r]
            self.has_abstract[np.unique(aw_t["a"].values[abs_rows])] = True
            U = np.asarray(P @ emb, dtype=np.float32)
            self.U = _normalise(U)
            Nm = np.asarray(self.A @ self.U, dtype=np.float32)   # sum of neighbours' unit vectors
            self.Nm = _normalise(Nm)

    def _louvain(self, graph_edges: pd.DataFrame, seed: int) -> np.ndarray:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_weighted_edges_from(zip(graph_edges["u_idx"].values.tolist(), graph_edges["v_idx"].values.tolist(),
                                      graph_edges["weight"].values.astype(float).tolist()))
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        lab = np.empty(self.n, dtype=np.int64)
        for k, c in enumerate(sorted(comms, key=lambda s: min(s))):   # deterministic labels
            lab[list(c)] = k
        return lab

    # ------------------------------------------------------------ features

    def pair_distance(self, u: np.ndarray, v: np.ndarray, chunk: int = 128) -> np.ndarray:
        out = np.full(len(u), float(DIST_CAP))
        heads, inv = np.unique(u, return_inverse=True)
        for s in range(0, len(heads), chunk):
            src = heads[s:s + chunk]
            D = dijkstra(self.A, directed=False, indices=src, unweighted=True, limit=DIST_CAP)
            sel = np.flatnonzero((inv >= s) & (inv < s + len(src)))
            out[sel] = np.minimum(D[inv[sel] - s, v[sel]], DIST_CAP)
        return out

    def recency_matrix(self, half_life: float) -> sp.csr_matrix:
        T = self.L.copy()
        T.data = 0.5 ** ((self.c - T.data) / half_life)
        return T

    def rw_scores(self, u, v, half_life: float, T: sp.csr_matrix | None = None):
        T = T if T is not None else self.recency_matrix(half_life)
        out_cn, out_aa = np.empty(len(u)), np.empty(len(u))
        for s in range(0, len(u), H.CHUNK):
            M = T[u[s:s + H.CHUNK]].multiply(T[v[s:s + H.CHUNK]]).tocsr()
            out_cn[s:s + H.CHUNK] = np.asarray(M.sum(axis=1)).ravel()
            out_aa[s:s + H.CHUNK] = M @ self.g.w_aa
        return out_cn, out_aa

    def compute(self, u: np.ndarray, v: np.ndarray, families=("structural", "temporal", "text")) -> pd.DataFrame:
        u, v = np.asarray(u), np.asarray(v)
        f: dict[str, np.ndarray] = {}
        if "structural" in families:
            for name in H.HEURISTICS:
                f[name] = H._chunked(H._FNS[name], self.g, u, v)
            f["katz"] = H.katz_from_counts(H.katz_walk_counts({"A": self.A}, u, v), self.p["katz_beta"])
            du, dv = self.g.deg[u], self.g.deg[v]
            f["deg_min"], f["deg_max"] = np.minimum(du, dv), np.maximum(du, dv)
            f["distance"] = self.pair_distance(u, v)
            f["disconnected"] = (self.comp[u] != self.comp[v]).astype(float)
            f["same_community"] = (self.community[u] == self.community[v]).astype(float)
            f["shared_institutions"] = np.asarray(self.I[u].multiply(self.I[v]).sum(axis=1)).ravel()
            cn_uts = np.empty(len(u))
            for s in range(0, len(u), H.CHUNK):
                cn_uts[s:s + H.CHUNK] = H._common(self.g, u[s:s + H.CHUNK], v[s:s + H.CHUNK]) @ self.is_uts
            f["cn_uts"], f["cn_non_uts"] = cn_uts, f["common_neighbors"] - cn_uts
        if "temporal" in families:
            for name, arr in (("career_age", self.career), ("recent_works", self.recent),
                              ("years_since_last", self.since_last)):
                f[f"{name}_min"], f[f"{name}_max"] = np.minimum(arr[u], arr[v]), np.maximum(arr[u], arr[v])
            f["rw_cn"], f["rw_aa"] = self.rw_scores(u, v, self.p["half_life"])
            last_shared = np.zeros(len(u))
            for s in range(0, len(u), H.CHUNK):
                M = self.L[u[s:s + H.CHUNK]].minimum(self.L[v[s:s + H.CHUNK]])
                last_shared[s:s + H.CHUNK] = M.max(axis=1).toarray().ravel()
            f["years_since_shared"] = np.where(last_shared > 0, self.c - last_shared, NO_SHARED)
        if "text" in families:
            if self.U is None:
                raise ValueError("text features need embeddings")
            f["text_cos"] = _rowdot(self.U, u, v)
            a, b = _rowdot2(self.U, self.Nm, u, v), _rowdot2(self.U, self.Nm, v, u)
            f["nbr_text_sim_min"], f["nbr_text_sim_max"] = np.minimum(a, b), np.maximum(a, b)
            f["text_both"] = (self.has_text[u] & self.has_text[v]).astype(float)
        return pd.DataFrame(f)

    def coverage(self) -> dict:
        uts = self.is_uts.astype(bool)
        return {"nodes": int(self.n), "share_nodes_with_text": float(self.has_text.mean()),
                "share_uts_nodes_with_text": float(self.has_text[uts].mean()) if uts.any() else 0.0,
                "share_nodes_with_abstract": float(self.has_abstract.mean()),
                "share_uts_nodes_with_abstract": float(self.has_abstract[uts].mean()) if uts.any() else 0.0}


def _normalise(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)


def _rowdot(X, u, v, chunk=50_000):
    out = np.empty(len(u), dtype=np.float64)
    for s in range(0, len(u), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", X[u[s:s + chunk]], X[v[s:s + chunk]])
    return out


def _rowdot2(X, Y, u, v, chunk=50_000):
    out = np.empty(len(u), dtype=np.float64)
    for s in range(0, len(u), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", X[u[s:s + chunk]], Y[v[s:s + chunk]])
    return out
