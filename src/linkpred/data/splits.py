"""Leak-free temporal link-prediction splits.

For each split with cutoff year c and label window [c, window_end]:

input graph   ALL authors with an authorship in a work with year < c (isolated authors
              included) and ALL co-authorship edges from works with year < c. Edge weight =
              number of joint works; first_year / last_year of collaboration. Node attributes
              (is_uts, n_works_before) also use only works with year < c.
UTS author    >= 1 UTS-affiliated authorship in a work with year < c.
positives     UTS-UTS pairs, both in the input graph, whose FIRST joint work of any size is in
              the window and who have an edge (<= max_authors work) in the window.
"collab in window"
              pairs with ANY joint work in the window, including hyper-authored works. No
              negative or ranking candidate may be such a pair (this covers every positive).
neg_random    uniform unconnected UTS-UTS pairs (1 per positive by default).
neg_hard      uniform among unconnected UTS-UTS pairs at shortest-path distance exactly 2 in
              the full input graph (paths may pass through non-UTS authors).
ranking       per positive (u, v): K candidates (u, w), w a UTS node in the graph, w != u,
              (u, w) unconnected and not a collab-in-window pair.
ranking_hard  per distance-2 positive (u, v): up to K candidates (u, w) as above but with w at
              distance exactly 2 from u; fewer than K when fewer exist (count in meta.json).

All sampling uses numpy Generators derived from cfg["seed"] and the split index, one
independent stream per sampler, so outputs are reproducible and do not depend on the
order in which other samplers run.

Output format, data/processed/splits/<split>/:
  nodes.parquet        node_idx, author_id, is_uts, n_works_before, degree
  graph_edges.parquet  u_idx < v_idx, author_u, author_v, weight, first_year, last_year
  adjacency.npz        scipy CSR, symmetric, weighted (fast load for heuristics / PyG)
  positives.parquet    author_u < author_v, u_idx, v_idx, distance (unweighted, in the input
                       graph; inf = disconnected), cold_start (an endpoint with <= cold_start_max_works
                       works before the cutoff)
  neg_random.parquet   same columns
  neg_hard.parquet     same columns + n_common (common neighbours in the input graph)
  ranking.parquet      query_id (row of positives), head_idx, head, cand_idx, cand
  ranking_hard.parquet same columns; distance-2 positives only, <= K candidates at distance 2 from head
  train_pool_random.parquet, train_pool_hard.parquet   (train split only) pool_ratio x positives
                       training negatives of each type, in random order (prefix = uniform sample)
  meta.json            counts, distance distribution, cold-start stats, config used
Node indices are per split (author_id is the stable key across splits).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
PAIR_COLS = ["author_u", "author_v", "u_idx", "v_idx"]


# ---------------------------------------------------------------- loading

def load_tables(processed: Path | str) -> dict[str, pd.DataFrame]:
    processed = Path(processed)
    works = pd.read_parquet(processed / "works.parquet", columns=["work_id", "year", "n_authors"])
    auth = pd.read_parquet(processed / "authorships.parquet", columns=["work_id", "author_id", "is_uts"])
    auth = auth.merge(works[["work_id", "year"]], on="work_id", validate="many_to_one")
    edges = pd.read_parquet(processed / "edges.parquet")
    return {"works": works, "auth": auth, "edges": edges}


def all_uts_coauthor_pairs(auth: pd.DataFrame) -> pd.DataFrame:
    """(author_u, author_v, year) for every joint work of two ever-UTS authors, any work size.

    edges.parquet omits hyper-authored works; this table puts them back so that
    "first co-authorship" and "collaborated in the window" count every joint work.
    """
    ever_uts = set(auth.loc[auth["is_uts"], "author_id"])
    a = auth.loc[auth["author_id"].isin(ever_uts), ["work_id", "author_id", "year"]].drop_duplicates()
    m = a.merge(a[["work_id", "author_id"]], on="work_id", suffixes=("_u", "_v"))
    m = m[m["author_id_u"] < m["author_id_v"]]
    return m.rename(columns={"author_id_u": "author_u", "author_id_v": "author_v"})[
        ["author_u", "author_v", "year"]].drop_duplicates()


# ---------------------------------------------------------------- graph

def build_input_graph(t: dict, cutoff: int):
    auth_b = t["auth"][t["auth"]["year"] < cutoff]
    edges_b = t["edges"][t["edges"]["year"] < cutoff]

    nodes = pd.DataFrame({"author_id": np.sort(auth_b["author_id"].unique())})
    nodes["node_idx"] = np.arange(len(nodes), dtype=np.int64)
    uts_ids = set(auth_b.loc[auth_b["is_uts"], "author_id"])
    nodes["is_uts"] = nodes["author_id"].isin(uts_ids)
    nodes["n_works_before"] = nodes["author_id"].map(auth_b.groupby("author_id")["work_id"].nunique()).astype(int)

    g = edges_b.groupby(["author_u", "author_v"], as_index=False).agg(
        weight=("work_id", "nunique"), first_year=("year", "min"), last_year=("year", "max"))
    idx = pd.Series(nodes["node_idx"].values, index=nodes["author_id"])
    g["u_idx"] = idx.loc[g["author_u"]].values
    g["v_idx"] = idx.loc[g["author_v"]].values
    # author_u < author_v (string order) and node_idx follows sorted author_id, so u_idx < v_idx.
    g = g[["u_idx", "v_idx", "author_u", "author_v", "weight", "first_year", "last_year"]]

    n = len(nodes)
    A = sp.coo_matrix(
        (np.r_[g["weight"], g["weight"]].astype(np.float32), (np.r_[g["u_idx"], g["v_idx"]], np.r_[g["v_idx"], g["u_idx"]])),
        shape=(n, n)).tocsr()
    nodes["degree"] = np.diff(A.indptr)
    return nodes[["node_idx", "author_id", "is_uts", "n_works_before", "degree"]], g, A


# ---------------------------------------------------------------- helpers

def _keys(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    return lo.astype(np.int64) * n + hi


def _key(a: int, b: int, n: int) -> int:
    return (a * n + b) if a < b else (b * n + a)


def _pairs_frame(keys: np.ndarray, n: int, nodes: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    keys = np.asarray(keys, dtype=np.int64)
    if sort:
        keys = np.sort(keys)
    u, v = keys // n, keys % n
    ids = nodes["author_id"].values
    return pd.DataFrame({"author_u": ids[u], "author_v": ids[v], "u_idx": u, "v_idx": v})


def _sample_unconnected(rng, pool: np.ndarray, k: int, forbidden: set, n: int) -> np.ndarray:
    """Uniform distinct unordered pairs from `pool` (node indices), avoiding `forbidden` keys."""
    chosen: list[int] = []
    seen: set[int] = set()
    budget = 200 * k + 10_000
    while len(chosen) < k and budget > 0:
        batch = rng.integers(0, len(pool), size=(2 * (k - len(chosen)) + 16, 2))
        for i, j in batch:
            budget -= 1
            if i == j:
                continue
            key = _key(int(pool[i]), int(pool[j]), n)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            chosen.append(key)
            if len(chosen) == k:
                break
    if len(chosen) < k:
        raise RuntimeError(f"could only sample {len(chosen)} of {k} random negatives")
    return np.array(chosen, dtype=np.int64)


def positive_distances(A: sp.csr_matrix, pos: pd.DataFrame, chunk: int = 256) -> np.ndarray:
    """Unweighted shortest-path distance of each positive pair in the input graph (inf = disconnected)."""
    B = A.copy()
    B.data[:] = 1.0
    heads = pos["u_idx"].values
    out = np.full(len(pos), np.inf)
    uniq = np.unique(heads)
    for s in range(0, len(uniq), chunk):
        src = uniq[s:s + chunk]
        D = shortest_path(B, method="D", unweighted=True, directed=False, indices=src)
        row = {h: r for r, h in enumerate(src)}
        mask = np.isin(heads, src)
        rows = np.array([row[h] for h in heads[mask]])
        out[mask] = D[rows, pos["v_idx"].values[mask]]
    return out


def distance_summary(d: np.ndarray) -> dict:
    n = len(d)
    counts = {
        "d2": int((d == 2).sum()), "d3": int((d == 3).sum()),
        "d4plus": int(((d >= 4) & np.isfinite(d)).sum()), "disconnected": int((~np.isfinite(d)).sum()),
        "d1_or_less": int((d <= 1).sum()),
    }
    return {**counts, **{f"share_{k}": (v / n if n else 0.0) for k, v in counts.items()}}


# ---------------------------------------------------------------- one split

def make_split(t: dict, uts_pairs: pd.DataFrame, cutoff: int, window_end: int, seed: int, split_no: int,
               neg_ratio: dict, n_rank: int, cold_max_works: int, pool_ratio: int = 0) -> dict:
    nodes, g, A = build_input_graph(t, cutoff)
    n = len(nodes)
    idx = pd.Series(nodes["node_idx"].values, index=nodes["author_id"])
    uts_pool = nodes.loc[nodes["is_uts"], "node_idx"].values  # sorted -> deterministic
    uts_set = set(nodes.loc[nodes["is_uts"], "author_id"])

    connected = set(_keys(g["u_idx"].values, g["v_idx"].values, n).tolist())

    # Pairs of (ever-)UTS authors by first joint work (any size) and window collaborations.
    first_any = uts_pairs.groupby(["author_u", "author_v"], as_index=False)["year"].min()
    win_any = uts_pairs[uts_pairs["year"].between(cutoff, window_end)]
    in_graph_uts = win_any["author_u"].isin(uts_set) & win_any["author_v"].isin(uts_set)
    win_any = win_any[in_graph_uts].drop_duplicates(["author_u", "author_v"])
    collab_window = set(_keys(idx.loc[win_any["author_u"]].values, idx.loc[win_any["author_v"]].values, n).tolist())

    # Positives.
    ew = t["edges"][t["edges"]["year"].between(cutoff, window_end)].drop_duplicates(["author_u", "author_v"])
    ew = ew[ew["author_u"].isin(uts_set) & ew["author_v"].isin(uts_set)]
    ew = ew.merge(first_any, on=["author_u", "author_v"], suffixes=("", "_first"))
    stats = {"window_uts_pairs_with_edge": len(ew)}
    prior_hyper = ew["year_first"] < cutoff
    pk = _keys(idx.loc[ew["author_u"]].values, idx.loc[ew["author_v"]].values, n)
    prior_edge = np.isin(pk, np.fromiter(connected, dtype=np.int64)) if connected else np.zeros(len(pk), bool)
    stats["excluded_connected_in_graph"] = int(prior_edge.sum())
    stats["excluded_prior_hyper_work_only"] = int((prior_hyper.values & ~prior_edge).sum())
    pos_keys = pk[~prior_hyper.values & ~prior_edge]
    positives = _pairs_frame(pos_keys, n, nodes)

    # Independent, reproducible streams.
    # spawn(k)[:j] == spawn(j): adding streams (rank_hard, training pools) leaves earlier outputs unchanged.
    r_rand, r_hard, r_rank, r_rank_hard, r_pool_rand, r_pool_hard = (
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, split_no]).spawn(6))
    forbidden = connected | collab_window

    k_rand = int(round(neg_ratio.get("random", 1) * len(positives)))
    neg_random = _pairs_frame(_sample_unconnected(r_rand, uts_pool, k_rand, forbidden, n), n, nodes)

    # Hard: all UTS-UTS pairs with >= 1 common neighbour (any node), minus connected / window pairs.
    Ab = A.copy()
    Ab.data[:] = 1.0
    Bu = Ab[uts_pool]
    C = sp.triu((Bu @ Bu.T).tocsr(), k=1).tocoo()
    cand = _keys(uts_pool[C.row], uts_pool[C.col], n)
    n_common = C.data.astype(np.int64)
    ok = ~np.isin(cand, np.fromiter(forbidden, dtype=np.int64))
    cand, n_common = cand[ok], n_common[ok]
    order = np.argsort(cand)
    cand, n_common = cand[order], n_common[order]
    k_hard = int(round(neg_ratio.get("hard_2hop", 1) * len(positives)))
    stats["hard_candidate_pool"] = int(len(cand))
    if k_hard > len(cand):
        log.warning("hard pool (%d) smaller than requested (%d); taking all", len(cand), k_hard)
        k_hard = len(cand)
    pick = np.sort(r_hard.choice(len(cand), size=k_hard, replace=False))
    neg_hard = _pairs_frame(cand[pick], n, nodes)
    neg_hard["n_common"] = n_common[pick]

    # Training-negative pools (train split only): pool_ratio x positives of each type, same
    # constraints as the evaluation negatives. Rows are in random order, so any prefix is a
    # uniform sample -- training mixes take prefixes. May overlap the 1:1 sets above.
    train_pools = {}
    if pool_ratio:
        k = pool_ratio * len(positives)
        pr = _sample_unconnected(r_pool_rand, uts_pool, k, forbidden, n)   # sampling order is random
        train_pools["train_pool_random"] = _pairs_frame(pr, n, nodes, sort=False)
        ph = r_pool_hard.choice(len(cand), size=min(k, len(cand)), replace=False)  # random order
        train_pools["train_pool_hard"] = _pairs_frame(cand[ph], n, nodes, sort=False)

    # Ranking candidates.
    rows = []
    for q, (h, partner) in enumerate(zip(positives["u_idx"].values, positives["v_idx"].values)):
        got: list[int] = []
        tried = 0
        while len(got) < n_rank:
            tried += 1
            if tried > 100 * n_rank + 1000:
                raise RuntimeError(f"cannot find {n_rank} ranking candidates for node {h}")
            w = int(uts_pool[r_rank.integers(0, len(uts_pool))])
            if w == h or w in got:
                continue
            if _key(int(h), w, n) in forbidden:
                continue
            got.append(w)
        rows.extend((q, h, w) for w in got)
    rk = pd.DataFrame(rows, columns=["query_id", "head_idx", "cand_idx"])
    ids = nodes["author_id"].values
    rk["head"], rk["cand"] = ids[rk["head_idx"]], ids[rk["cand_idx"]]

    # Diagnostics.
    dist = positive_distances(A, positives)
    nwb = nodes.set_index("node_idx")["n_works_before"]
    cold_u = nwb.loc[positives["u_idx"]].values <= cold_max_works
    cold_v = nwb.loc[positives["v_idx"]].values <= cold_max_works
    deg = nodes.set_index("node_idx")["degree"]
    isolated = (deg.loc[positives["u_idx"]].values == 0) | (deg.loc[positives["v_idx"]].values == 0)
    # Evaluation strata, both computed from the pre-cutoff graph only.
    positives["distance"] = dist
    positives["cold_start"] = cold_u | cold_v

    # Hard ranking candidates (rank_hard protocol): for each distance-2 positive (u, v), up to
    # n_rank w drawn uniformly from UTS authors at distance exactly 2 from u, excluding u's
    # neighbours and anyone with a joint work with u in the window (incl. v).
    is_uts_node = nodes["is_uts"].values
    forbidden_arr = np.sort(np.fromiter(forbidden, dtype=np.int64))
    d2_q = np.flatnonzero(dist == 2)
    heads = positives["u_idx"].values[d2_q]
    uniq = np.unique(heads)
    H2 = (Ab[uniq] @ Ab).tocsr()
    H2.sort_indices()
    head_pools = {}
    for i, h in enumerate(uniq):
        cols = H2.indices[H2.indptr[i]:H2.indptr[i + 1]]
        cols = cols[is_uts_node[cols] & (cols != h)]
        keys = _keys(np.full(len(cols), h), cols, n)
        pos_ = np.searchsorted(forbidden_arr, keys)
        hit = (pos_ < len(forbidden_arr)) & (forbidden_arr[np.minimum(pos_, len(forbidden_arr) - 1)] == keys)
        head_pools[int(h)] = cols[~hit]
    rows_h, n_cands = [], []
    for q, h in zip(d2_q, heads):
        pool = head_pools[int(h)]
        k = min(n_rank, len(pool))
        pick = r_rank_hard.choice(pool, size=k, replace=False) if k else np.empty(0, dtype=np.int64)
        n_cands.append(k)
        rows_h.extend((int(q), int(h), int(w)) for w in pick)
    rkh = pd.DataFrame(rows_h, columns=["query_id", "head_idx", "cand_idx"])
    rkh["head"], rkh["cand"] = ids[rkh["head_idx"]], ids[rkh["cand_idx"]]
    n_cands = np.array(n_cands)

    stats.update(
        cutoff=cutoff, window_end=window_end,
        nodes=n, edges=len(g), uts_nodes=int(nodes["is_uts"].sum()),
        nodes_with_edges=int((nodes["degree"] > 0).sum()),
        uts_nodes_with_edges=int((nodes["is_uts"] & (nodes["degree"] > 0)).sum()),
        positives=len(positives), neg_random=len(neg_random), neg_hard=len(neg_hard),
        ranking_rows=len(rk), ranking_k=n_rank,
        rank_hard_queries=int(len(d2_q)), rank_hard_full=int((n_cands == n_rank).sum()),
        rank_hard_partial=int(((n_cands > 0) & (n_cands < n_rank)).sum()),
        rank_hard_no_candidates=int((n_cands == 0).sum()), rank_hard_rows=len(rkh),
        rank_hard_median_candidates=float(np.median(n_cands)) if len(n_cands) else 0.0,
        pos_with_isolated_endpoint=int(isolated.sum()),
        pos_cold_start_any=int((cold_u | cold_v).sum()), pos_cold_start_both=int((cold_u & cold_v).sum()),
        cold_start_max_works=cold_max_works,
        distance=distance_summary(dist),
        # Context for interpreting random negatives (diagnostic only).
        distance_neg_random=distance_summary(positive_distances(A, neg_random)),
    )
    return {"nodes": nodes, "graph_edges": g, "A": A, "positives": positives,
            "neg_random": neg_random, "neg_hard": neg_hard, "ranking": rk[["query_id", "head_idx", "head", "cand_idx", "cand"]],
            "ranking_hard": rkh[["query_id", "head_idx", "head", "cand_idx", "cand"]], "stats": stats, **train_pools}


# ---------------------------------------------------------------- pipeline

def build_splits(cfg: dict, write: bool = True) -> dict[str, dict]:
    processed = Path(cfg["paths"]["processed"])
    out_root = Path(cfg["paths"].get("splits", processed / "splits"))
    t = load_tables(processed)
    uts_pairs = all_uts_coauthor_pairs(t["auth"])
    neg = cfg["negatives"]
    results = {}
    for split_no, name in enumerate(SPLITS):
        s = cfg["splits"][name]
        res = make_split(t, uts_pairs, s["cutoff"], s["window_end"], cfg["seed"], split_no,
                         neg["ratios"], neg["ranking_candidates"], cfg["splits"]["cold_start_max_works"],
                         pool_ratio=neg["train_pool_ratio"] if name == "train" else 0)
        res["stats"]["seed"] = cfg["seed"]
        results[name] = res
        log.info("%s: %s", name, {k: v for k, v in res["stats"].items() if k != "distance"})
        if write:
            d = out_root / name
            d.mkdir(parents=True, exist_ok=True)
            for key in ("nodes", "graph_edges", "positives", "neg_random", "neg_hard", "ranking", "ranking_hard",
                        "train_pool_random", "train_pool_hard"):
                if key in res:
                    res[key].to_parquet(d / f"{key}.parquet", index=False)
            sp.save_npz(d / "adjacency.npz", res["A"])
            (d / "meta.json").write_text(json.dumps(res["stats"], indent=2) + "\n")
    return results


def load_split(root: Path, name: str) -> dict:
    d = Path(root) / name
    out = {k: pd.read_parquet(d / f"{k}.parquet")
           for k in ("nodes", "graph_edges", "positives", "neg_random", "neg_hard", "ranking", "ranking_hard",
                     "train_pool_random", "train_pool_hard") if (d / f"{k}.parquet").exists()}
    out["A"] = sp.load_npz(d / "adjacency.npz").tocsr()
    out["stats"] = json.loads((d / "meta.json").read_text())
    return out


def main(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from linkpred.data import splits_report

    results = build_splits(cfg)
    splits_report.write(cfg, {k: v["stats"] for k, v in results.items()})
