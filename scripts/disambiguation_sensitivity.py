"""Phase 7, step 2: disambiguation sensitivity (planned in docs/decisions.md, 2026-10-14).

  python scripts/disambiguation_sensitivity.py --splits val          # validation first (rule 8)
  python scripts/disambiguation_sensitivity.py --splits val test     # final

Every author ID in data/processed/disambiguation_flags.parquet (same normalised name, more than one ID at the same
institution; 8,961 IDs over all institutions) is removed from BOTH the input graphs and the target pairs, on the
SAME splits and seeds:
  * input graph: every edge touching a flagged ID is dropped (adjacency, graph_edges, node degrees); the
    flagged authors' own pre-cutoff works are dropped from the author tables used by the features;
  * target pairs: every positive, negative, ranking query and candidate with a flagged endpoint is dropped;
  * distance-2 sets are re-checked in the reduced graph: positive distances are recomputed, hard /
    activity-matched negatives, rank_hard(_active) candidates and the hard training pools are kept only if the
    pair still has a common neighbour, and rank_hard(_active) queries only if their positive is still at
    distance 2. Rule 7 holds: pair-focused training uses positives at distance 2 in the REDUCED graph;
  * pair features are recomputed on the reduced graphs with the main pipeline's fixed parameters
    (data/processed/features/params.json: Katz beta, half-life, seed) -- nothing is re-selected.
Models: the pair-focused LightGBM (main selected parameters and mix, 5 seeds), Adamic-Adar, and the prefer_newer
reference. The flag is a heuristic: it also removes correctly identified authors who share a name.

Outputs: results/disambiguation_sensitivity.csv (models suffixed `|no_flagged_ids`),
         results/disambiguation_sensitivity_tests.csv (pair-focused LightGBM - Adamic-Adar, raw p; Holm in
         final_comparison.py), report/disambiguation/removal.csv (what was removed per split).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.splits import load_split, positive_distances  # noqa: E402
from linkpred.eval.protocols import evaluate  # noqa: E402
from linkpred.features.build import canonical, make_context, orient_by_frequency, pair_sets  # noqa: E402
from linkpred.features.embed import load_embeddings  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES, load_feature_tables  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.models import heuristics as H  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from phase7_common import averaged, n_boot_final, raws, score_dict, summary_rows, test_rows  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

log = logging.getLogger("disambiguation")
PF = "lightgbm_pair_focused"
TAG = "|no_flagged_ids"
PROTOS = ("hard_active", "rank_hard_active")
HEAD = (("hard_active", "macro", "roc_auc"), ("rank_hard_active", "macro", "query_auc"))
CELLS = HEAD + (("hard_active", "established_matched", "roc_auc"), ("hard_active", "newcomer_matched", "roc_auc"),
                ("rank_hard_active", "established_matched", "query_auc"),
                ("rank_hard_active", "newcomer_matched", "query_auc"))
D2_TABLES = ("neg_hard", "neg_hard_active", "train_pool_hard", "train_pool_hard_active")
RANK_D2 = ("ranking_hard", "ranking_hard_active")


def flagged_ids(processed: Path) -> set:
    f = pd.read_parquet(processed / "disambiguation_flags.parquet")
    return {i for ids in f["author_ids"] for i in ids}


def common_neighbour(Ab: sp.csr_matrix, u: np.ndarray, v: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    out = np.zeros(len(u), dtype=np.int64)
    for s in range(0, len(u), chunk):
        out[s:s + chunk] = np.asarray(Ab[u[s:s + chunk]].multiply(Ab[v[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def reduce_split(sp_: dict, flagged: set) -> tuple[dict, dict]:
    """The split with flagged IDs removed from the input graph and from every pair set (see module docstring)."""
    nodes = sp_["nodes"].copy()
    assert (nodes["node_idx"].values == np.arange(len(nodes))).all()
    flag = nodes["author_id"].isin(flagged).values
    keep_node = sp.diags((~flag).astype(np.float32))
    A = (keep_node @ sp_["A"] @ keep_node).tocsr()
    A.eliminate_zeros()
    nodes["degree"] = np.diff(A.indptr)
    g = sp_["graph_edges"]
    g = g[~flag[g["u_idx"].values] & ~flag[g["v_idx"].values]].reset_index(drop=True)
    Ab = A.copy()
    Ab.data[:] = 1.0
    out = {"nodes": nodes, "graph_edges": g, "A": A, "stats": dict(sp_["stats"])}
    info = {"flagged_nodes": int(flag.sum()), "flagged_uts_nodes": int((flag & nodes["is_uts"].values).sum()),
            "edges_before": len(sp_["graph_edges"]), "edges_after": len(g)}

    pos = sp_["positives"]
    keep = ~flag[pos["u_idx"].values] & ~flag[pos["v_idx"].values]
    new_pos = pos[keep].reset_index(drop=True)
    new_pos["distance"] = positive_distances(A, new_pos)
    remap = np.full(len(pos), -1)
    remap[np.flatnonzero(keep)] = np.arange(keep.sum())
    out["positives"] = new_pos
    info.update(positives_before=len(pos), positives_after=len(new_pos),
                d2_positives_before=int((pos["distance"] == 2).sum()), d2_positives_after=int((new_pos["distance"] == 2).sum()))

    for t in ("neg_random", "neg_hard", "neg_hard_active", "train_pool_random", "train_pool_hard", "train_pool_hard_active"):
        if t not in sp_:
            continue
        df = sp_[t]
        m = ~flag[df["u_idx"].values] & ~flag[df["v_idx"].values]
        if t in D2_TABLES:
            cn = common_neighbour(Ab, df["u_idx"].values, df["v_idx"].values)
            m &= cn > 0
            df = df.assign(n_common=cn) if "n_common" in df else df
        out[t] = df[m].reset_index(drop=True)
        info[f"{t}_before"], info[f"{t}_after"] = len(df), int(m.sum())

    d2_new = (new_pos["distance"] == 2).values
    for t in ("ranking",) + RANK_D2:
        rk = sp_[t]
        q_new = remap[rk["query_id"].values]
        m = (q_new >= 0) & ~flag[rk["cand_idx"].values]
        if t in RANK_D2:
            m &= np.where(q_new >= 0, d2_new[np.maximum(q_new, 0)], False)
            m &= common_neighbour(Ab, rk["head_idx"].values, rk["cand_idx"].values) > 0
        out[t] = rk[m].assign(query_id=q_new[m]).reset_index(drop=True)
        info[f"{t}_rows_before"], info[f"{t}_rows_after"] = len(rk), int(m.sum())
    return out, info


def reduced_features(cfg, name: str, split: dict, tables: dict, emb, emb_ids, params: dict) -> pd.DataFrame:
    """Pair features for every pair set of the reduced split (same steps as linkpred.features.build)."""
    ctx = make_context(cfg, tables, name, split, emb, emb_ids, params)
    ctx.p = params
    us, vs = [], []
    for table, a, b in pair_sets(name):
        cu, cv = canonical(split[table][a], split[table][b])
        us.append(cu)
        vs.append(cv)
    pairs = pd.DataFrame({"u_idx": np.concatenate(us), "v_idx": np.concatenate(vs)}).drop_duplicates()
    su, sv = orient_by_frequency(pairs["u_idx"].values, pairs["v_idx"].values)
    return pd.concat([pairs.reset_index(drop=True), ctx.compute(su, sv)], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    processed, res = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    n_boot = n_boot_final(cfg)
    flagged = flagged_ids(processed)
    params = json.loads((processed / "features" / "params.json").read_text())
    tables = load_feature_tables(processed)
    tables["auth"] = tables["auth"][~tables["auth"]["author_id"].isin(flagged)]
    emb, emb_ids = load_embeddings(cfg)
    log.info("%d flagged author IDs; feature params %s", len(flagged), params)

    names = ["train"] + args.splits
    root = Path(cfg["paths"]["splits"])
    E = Experiment.__new__(Experiment)
    E.cfg, E.splits, E.feat, E.X = cfg, {}, {}, {}
    E.k_full, E.n_boot, E.seed = cfg["negatives"]["ranking_candidates"], 0, cfg["seed"]
    removal = []
    for name in names + (["val"] if "val" not in names else []):      # validation is needed for early stopping
        if name in E.splits:
            continue
        red, info = reduce_split(load_split(root, name), flagged)
        removal.append({"split": name, **info})
        E.splits[name] = red
        E.feat[name] = reduced_features(cfg, name, red, tables, emb, emb_ids, params)
        if name != "train":
            E.X[name] = FM.eval_features(red, E.feat[name], ALL_FEATURES)
        log.info("%s reduced: %s", name, info)
    rep = Path(cfg["paths"]["report"]) / "disambiguation"
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(removal).to_csv(rep / "removal.csv", index=False)

    pfp = search_best(cfg, PF)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    models = {s: [E.fit("lightgbm", ALL_FEATURES, lmix, s, pfp, active=True, d2_only=True)] for s in cfg["seeds"]}
    rows, tests = [], []
    for split in args.splits:
        sp_ = E.splits[split]
        sc = {PF: [FM.score_sets(models[s][0], E.Xsets(split, ALL_FEATURES)) for s in cfg["seeds"]],
              "adamic_adar": [{k: v["adamic_adar"].values for k, v in E.X[split].items()}],
              "prefer_newer": [score_dict(sp_, lambda u, v: H.prefer_newer(sp_, u, v))]}
        for i, (m, sl) in enumerate(sc.items()):
            rows += summary_rows(m + TAG, split, raws(sp_, split, m, sl, PROTOS, cfg), CELLS, cfg, n_boot, 61 + i)
        for p, sub, metric in (("hard_active", "established_matched", "n_positives"),
                               ("hard_active", "established_matched", "n_negatives"),
                               ("hard_active", "newcomer_matched", "n_positives"),
                               ("hard_active", "newcomer_matched", "n_negatives"),
                               ("rank_hard_active", "established_matched", "n_queries"),
                               ("rank_hard_active", "newcomer_matched", "n_queries")):
            r = evaluate(None, sp_, split, "adamic_adar" + TAG, cfg["seed"], cfg["seed"], 0, protocols=(p,),
                         k_full=E.k_full, precomputed=sc["adamic_adar"][0])[0]
            rows += [x for x in r if x["subset"] == sub and x["metric"] == metric]
        A = averaged(sp_, split, PF, sc[PF], PROTOS, cfg)
        B = averaged(sp_, split, "adamic_adar", sc["adamic_adar"], PROTOS, cfg)
        tests += test_rows(f"{PF}{TAG} - adamic_adar{TAG}", split, A, B, HEAD, cfg, n_boot, 71)
        for r in rows + tests:
            if r["split"] == split and r["metric"] in ("roc_auc", "query_auc", "roc_auc_diff", "query_auc_diff",
                                                       "roc_auc_diff_p_value", "query_auc_diff_p_value") and r["subset"] == "macro":
                print(split, r["model"], r["protocol"], r["metric"], round(r["value"], 4), flush=True)
    write_results(rows, res / "disambiguation_sensitivity.csv", append=False)
    write_results(tests, res / "disambiguation_sensitivity_tests.csv", append=False)


if __name__ == "__main__":
    main()
