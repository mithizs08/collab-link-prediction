"""Final-pass checks (reviewer points; docs/reviewer_pass.md). Validation first (rule 8), then test.

  python scripts/final_pass_checks.py --splits val
  python scripts/final_pass_checks.py --splits val test

All EXPLORATORY: raw p-values only, outside the Holm family of results/final_comparison_tests.csv.

1. Pair-focused logistic regression: logistic regression trained on exactly the pair-focused LightGBM's training pairs
   (train-split distance-2 positives vs 2023 activity-matched distance-2 negatives, 1:3), 5 seeds, all protocols.
   Paired tests on seed-averaged scores: pair-focused LR - Adamic-Adar, pair-focused LightGBM - pair-focused LR,
   pair-focused LR - general LR. Question: does the gain come from the training pairs or from the model?
2. Gain against the validation-best heuristic per headline protocol (results/baselines.csv via report.val_best:
   Adamic-Adar on hard_active, Jaccard on rank_hard_active): paired test, same procedure as the primary tests.
3. node2vec sanity: Spearman correlation of the node2vec cosine (seed 0, reported p, q) with Adamic-Adar and with
   common neighbours on the hard_active pairs; ROC-AUC of cosine on random pairs. A working embedding should track
   shared-neighbourhood structure.
4. Author-clustered bootstrap for the headline comparisons (pair-focused LightGBM - Adamic-Adar, validation-best
   heuristic - Adamic-Adar, pair-focused LightGBM - validation-best heuristic), macro hard_active ROC-AUC and macro
   rank_hard_active query AUC, seed-averaged scores. Pairs share authors, so pairs are not independent. Each resample
   draws a Poisson(1) weight per AUTHOR; a pair's weight is the product of its two authors' weights (a dyadic
   bootstrap); for ranking, a query's weight is the product of its positive pair's two authors' weights (candidates
   inside a query are not reweighted). The same estimator with Poisson(1) weights per PAIR / QUERY gives the
   independent-pairs reference, so CI widths are compared like for like. 5,000 resamples.

Outputs: results/pf_logreg_exploratory.csv, results/final_pass_tests.csv (exploratory tests),
         results/node2vec_sanity.csv, results/clustered_bootstrap.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.metrics import query_auc, rank_counts, roc_auc, snap  # noqa: E402
from linkpred.eval.protocols import newcomer_pairs  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.models import node2vec as N2V  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from final_comparison import node2vec_choice  # noqa: E402
from phase7_common import averaged, n_boot_final, raws, summary_rows, test_rows  # noqa: E402
from phase8_shap import other_split  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

log = logging.getLogger("final_pass")
PF, PFLR, AA = "lightgbm_pair_focused", "logreg_pair_focused", "adamic_adar"
ALL_PROTOS = ("hard", "rank_hard", "random", "hard_active", "rank_hard_active")
CELLS = (("hard_active", "macro", "roc_auc"), ("rank_hard_active", "macro", "query_auc"),
         ("hard_active", "established_matched", "roc_auc"), ("hard_active", "newcomer_matched", "roc_auc"),
         ("rank_hard_active", "established_matched", "query_auc"), ("rank_hard_active", "newcomer_matched", "query_auc"),
         ("hard", "all", "roc_auc"), ("hard", "macro", "roc_auc"), ("rank_hard", "macro", "query_auc"),
         ("random", "all", "roc_auc"))
HEAD = (("hard_active", "macro", "roc_auc"), ("rank_hard_active", "macro", "query_auc"))


# ------------------------------------------------------------------------------------- weighted metrics
def wauc(sp, wp, sn, wn) -> float:
    """Weighted ROC-AUC (ties count half), scores snapped as in linkpred.eval.metrics."""
    s = np.r_[snap(sp), snap(sn)]
    lev, inv = np.unique(s, return_inverse=True)
    negw = np.bincount(inv[len(sp):], weights=wn, minlength=len(lev))
    below = np.r_[0.0, np.cumsum(negw)[:-1]]
    ip = inv[:len(sp)]
    num = np.sum(wp * (below[ip] + 0.5 * negw[ip]))
    den = wp.sum() * wn.sum()
    return num / den if den > 0 else np.nan


class Headline:
    """Group-matched pairs / queries of one split with author ids, for weighted resampling."""

    def __init__(self, sp_):
        pos = sp_["positives"]
        d2 = (pos["distance"] == 2).values
        neg = sp_["neg_hard_active"]
        self.pu, self.pv = pos["u_idx"].values[d2], pos["v_idx"].values[d2]
        self.nu, self.nv = neg["u_idx"].values, neg["v_idx"].values
        self.pg = newcomer_pairs(sp_, self.pu, self.pv)
        self.ng = newcomer_pairs(sp_, self.nu, self.nv)
        self.d2 = d2
        rk = sp_["ranking_hard_active"]
        qn = newcomer_pairs(sp_, pos["u_idx"].values, pos["v_idx"].values)
        cn = newcomer_pairs(sp_, rk["head_idx"].values, rk["cand_idx"].values)
        q = rk["query_id"].values
        self.keep = cn == qn[q]                                    # candidates in the query's own group
        self.q, self.nq = q, len(pos)
        self.qn = qn
        self.qu, self.qv = pos["u_idx"].values, pos["v_idx"].values
        self.n_nodes = len(sp_["nodes"])

    def per_query(self, s_pos, s_rank):
        g, e, n = rank_counts(s_pos, s_rank[self.keep], self.q[self.keep], self.nq)
        has = n > 0
        return np.where(has, query_auc(g, e, np.maximum(n, 1)), np.nan), has

    def stats(self, sc, w_author=None, w_pair=None, w_query=None):
        """(macro hard_active ROC-AUC, macro rank_hard_active query AUC) under the given weights."""
        sp, sn = sc["positives"][self.d2], sc["neg_hard_active"]
        if w_author is not None:
            wp, wn = w_author[self.pu] * w_author[self.pv], w_author[self.nu] * w_author[self.nv]
            wq = w_author[self.qu] * w_author[self.qv]
        else:
            wp = w_pair[0] if w_pair is not None else np.ones(len(sp))
            wn = w_pair[1] if w_pair is not None else np.ones(len(sn))
            wq = w_query if w_query is not None else np.ones(self.nq)
        cls = np.mean([wauc(sp[self.pg == g], wp[self.pg == g], sn[self.ng == g], wn[self.ng == g]) for g in (False, True)])
        qa, has = sc["_qauc"]
        rk = []
        for g in (False, True):
            m = has & (self.qn == g)
            rk.append(np.sum(qa[m] * wq[m]) / wq[m].sum())
        return np.array([cls, np.mean(rk)])


def clustered(H: Headline, scores: dict, pairs, rng_seed, n_boot):
    """Paired author-clustered and independent-pair bootstraps of the differences."""
    for sc in scores.values():
        sc["_qauc"] = H.per_query(sc["positives"], sc["ranking_hard_active"])
    point = {m: H.stats(sc) for m, sc in scores.items()}
    rng = np.random.default_rng(rng_seed)
    out = {kind: {p: [] for p in pairs} for kind in ("author_clustered", "independent_pairs")}
    n_p, n_n = int(H.d2.sum()), len(H.nu)
    for _ in range(n_boot):
        wa = rng.poisson(1.0, H.n_nodes).astype(float)
        wpp = (rng.poisson(1.0, n_p).astype(float), rng.poisson(1.0, n_n).astype(float))
        wq = rng.poisson(1.0, H.nq).astype(float)
        sa = {m: H.stats(sc, w_author=wa) for m, sc in scores.items()}
        si = {m: H.stats(sc, w_pair=wpp, w_query=wq) for m, sc in scores.items()}
        for a, b in pairs:
            out["author_clustered"][(a, b)].append(sa[a] - sa[b])
            out["independent_pairs"][(a, b)].append(si[a] - si[b])
    return point, out


def row(model, split, protocol, subset, metric, value, lo=None, hi=None, negative_type=None):
    return {"model": model, "split": split, "protocol": protocol, "subset": subset,
            "negative_type": negative_type or ("hard_active" if protocol.endswith("active") else
                                               ("random" if protocol == "random" else "hard")),
            "seed": -1, "metric": metric, "value": float(value),
            "ci_low": None if lo is None else float(lo), "ci_high": None if hi is None else float(hi)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    n_boot = n_boot_final(cfg)
    E = Experiment(cfg)
    grid = pd.read_csv(res / "negative_mix_grid.csv").to_dict("records")
    lmix, lrmix = select_mix(grid, "lightgbm")[0], select_mix(grid, "logreg")[0]
    pfp = search_best(cfg, PF)
    vb = report.val_best(pd.read_csv(res / "baselines.csv"), ("hard_active", "rank_hard_active"))
    log.info("validation-best heuristics: %s", vb)

    sc = {m: {s: [] for s in args.splits} for m in (PFLR, PF, "logreg")}
    for seed in cfg["seeds"]:
        # Same training pairs as the pair-focused LightGBM: active=True, d2_only=True, the same mix (ratio 1:3).
        mlr = E.fit("logreg", ALL_FEATURES, lmix, seed, active=True, d2_only=True)
        mpf = E.fit("lightgbm", ALL_FEATURES, lmix, seed, pfp, active=True, d2_only=True)
        for split in args.splits:
            Xs = E.Xsets(split, ALL_FEATURES)
            sc[PFLR][split].append(FM.score_sets(mlr, Xs))
            sc[PF][split].append(FM.score_sets(mpf, Xs))
    glr = E.fit("logreg", ALL_FEATURES, lrmix, cfg["seed"])          # general logistic regression (deterministic)
    for split in args.splits:
        sc["logreg"][split] = [FM.score_sets(glr, E.Xsets(split, ALL_FEATURES))]

    P, Q, _, _ = node2vec_choice(res, cfg["models"]["node2vec"])
    cache = Path(cfg["paths"]["interim"]) / "node2vec"
    rows, tests, n2v, clus = [], [], [], []
    for split in args.splits:
        sp_ = E.splits[split]
        heur = {h: [{k: v[h].values for k, v in E.X[split].items()}] for h in ("adamic_adar", "jaccard", "common_neighbors")}
        R = raws(sp_, split, PFLR, sc[PFLR][split], ALL_PROTOS, cfg)
        rows += summary_rows(PFLR, split, R, CELLS, cfg, n_boot, 141)
        A = {m: averaged(sp_, split, m, sc[m][split], ALL_PROTOS, cfg) for m in (PFLR, PF, "logreg")}
        A.update({h: averaged(sp_, split, h, heur[h], ALL_PROTOS, cfg) for h in ("adamic_adar", "jaccard")})
        for k, (a, b) in enumerate(((PFLR, AA), (PF, PFLR), (PFLR, "logreg"))):
            tests += test_rows(f"{a} - {b}", split, A[a], A[b], HEAD, cfg, n_boot, 151 + k)
        # gain against the validation-best heuristic of each protocol (AA on hard_active, Jaccard on rank_hard_active)
        for j, (p, sub, metric) in enumerate(HEAD):
            best = vb[p]
            tests += test_rows(f"{PF} - {best} (validation-best heuristic for {p})", split, A[PF], A[best],
                               ((p, sub, metric),), cfg, n_boot, 161 + j)
            if best != AA:
                tests += test_rows(f"{best} - {AA}", split, A[best], A[AA], ((p, sub, metric),), cfg, n_boot, 171 + j)

        # node2vec sanity (seed 0 embeddings of the reported p, q; raw within-split cosine, no alignment)
        emb = N2V.cached_embeddings(cache, split, sp_["A"], P, Q, cfg["seed"], cfg["models"]["node2vec"])
        d2 = (sp_["positives"]["distance"] == 2).values
        u = np.r_[sp_["positives"]["u_idx"].values[d2], sp_["neg_hard_active"]["u_idx"].values]
        v = np.r_[sp_["positives"]["v_idx"].values[d2], sp_["neg_hard_active"]["v_idx"].values]
        X = pd.concat([E.X[split]["positives"][d2], E.X[split]["neg_hard_active"]], ignore_index=True)
        cos = N2V.cosine(emb, u, v)
        for f in ("adamic_adar", "common_neighbors"):
            n2v.append(row("node2vec_cosine", split, "hard_active", "all", f"spearman_with_{f}",
                           spearmanr(cos, X[f].values).statistic))
        pr, nr = sp_["positives"], sp_["neg_random"]
        cp = N2V.cosine(emb, pr["u_idx"].values, pr["v_idx"].values)
        cn = N2V.cosine(emb, nr["u_idx"].values, nr["v_idx"].values)
        n2v.append(row("node2vec_cosine", split, "random", "all", "roc_auc_seed0", roc_auc(cp, cn)))

        # author-clustered bootstrap
        H = Headline(sp_)
        best_rk = vb["rank_hard_active"]
        scores = {PF: {k: np.mean([s[k] for s in sc[PF][split]], axis=0) for k in sc[PF][split][0]},
                  AA: heur["adamic_adar"][0], best_rk: heur[best_rk][0]}
        comps = [(PF, AA), (best_rk, AA), (PF, best_rk)]
        point, boot = clustered(H, scores, comps, [cfg["seed"], 181, 0 if split == "val" else 1], n_boot)
        for kind, d in boot.items():
            for (a, b), diffs in d.items():
                diffs = np.array(diffs)
                for j, (p, sub, metric) in enumerate(HEAD):
                    x = diffs[:, j]
                    lo, hi = np.nanpercentile(x, [2.5, 97.5])
                    pval = min(1.0, max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1.0 / len(x)))
                    base = dict(protocol=p, subset=sub)
                    name = f"{a} - {b}"
                    clus += [row(name, split, p, sub, f"{metric}_diff|{kind}", point[a][j] - point[b][j], lo, hi),
                             row(name, split, p, sub, f"{metric}_diff_p_value|{kind}", pval),
                             row(name, split, p, sub, f"{metric}_diff_ci_width|{kind}", hi - lo)]
        for (a, b) in comps:
            for j, (p, sub, metric) in enumerate(HEAD):
                w = {k: next(r["value"] for r in clus if r["model"] == f"{a} - {b}" and r["split"] == split and
                             r["protocol"] == p and r["metric"] == f"{metric}_diff_ci_width|{k}")
                     for k in ("author_clustered", "independent_pairs")}
                clus.append(row(f"{a} - {b}", split, p, sub, f"{metric}_ci_width_ratio_clustered_over_independent",
                                w["author_clustered"] / w["independent_pairs"]))
        for r in rows + tests + n2v + clus:
            if r["split"] == split and not r["metric"].endswith("_sd") and "ci_width|" not in r["metric"]:
                print(split, r["model"], r["protocol"], r["subset"], r["metric"], round(r["value"], 4),
                      "" if r["ci_low"] is None else f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", flush=True)
    for name, data in (("pf_logreg_exploratory.csv", rows), ("final_pass_tests.csv", tests),
                       ("node2vec_sanity.csv", n2v), ("clustered_bootstrap.csv", clus)):
        keep = other_split(res / name, args.splits[0]) if len(args.splits) == 1 else []
        write_results(keep + data, res / name, append=False)


if __name__ == "__main__":
    main()
