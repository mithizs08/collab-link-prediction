"""Phase 5: node2vec on each split's input graph.

  python scripts/run_node2vec.py --config configs/default.yaml

Steps (selection on VALIDATION only; test scored at the end):
 1. grid (seed 0): p, q in {0.5, 1, 2}; embeddings of the train and val graphs; val aligned to train
    (orthogonal Procrustes, see linkpred.models.node2vec). For each edge operator and each Phase 4
    negative mix, logistic regression on train pairs; the mix per (p, q, operator) by the Phase 4
    rule (best val `hard` ROC-AUC among mixes with val rank_random MRR >= 90% of the best); then
    (p, q, operator) by val `hard_active` ROC-AUC.
 2. final: 5 seeds; embeddings of train/val/test graphs (val/test aligned to train, same seed);
    `node2vec` (LR on the chosen operator) and `node2vec_cosine` (alignment-free) under all protocols.
 3. node2vec as extra features to the best LightGBM (chosen on val hard_active between the
    general and pair-focused models): operator vector + raw cosine, same settings, 5 seeds.
 4. paired tests on test: vs Adamic-Adar / val-best heuristic, vs LightGBM; LightGBM+node2vec vs LightGBM.

Outputs: results/node2vec_grid.csv, results/node2vec.csv, results/node2vec_paired.csv, results/node2vec.md
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.protocols import PROTOCOL_ORDER, PairScores, evaluate  # noqa: E402
from linkpred.features.build import features_for  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.models import node2vec as N2V  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from run_feature_models import HEURISTICS, Experiment, paired, select_mix  # noqa: E402

log = logging.getLogger("node2vec")
GRID_PROTOCOLS = ("hard_active", "hard", "rank_random", "rank_hard_active")
PAIR_PROTOCOLS = ("hard_active", "rank_hard_active", "hard", "rank_hard")
NEG_TYPE = {"random": "random", "rank_random": "random", "hard_active": "hard_active", "rank_hard_active": "hard_active"}


def pair_sets(split: dict):
    for name, (table, a, b) in PairScores.SOURCES.items():
        yield name, split[table][a].values, split[table][b].values


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


class N2VExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.E = Experiment(cfg)
        self.splits = self.E.splits
        self.ncfg = cfg["models"]["node2vec"]
        self.cache = Path(cfg["paths"]["interim"]) / "node2vec"

    def embeddings(self, p, q, seed, names=("train", "val", "test")):
        raw = {s: N2V.cached_embeddings(self.cache, s, self.splits[s]["A"], p, q, seed, self.ncfg) for s in names}
        aligned = {"train": raw["train"]}
        for s in names:
            if s != "train":
                aligned[s] = N2V.align_to(raw[s], self.splits[s]["nodes"], raw["train"], self.splits["train"]["nodes"])
        return raw, aligned

    def fit_lr(self, emb_train, op, mix):
        pairs, y = FM.training_pairs(self.splits["train"], mix[0], mix[1])
        X = N2V.edge_features(emb_train, pairs["u_idx"].values, pairs["v_idx"].values, op)
        return lr_model().fit(X, y)

    def lr_scores(self, model, emb, split, op):
        return {n: model.predict_proba(N2V.edge_features(emb, u, v, op))[:, 1] for n, u, v in pair_sets(self.splits[split])}

    def cos_scores(self, emb, split):
        return {n: N2V.cosine(emb, u, v) for n, u, v in pair_sets(self.splits[split])}

    def evaluate(self, scores, split, name, seed, n_boot, protocols=PROTOCOL_ORDER):
        return evaluate(None, self.splits[split], split, name, seed, self.cfg["seed"], n_boot, protocols=protocols,
                        k_full=self.cfg["negatives"]["ranking_candidates"], precomputed=scores)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.n_boot is not None:
        cfg["evaluation"]["n_bootstrap"] = args.n_boot
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    X = N2VExperiment(cfg)
    n_boot = cfg["evaluation"]["n_bootstrap"]
    t0 = time.time()

    # 1. grid -----------------------------------------------------------------------------
    mixes = list(itertools.product(cfg["models"]["training_mix"]["hard_fractions"], cfg["models"]["training_mix"]["ratios"]))
    grid_rows, choice = [], []
    for p, q in itertools.product(X.ncfg["p_grid"], X.ncfg["q_grid"]):
        raw, al = X.embeddings(p, q, cfg["seed"], ("train", "val"))
        r, _ = X.evaluate(X.cos_scores(raw["val"], "val"), "val", f"node2vec_cosine|p={p:g}|q={q:g}", cfg["seed"], 0, GRID_PROTOCOLS)
        grid_rows += r
        for op in N2V.OPERATORS:
            kind = f"node2vec|p={p:g}|q={q:g}|op={op}"
            for frac, ratio in mixes:
                m = X.fit_lr(al["train"], op, (frac, ratio))
                r, _ = X.evaluate(X.lr_scores(m, al["val"], "val", op), "val", f"{kind}|hard={frac:g}|1:{ratio}",
                                  cfg["seed"], 0, GRID_PROTOCOLS)
                grid_rows += r
            mix, _ = select_mix(grid_rows, kind)
            name = f"{kind}|hard={mix[0]:g}|1:{mix[1]}"
            ha = next(x["value"] for x in grid_rows if x["model"] == name and x["protocol"] == "hard_active"
                      and x["subset"] == "all" and x["metric"] == "roc_auc")
            choice.append({"p": p, "q": q, "op": op, "mix": mix, "hard_active": ha})
        log.info("grid p=%g q=%g done (%.0fs)", p, q, time.time() - t0)
    write_results(grid_rows, res / "node2vec_grid.csv", append=False)
    ch = pd.DataFrame(choice)
    ch["v"] = ch["hard_active"].round(6)
    best = ch.sort_values(["v", "p", "q"], ascending=[False, True, True], kind="stable").iloc[0]
    P, Q, OP, MIX = float(best["p"]), float(best["q"]), best["op"], tuple(best["mix"])
    log.info("chosen p=%g q=%g op=%s mix=%s (val hard_active %.4f)", P, Q, OP, MIX, best["hard_active"])

    # 2. final node2vec models + 3. LightGBM + node2vec ------------------------------------------
    fm = pd.read_csv(res / "feature_models.csv")
    va = fm[(fm.split == "val") & (fm.protocol == "hard_active") & (fm.subset == "all") & (fm.metric == "roc_auc")]
    va = va[va.model.isin(["lightgbm", "lightgbm_pair_focused"])].groupby("model")["value"].mean()
    base_lgbm = va.round(6).sort_values(ascending=False, kind="stable").index[0]
    active = base_lgbm == "lightgbm_pair_focused"
    lparams = search_best(cfg, base_lgbm)
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    log.info("base LightGBM for +node2vec: %s %s", base_lgbm, lparams)

    rows, raw_scores = [], {}
    feat = X.E.feat
    n2v_cols = [f"n2v_{OP}_{i}" for i in range(X.ncfg["embedding_dim"])] + ["n2v_cosine"]

    def lgbm_frame(split, emb_al, emb_raw, u, v):
        base = features_for(feat[split], u, v, ALL_FEATURES)
        ef = pd.DataFrame(N2V.edge_features(emb_al, u, v, OP), columns=n2v_cols[:-1])
        ef["n2v_cosine"] = N2V.cosine(emb_raw, u, v)
        return pd.concat([base, ef], axis=1)

    for s in seeds:
        raw, al = X.embeddings(P, Q, s)
        lr = X.fit_lr(al["train"], OP, MIX)
        # LightGBM base (refit, identical to Phase 4 settings) and with node2vec features.
        pairs, y = FM.training_pairs(X.splits["train"], lmix[0], lmix[1], active=active)
        u_tr, v_tr = pairs["u_idx"].values, pairs["v_idx"].values
        neg_es = "neg_hard_active" if active else "neg_hard"
        d2 = (X.splits["val"]["positives"]["distance"] == 2).values
        vpos, vneg = X.splits["val"]["positives"], X.splits["val"][neg_es]
        uv = np.r_[vpos["u_idx"].values[d2], vneg["u_idx"].values]
        vv = np.r_[vpos["v_idx"].values[d2], vneg["v_idx"].values]
        yv = np.r_[np.ones(d2.sum()), np.zeros(len(vneg))]
        Xtr_n = lgbm_frame("train", al["train"], raw["train"], u_tr, v_tr)
        Xv_n = lgbm_frame("val", al["val"], raw["val"], uv, vv)
        m_base = FM.fit_lightgbm(Xtr_n[ALL_FEATURES], y, Xv_n[ALL_FEATURES], yv, lparams, s)
        m_n2v = FM.fit_lightgbm(Xtr_n, y, Xv_n, yv, lparams, s)
        for split in ("val", "test"):
            sets = {"node2vec": X.lr_scores(lr, al[split], split, OP), "node2vec_cosine": X.cos_scores(raw[split], split)}
            lb, ln = {}, {}
            for n, u, v in pair_sets(X.splits[split]):
                F = lgbm_frame(split, al[split], raw[split], u, v)
                lb[n] = FM.predict(m_base, F[ALL_FEATURES])
                ln[n] = FM.predict(m_n2v, F)
            sets[base_lgbm] = lb
            sets[f"{base_lgbm}+node2vec"] = ln
            for name, sc in sets.items():
                r, x = X.evaluate(sc, split, name, s, n_boot)
                if name != base_lgbm:          # base LightGBM rows already live in feature_models.csv
                    rows += r
                raw_scores[(name, s, split)] = x
        log.info("final seed %d (%.0fs)", s, time.time() - t0)
    write_results(rows, res / "node2vec.csv", append=False)

    # 4. paired tests ---------------------------------------------------------------------------
    base = pd.read_csv(res / "baselines.csv")
    val_best = {}
    for p in PROTOCOL_ORDER:
        subset, metric = report.HEADLINE[p]
        d = base[(base.split == "val") & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)]
        val_best[p] = d.sort_values(["value", "model"], ascending=[False, True])["model"].iloc[0]
    href = X.E.heuristic_raw("test")
    prow = []
    main_protocols = ("hard_active", "rank_hard_active", "hard", "rank_hard", "rank_random", "random")
    for s in seeds:
        for p in main_protocols:
            nt = NEG_TYPE.get(p, "hard")
            for name in ("node2vec", "node2vec_cosine", f"{base_lgbm}+node2vec"):
                refs = {val_best[p]} | ({"adamic_adar"} if p in PAIR_PROTOCOLS else set())
                for h in sorted(refs):
                    prow += paired(cfg, raw_scores[(name, s, "test")], href[h][1], p, name, h, s, nt)
            prow += paired(cfg, raw_scores[("node2vec", s, "test")], raw_scores[(base_lgbm, s, "test")], p,
                           "node2vec", base_lgbm, s, nt)
            prow += paired(cfg, raw_scores[(f"{base_lgbm}+node2vec", s, "test")], raw_scores[(base_lgbm, s, "test")], p,
                           f"{base_lgbm}+node2vec", base_lgbm, s, nt)
    write_results(prow, res / "node2vec_paired.csv", append=False)
    write_report(cfg, P, Q, OP, MIX, base_lgbm, lparams, ch)
    log.info("done in %.0fs", time.time() - t0)


def write_report(cfg, P, Q, OP, MIX, base_lgbm, lparams, choice: pd.DataFrame):
    from linkpred.eval.feature_report import msd, seed_summary

    res = Path(cfg["paths"]["results"])
    nv = pd.read_csv(res / "node2vec.csv")
    fm = pd.read_csv(res / "feature_models.csv")
    base = pd.read_csv(res / "baselines.csv")
    grid = pd.read_csv(res / "node2vec_grid.csv")
    pr = pd.read_csv(res / "node2vec_paired.csv")
    s = seed_summary(pd.concat([nv, fm[fm.model == base_lgbm]]))
    models = ["node2vec", "node2vec_cosine", base_lgbm, f"{base_lgbm}+node2vec"]
    ks_h = [f"hits@{k}" for k in cfg["evaluation"]["rank_ks_hard"]]
    ks_r = [f"hits@{k}" for k in cfg["evaluation"]["rank_ks_random"]]
    n = cfg["models"]["node2vec"]
    out = ["# node2vec", "",
           "_Generated by `scripts/run_node2vec.py` from `results/node2vec.csv`, `node2vec_grid.csv`, "
           "`node2vec_paired.csv` (and `feature_models.csv` / `baselines.csv` for reference rows). Learned models: "
           "mean ± sd over 5 seeds; heuristics: point estimate [bootstrap 95% CI]._", "",
           "## Setup", "",
           f"- node2vec (torch_geometric + torch-cluster) on each split's input graph only: dimension {n['embedding_dim']}, "
           f"walk length {n['walk_length']}, context {n['context_size']}, {n['walks_per_node']} walks per node, "
           f"{n['epochs']} epoch(s), batch {n['batch_size']}, lr {n['lr']}, 1 negative sample.",
           "- Val/test embeddings are rotated into the train embedding's space (orthogonal Procrustes on authors with "
           "co-authors in both graphs) so a classifier fitted on train pairs applies. `node2vec_cosine` uses raw "
           "within-split cosine similarity and needs no alignment.",
           f"- **Chosen on validation:** p = {P:g}, q = {Q:g}, operator **{OP}**, negative mix hard fraction {MIX[0]:g}, 1:{MIX[1]} "
           "(mix by the Phase 4 rule; p, q, operator by val hard_active ROC-AUC).",
           f"- LightGBM + node2vec: base model `{base_lgbm}` (better on val hard_active), same settings "
           f"(num_leaves {lparams['num_leaves']}, lr {lparams['learning_rate']:g}, min_child {lparams['min_child_samples']}), "
           f"plus the {n['embedding_dim']} {OP} dimensions and the raw cosine.", ""]
    out += ["## Validation grid (seed 0): val hard_active ROC-AUC at the rule-chosen mix", ""]
    piv = choice.assign(cell=lambda d: d["hard_active"].map(lambda v: f"{v:.4f}"))
    piv = piv.pivot_table(index=["p", "q"], columns="op", values="cell", aggfunc="first").reset_index()
    cos = grid[(grid.model.str.startswith("node2vec_cosine")) & (grid.protocol == "hard_active") & (grid.subset == "all")
               & (grid.metric == "roc_auc")].copy()
    cos["p"] = cos.model.str.extract(r"p=([\d.]+)")[0].astype(float)
    cos["q"] = cos.model.str.extract(r"q=([\d.]+)")[0].astype(float)
    piv = piv.merge(cos[["p", "q", "value"]].rename(columns={"value": "cosine"}), on=["p", "q"])
    piv["cosine"] = piv["cosine"].map(lambda v: f"{v:.4f}")
    out += [report.to_md(piv), ""]

    def sec(split, p, sub, metrics, refs):
        d = s[(s.split == split) & (s.protocol == p) & (s.subset == sub)]
        t = pd.DataFrame({"model": models})
        for m in metrics:
            x = d[d.metric == m].set_index("model")
            t[m] = [msd(x.loc[k]) if k in x.index else "" for k in models]
        h = report.metric_table(base, p, sub, split, metrics, refs)
        h["model"] = h["model"] + " (heuristic)"
        return report.to_md(pd.concat([t, h]))

    for split in ("test", "val"):
        out += [f"## {split}", ""]
        for p, sub, metrics in (("hard_active", "all", ["roc_auc", "average_precision"]),
                                ("rank_hard_active", "full_candidates", ks_h + ["mrr", "query_auc"]),
                                ("hard", "all", ["roc_auc", "average_precision"]),
                                ("rank_hard", "full_candidates", ks_h + ["mrr", "query_auc"]),
                                ("rank_random", "all", ks_r + ["mrr"]),
                                ("random", "all", ["roc_auc", "average_precision"])):
            tag = {"hard_active": " (lead pair-level metric)", "hard": " (includes activity)"}.get(p, "")
            out += [f"### {p} — {sub}{tag}", "", sec(split, p, sub, metrics, ["adamic_adar"]), ""]
        out += [f"### Cold start vs established ({split})", ""]
        rows = []
        for p, metric in (("hard_active", "roc_auc"), ("rank_hard_active", "query_auc"), ("hard", "roc_auc")):
            for sub in ("cold_start", "established"):
                d = s[(s.split == split) & (s.protocol == p) & (s.subset == sub) & (s.metric == metric)].set_index("model")
                b = base[(base.split == split) & (base.protocol == p) & (base.subset == sub) & (base.metric == metric)
                         & (base.model == "adamic_adar")]
                row = {"protocol / subset": f"{p} {metric} — {sub}"}
                row.update({m: msd(d.loc[m]) for m in models if m in d.index})
                row["adamic_adar"] = f"{b['value'].iloc[0]:.3f}" if len(b) else ""
                rows.append(row)
        out += [report.to_md(pd.DataFrame(rows)), ""]
    out += ["## Paired bootstrap tests (test, per seed)", ""]
    pv = pr[pr.metric.str.endswith("_p_value")].rename(columns={"value": "p"})
    dv = pr[~pr.metric.str.endswith("_p_value")]
    m = dv.merge(pv[["model", "protocol", "seed", "p"]], on=["model", "protocol", "seed"])
    rows = []
    for (model, p), d in m.groupby(["model", "protocol"], sort=False):
        rows.append({"comparison": model, "protocol": p, "metric": d["metric"].iloc[0],
                     "mean diff": f"{d['value'].mean():+.4f}",
                     "mean 95% CI": f"[{d['ci_low'].mean():+.4f}, {d['ci_high'].mean():+.4f}]",
                     "sig. better": f"{int(((d.p < 0.05) & (d.value > 0)).sum())}/{len(d)}",
                     "sig. worse": f"{int(((d.p < 0.05) & (d.value < 0)).sum())}/{len(d)}",
                     "median p": f"{d['p'].median():.3f}"})
    out += [report.to_md(pd.DataFrame(rows)), ""]
    out += ["## Appendix: diagnostic_all_vs_hard", "", report.DIAGNOSTIC_WARNING, "",
            sec("test", "diagnostic_all_vs_hard", "all", ["roc_auc", "average_precision"], ["adamic_adar"]), ""]
    (res / "node2vec.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
