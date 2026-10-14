"""Phase 6, local part (after scripts/import_colab_results.py): paired tests, hybrid, report.

  python scripts/gnn_analysis.py [--scores-dir data/interim/gnn] [--results-dir results] [--n-boot N]

- Paired tests on test, per seed: every final GNN vs Adamic-Adar, vs the validation-best heuristic
  per protocol, and vs the pair-focused LightGBM (refitted here; identical to Phase 4/5).
- Hybrid: pair-focused LightGBM (same features/settings) + the best GNN's score as one extra
  feature. Training-pair GNN scores are OUT-OF-FOLD (GNNs trained on the train split without
  those pairs); val/test GNN scores come from the final run with the same seed.
Writes results/gnn_paired.csv, results/gnn_hybrid.csv, results/gnn.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.feature_report import msd, seed_summary  # noqa: E402
from linkpred.eval.protocols import PROTOCOL_ORDER, PairScores, evaluate  # noqa: E402
from linkpred.features.build import features_for  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from run_feature_models import Experiment, paired, select_mix  # noqa: E402

MAIN = ("hard_active", "rank_hard_active", "hard", "rank_hard", "rank_random", "random")
PAIR = ("hard_active", "rank_hard_active", "hard", "rank_hard")
NEG = {"random": "random", "rank_random": "random", "hard_active": "hard_active", "rank_hard_active": "hard_active"}
PF = "lightgbm_pair_focused"


def load_scores(d: Path, split: str) -> dict:
    z = np.load(d / f"scores_{split}.npz")
    return {k: z[k].astype(np.float64) for k in PairScores.SOURCES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scores-dir", type=Path, default=Path("data/interim/gnn"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.n_boot is not None:
        cfg["evaluation"]["n_bootstrap"] = args.n_boot
    set_seed(cfg["seed"])
    res, sd = args.results_dir, args.scores_dir
    E = Experiment(cfg)
    k_full = cfg["negatives"]["ranking_candidates"]
    gnn = pd.read_csv(res / "gnn.csv")
    seeds = args.seeds if args.seeds is not None else sorted(gnn.seed.unique())
    models = sorted(gnn.model.unique())
    base = pd.read_csv(Path("results") / "baselines.csv")
    val_best = {}
    for p in PROTOCOL_ORDER:
        subset, metric = report.HEADLINE[p]
        d = base[(base.split == "val") & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)]
        val_best[p] = d.sort_values(["value", "model"], ascending=[False, True])["model"].iloc[0]

    # Pair-focused LightGBM (reference), refitted per seed with the Phase 4 settings.
    pf_params = search_best(cfg, PF)
    lmix = select_mix(pd.read_csv(Path("results") / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    href = E.heuristic_raw("test")

    def raw_of(scores, split, name, seed):
        return evaluate(None, E.splits[split], split, name, seed, cfg["seed"], 0, k_full=k_full, precomputed=scores)[1]

    pf_raw, pf_models = {}, {}
    for s in seeds:
        m = E.fit("lightgbm", ALL_FEATURES, lmix, s, pf_params, active=True)
        pf_models[s] = m
        pf_raw[s] = raw_of(FM.score_sets(m, E.Xsets("test", ALL_FEATURES)), "test", PF, s)

    # Paired tests.
    prow = []
    for name in models:
        sub = "ablation" if name.endswith("_dot") else "final"
        for s in seeds:
            d = sd / sub / name / f"seed{s}"
            if not (d / "scores_test.npz").exists():
                continue
            r = raw_of(load_scores(d, "test"), "test", name, s)
            for p in MAIN:
                nt = NEG.get(p, "hard")
                refs = {val_best[p]} | ({"adamic_adar"} if p in PAIR else set())
                for h in sorted(refs):
                    prow += paired(cfg, r, href[h][1], p, name, h, s, nt)
                prow += paired(cfg, r, pf_raw[s], p, name, PF, s, nt)
    write_results(prow, res / "gnn_paired.csv", append=False)

    # Hybrid: pair-focused LightGBM + out-of-fold GNN score.
    hyb_rows, hyb_name = [], None
    oof_dirs = sorted((sd / "oof").glob("seed*/meta.json"))
    if oof_dirs:
        gnn_name = json.loads(oof_dirs[0].read_text())["model"]
        hyb_name = f"{PF}+{gnn_name}"
        cols = ALL_FEATURES + ["gnn_score"]
        tr = E.splits["train"]
        for s in seeds:
            od = sd / "oof" / f"seed{s}"
            if not (od / "meta.json").exists():
                continue
            z = np.load(od / "train_pairs.npz")
            pairs, y = FM.training_pairs(tr, lmix[0], lmix[1], active=True)
            assert np.array_equal(pairs["u_idx"].values, z["u"]) and np.array_equal(pairs["v_idx"].values, z["v"])
            assert np.isfinite(z["score"]).all(), "every training pair needs an out-of-fold score"
            X = features_for(E.feat["train"], z["u"], z["v"], ALL_FEATURES).assign(gnn_score=z["score"])
            fd = sd / "final" / gnn_name / f"seed{s}"
            sc = {sp_: load_scores(fd, sp_) for sp_ in ("val", "test")}
            Xs = {sp_: {k: E.Xsets(sp_, ALL_FEATURES)[k].assign(gnn_score=sc[sp_][k]) for k in PairScores.SOURCES}
                  for sp_ in ("val", "test")}
            Xv, yv = FM.hard_eval_set(E.splits["val"], Xs["val"], "neg_hard_active")
            m = FM.fit_lightgbm(X[cols], y, Xv[cols], yv, pf_params, s)
            for sp_ in ("val", "test"):
                scores = {k: FM.predict(m, Xs[sp_][k][cols]) for k in PairScores.SOURCES}
                rows, x = evaluate(None, E.splits[sp_], sp_, hyb_name, s, cfg["seed"], cfg["evaluation"]["n_bootstrap"],
                                   k_full=k_full, precomputed=scores)
                hyb_rows += rows
                if sp_ == "test":
                    for p in PAIR:
                        prow += paired(cfg, x, pf_raw[s], p, hyb_name, PF, s, NEG.get(p, "hard"))
        write_results(hyb_rows, res / "gnn_hybrid.csv", append=False)
        write_results(prow, res / "gnn_paired.csv", append=False)
    write_report(cfg, res, models, hyb_name)


def write_report(cfg, res: Path, models, hyb_name):
    gnn = pd.read_csv(res / "gnn.csv")
    fm = pd.read_csv(Path("results") / "feature_models.csv")
    parts = [gnn, fm[fm.model == PF]]
    if (res / "gnn_hybrid.csv").exists() and hyb_name:
        parts.append(pd.read_csv(res / "gnn_hybrid.csv"))
    s = seed_summary(pd.concat(parts))
    base = pd.read_csv(Path("results") / "baselines.csv")
    pr = pd.read_csv(res / "gnn_paired.csv")
    order = models + [PF] + ([hyb_name] if hyb_name else [])
    ver = json.loads((res / "gnn_versions.json").read_text()) if (res / "gnn_versions.json").exists() else {}
    ks_h = [f"hits@{k}" for k in cfg["evaluation"]["rank_ks_hard"]]
    ks_r = [f"hits@{k}" for k in cfg["evaluation"]["rank_ks_random"]]
    out = ["# GNN link predictors (GraphSAGE, GAT)", "",
           "_Generated by `scripts/gnn_analysis.py` from `results/gnn.csv` (Colab), `gnn_paired.csv`, "
           "`gnn_hybrid.csv`, `gnn_grid.csv`, `gnn_determinism.csv`. Mean ± sd over seeds._", "",
           f"Versions used for training: `{json.dumps(ver)}`", ""]
    for split in ("test", "val"):
        out += [f"## {split}", ""]
        for p, sub, metrics in (("hard_active", "all", ["roc_auc", "average_precision"]),
                                ("rank_hard_active", "full_candidates", ks_h + ["mrr", "query_auc"]),
                                ("hard", "all", ["roc_auc", "average_precision"]),
                                ("rank_hard", "full_candidates", ks_h + ["mrr", "query_auc"]),
                                ("rank_random", "all", ks_r + ["mrr"]), ("random", "all", ["roc_auc", "average_precision"])):
            d = s[(s.split == split) & (s.protocol == p) & (s.subset == sub)]
            t = pd.DataFrame({"model": order})
            for m in metrics:
                x = d[d.metric == m].set_index("model")
                t[m] = [msd(x.loc[k]) if k in x.index else "" for k in order]
            h = report.metric_table(base, p, sub, split, metrics, ["adamic_adar"])
            h["model"] = h["model"] + " (heuristic)"
            tag = {"hard_active": " (lead pair-level metric)", "hard": " (includes activity)"}.get(p, "")
            out += [f"### {p} — {sub}{tag}", "", report.to_md(pd.concat([t, h])), ""]
        rows = []
        for p, metric in (("hard_active", "roc_auc"), ("rank_hard_active", "query_auc")):
            for sub in ("cold_start", "established"):
                d = s[(s.split == split) & (s.protocol == p) & (s.subset == sub) & (s.metric == metric)].set_index("model")
                row = {"protocol / subset": f"{p} {metric} — {sub}"}
                row.update({m: msd(d.loc[m]) for m in order if m in d.index})
                rows.append(row)
        out += [f"### Cold start vs established ({split})", "", report.to_md(pd.DataFrame(rows)), ""]
    pv = pr[pr.metric.str.endswith("_p_value")].rename(columns={"value": "p"})
    dv = pr[~pr.metric.str.endswith("_p_value")]
    m = dv.merge(pv[["model", "protocol", "seed", "p"]], on=["model", "protocol", "seed"])
    rows = [{"comparison": k[0], "protocol": k[1], "metric": d["metric"].iloc[0], "mean diff": f"{d['value'].mean():+.4f}",
             "mean 95% CI": f"[{d['ci_low'].mean():+.4f}, {d['ci_high'].mean():+.4f}]",
             "sig. better": f"{int(((d.p < 0.05) & (d.value > 0)).sum())}/{len(d)}",
             "sig. worse": f"{int(((d.p < 0.05) & (d.value < 0)).sum())}/{len(d)}", "median p": f"{d['p'].median():.3f}"}
            for k, d in m.groupby(["model", "protocol"], sort=False)]
    out += ["## Paired bootstrap tests (test, per seed)", "", report.to_md(pd.DataFrame(rows)), ""]
    det = res / "gnn_determinism.csv"
    if det.exists():
        dd = pd.read_csv(det)
        out += ["## Run-to-run variation with the same seed (GPU determinism check)", "",
                report.to_md(dd[["model", "split", "protocol", "metric", "value"]]), ""]
    out += ["## Appendix: diagnostic_all_vs_hard", "", report.DIAGNOSTIC_WARNING, ""]
    (res / "gnn.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
