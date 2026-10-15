"""Primary model comparison (audit M2 + M5): macro within-group metrics, seed-mean CIs, and one paired
bootstrap per headline comparison on seed-averaged scores, Holm-corrected across the family.

  python scripts/final_comparison.py

Headline metrics (decisions.md 2026-10-15):
  hard_active        macro ROC-AUC  = mean of the established-group and newcomer-group ROC-AUCs
  rank_hard_active   macro query AUC = mean of the two groups' mean per-query AUCs
(group-matched: negatives / candidates come from the same group as the positives.)

Values for learned models are the mean over seeds of the per-seed metric, with a SEED-MEAN bootstrap CI
(each resample is applied to every seed and the seeds' metrics are averaged). Deterministic models have
an ordinary bootstrap CI. `seed` = -1 in the output marks a mean over seeds 0-4.

Primary test family (Holm-corrected together, test split; the same family on validation separately;
docs/decisions.md, "Phase 7 design"):
  each of {prefer_newer, jaccard, logreg, lightgbm, lightgbm_pair_focused, node2vec, best GNN, hybrid}
  minus adamic_adar, on both headline metrics                                          (16 tests)
  hybrid minus lightgbm_pair_focused, and lightgbm_pair_focused minus lightgbm, on both (4 tests)
  Phase 7: disambiguation sensitivity, pair-focused LightGBM minus Adamic-Adar on the reduced data, on both
  headline metrics (2; raw p from results/disambiguation_sensitivity_tests.csv); two-stage minus general
  LightGBM on pooled and macro hard ROC-AUC (2; raw p from results/two_stage_tests.csv)  (24 tests)
Each test: one paired bootstrap (evaluation.n_bootstrap_final = 5,000 resamples) on the seed-AVERAGED scores.

Writes results/final_comparison.csv, results/final_comparison_tests.csv, results/final_comparison.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.eval import bootstrap as bs  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.protocols import PairScores, evaluate, paired_test  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.models import heuristics as H  # noqa: E402
from linkpred.models import node2vec as N2V  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from activity_diagnostic import search_best  # noqa: E402
from gnn_analysis import fit_hybrid, load_scores  # noqa: E402
from run_feature_models import Experiment, select_mix  # noqa: E402

HEAD = (("hard_active", "macro", "roc_auc"), ("rank_hard_active", "macro", "query_auc"))
EXTRA = (("hard_active", "established_matched", "roc_auc"), ("hard_active", "newcomer_matched", "roc_auc"),
         ("rank_hard_active", "established_matched", "query_auc"), ("rank_hard_active", "newcomer_matched", "query_auc"),
         ("hard", "macro", "roc_auc"), ("rank_hard", "macro", "query_auc"),
         ("hard_active", "all", "roc_auc"), ("rank_hard_active", "full_candidates", "mrr"),
         ("hard", "all", "roc_auc"), ("random", "all", "roc_auc"))
PHASE7_TESTS = ("disambiguation_sensitivity_tests.csv", "two_stage_tests.csv")
PF = "lightgbm_pair_focused"
GNN_DIR = Path("data/interim/gnn")


NEG = {"hard_active": "hard_active", "rank_hard_active": "hard_active", "hard": "hard", "rank_hard": "hard",
       "random": "random"}


def phase7_tests(res: Path, split: str) -> dict:
    """{(comparison, protocol, subset, metric): diff, CI, raw p} from the Phase 7 scripts' test files."""
    out = {}
    for f in PHASE7_TESTS:
        t = pd.read_csv(res / f)
        t = t[t.split == split]
        for (name, p, sub), d in t.groupby(["model", "protocol", "subset"], sort=False):
            for m in d.metric[~d.metric.str.endswith("_p_value")]:
                metric = m[:-len("_diff")]
                dd, pp = d[d.metric == m].iloc[0], d[d.metric == f"{m}_p_value"].iloc[0]
                out[(name, p, sub, metric)] = {"diff": dd.value, "ci_low": dd.ci_low, "ci_high": dd.ci_high,
                                               "p_value": pp.value, "negative_type": dd.negative_type}
    return out


def node2vec_choice(res: Path, ncfg: dict):
    grid = pd.read_csv(res / "node2vec_grid.csv").to_dict("records")
    best = None
    for p in ncfg["p_grid"]:
        for q in ncfg["q_grid"]:
            for op in ncfg["operators"]:
                kind = f"node2vec|p={p:g}|q={q:g}|op={op}"
                mix, _ = select_mix(grid, kind)
                name = f"{kind}|hard={mix[0]:g}|1:{mix[1]}"
                v = next(x["value"] for x in grid if x["model"] == name and x["protocol"] == "hard_active"
                         and x["subset"] == "all" and x["metric"] == "roc_auc")
                key = (round(v, 6), -p, -q)
                if best is None or key > best[0]:
                    best = (key, float(p), float(q), op, mix)
    return best[1:]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    res = Path(cfg["paths"]["results"])
    E = Experiment(cfg)
    seeds = cfg["seeds"]
    n_boot = cfg["evaluation"]["n_bootstrap_final"]
    k_full = cfg["negatives"]["ranking_candidates"]
    lmix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "lightgbm")[0]
    lr_mix = select_mix(pd.read_csv(res / "negative_mix_grid.csv").to_dict("records"), "logreg")[0]
    gnn = pd.read_csv(res / "gnn.csv")
    gv = gnn[(gnn.split == "val") & (gnn.protocol == "hard_active") & (gnn.subset == "macro") & (gnn.metric == "roc_auc")
             & ~gnn.model.str.endswith("_dot")]
    best_gnn = gv.groupby("model").value.mean().round(6).sort_values(ascending=False, kind="stable").index[0]
    hyb_gnn = json.loads(sorted((GNN_DIR / "oof").glob("seed*/meta.json"))[0].read_text())["model"]
    hybrid = f"{PF}+{hyb_gnn}"

    # ---- scores[model][split] = list over seeds of {pair set: scores}
    scores: dict = {}

    def add(model, split, sc):
        scores.setdefault(model, {}).setdefault(split, []).append(sc)

    for split in ("val", "test"):
        sp_ = E.splits[split]
        add("prefer_newer", split, {k: H.prefer_newer(sp_, sp_[t][a].values, sp_[t][b].values)
                                    for k, (t, a, b) in PairScores.SOURCES.items()})
        for h in ("adamic_adar", "jaccard"):
            add(h, split, {k: v[h].values for k, v in E.X[split].items()})
    lr = E.fit("logreg", ALL_FEATURES, lr_mix, cfg["seed"])
    for split in ("val", "test"):
        add("logreg", split, FM.score_sets(lr, E.Xsets(split, ALL_FEATURES)))
    gp, pfp = search_best(cfg, "lightgbm"), search_best(cfg, PF)
    for s in seeds:
        mg = E.fit("lightgbm", ALL_FEATURES, lmix, s, gp)
        mp = E.fit("lightgbm", ALL_FEATURES, lmix, s, pfp, active=True, d2_only=True)
        hs = fit_hybrid(E, GNN_DIR, s, hyb_gnn, pfp, lmix)
        for split in ("val", "test"):
            add("lightgbm", split, FM.score_sets(mg, E.Xsets(split, ALL_FEATURES)))
            add(PF, split, FM.score_sets(mp, E.Xsets(split, ALL_FEATURES)))
            add(hybrid, split, hs[split])
            add(best_gnn, split, load_scores(GNN_DIR / "final" / best_gnn / f"seed{s}", split))
    # node2vec (reported L1 configuration, chosen on validation)
    ncfg = cfg["models"]["node2vec"]
    P, Q, OP, MIX = node2vec_choice(res, ncfg)
    cache = Path(cfg["paths"]["interim"]) / "node2vec"
    tr = E.splits["train"]
    for s in seeds:
        raw = {sp: N2V.cached_embeddings(cache, sp, E.splits[sp]["A"], P, Q, s, ncfg) for sp in ("train", "val", "test")}
        pairs, y = FM.training_pairs(tr, MIX[0], MIX[1])
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(
            N2V.edge_features(raw["train"], pairs.u_idx.values, pairs.v_idx.values, OP), y)
        for split in ("val", "test"):
            al = N2V.align_to(raw[split], E.splits[split]["nodes"], raw["train"], tr["nodes"])
            add("node2vec", split, {k: clf.predict_proba(N2V.edge_features(al, E.splits[split][t][a].values,
                                                                           E.splits[split][t][b].values, OP))[:, 1]
                                    for k, (t, a, b) in PairScores.SOURCES.items()})
    models = ["prefer_newer", "adamic_adar", "jaccard", "logreg", "lightgbm", PF, "node2vec", best_gnn, hybrid]
    protos = ("hard_active", "rank_hard_active", "hard", "rank_hard", "random")

    def raws_of(model, split):
        return [evaluate(None, E.splits[split], split, model, 0, cfg["seed"], 0, protocols=protos, k_full=k_full,
                         precomputed=sc)[1] for sc in scores[model][split]]

    def averaged_raw(model, split):
        avg = {k: np.mean([sc[k] for sc in scores[model][split]], axis=0) for k in PairScores.SOURCES}
        return evaluate(None, E.splits[split], split, model, 0, cfg["seed"], 0, protocols=protos, k_full=k_full,
                        precomputed=avg)[1]

    rows, tests = [], []
    for split in ("val", "test"):
        R = {m: raws_of(m, split) for m in models}
        for m in models:
            for k, (p, sub, metric) in enumerate(HEAD + EXTRA):
                vals = [bs.raw_stat(r[p][sub], metric) for r in R[m]]
                rng = np.random.default_rng([cfg["seed"], 31, k, 0 if split == "val" else 1])
                lo, hi = bs.seed_mean_ci([r[p][sub] for r in R[m]], metric, rng, n_boot)
                rows.append({"model": m, "split": split, "protocol": p, "subset": sub,
                             "negative_type": NEG[p],
                             "seed": -1 if len(vals) > 1 else cfg["seed"], "metric": metric,
                             "value": float(np.mean(vals)), "ci_low": lo, "ci_high": hi})
                rows.append({"model": m, "split": split, "protocol": p, "subset": sub,
                             "negative_type": NEG[p],
                             "seed": -1 if len(vals) > 1 else cfg["seed"], "metric": f"{metric}_seed_sd",
                             "value": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
        A = {m: averaged_raw(m, split) for m in models}
        family = [(m, "adamic_adar") for m in models if m != "adamic_adar"] + [(hybrid, PF), (PF, "lightgbm")]
        res_tests = {}
        for i, (a, b) in enumerate(family):
            for j, (p, sub, metric) in enumerate(HEAD):
                rng = np.random.default_rng([cfg["seed"], 41, i, j, 0 if split == "val" else 1])
                r = paired_test(A[a], A[b], p, sub, metric, rng, n_boot)
                res_tests[(a, b, p, metric)] = r
        # Phase 7 comparisons: raw p-values from their scripts (same 5,000-resample procedure), Holm-corrected here.
        ext = phase7_tests(res, split)
        pv = {("own",) + k: v["p_value"] for k, v in res_tests.items()}
        pv.update({("ext",) + k: v["p_value"] for k, v in ext.items()})
        adj = bs.holm(pv)
        assert len(pv) == 24, f"family has {len(pv)} tests, decisions.md defines 24"
        for (a, b, p, metric), r in res_tests.items():
            base = {"model": f"{a} - {b}", "split": split, "protocol": p, "subset": "macro",
                    "negative_type": "hard_active", "seed": -1}
            tests += [{**base, "metric": f"{metric}_diff", "value": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]},
                      {**base, "metric": f"{metric}_diff_p_value", "value": r["p_value"]},
                      {**base, "metric": f"{metric}_diff_p_holm", "value": adj[("own", a, b, p, metric)]}]
        for (name, p, sub, metric), r in ext.items():
            base = {"model": name, "split": split, "protocol": p, "subset": sub, "negative_type": r["negative_type"], "seed": -1}
            tests += [{**base, "metric": f"{metric}_diff", "value": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]},
                      {**base, "metric": f"{metric}_diff_p_value", "value": r["p_value"]},
                      {**base, "metric": f"{metric}_diff_p_holm", "value": adj[("ext", name, p, sub, metric)]}]
        print(split, "done", flush=True)
    write_results(rows, res / "final_comparison.csv", append=False)
    write_results(tests, res / "final_comparison_tests.csv", append=False)
    write_report(res, models, best_gnn, hybrid, (P, Q, OP))


def write_report(res: Path, models, best_gnn, hybrid, n2v):
    d = pd.read_csv(res / "final_comparison.csv")
    t = pd.read_csv(res / "final_comparison_tests.csv")

    def cell(m, split, p, sub, metric):
        x = d[(d.model == m) & (d.split == split) & (d.protocol == p) & (d.subset == sub) & (d.metric == metric)]
        sd = d[(d.model == m) & (d.split == split) & (d.protocol == p) & (d.subset == sub) & (d.metric == f"{metric}_seed_sd")]
        if not len(x):
            return ""
        r = x.iloc[0]
        s = f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
        return s + (f" (sd {sd.value.iloc[0]:.3f})" if len(sd) and sd.value.iloc[0] > 0 else "")

    def test_cell(a, split, p, metric, b="adamic_adar"):
        x = t[(t.model == f"{a} - {b}") & (t.split == split) & (t.protocol == p)]
        if not len(x):
            return "—"
        g = lambda k: x[x.metric == f"{metric}_diff{k}"].iloc[0]
        dd = g("")
        return f"{dd.value:+.3f} [{dd.ci_low:+.3f}, {dd.ci_high:+.3f}], p={g('_p_value').value:.4f}, Holm p={g('_p_holm').value:.4f}"

    sc_lab, sc_note = report.shortcut_labels(res, best_gnn, hybrid)
    label = {**sc_lab, PF: f"{PF} (retrained, distance-2 positives)", "prefer_newer": "prefer_newer (reference)"}
    out = ["# Final model comparison (primary results)", "",
           "_Generated by `scripts/final_comparison.py` from `results/final_comparison.csv` and "
           "`final_comparison_tests.csv`._", "",
           "**Metrics.** Macro within-group scores (audit M2): positives and negatives/candidates are compared inside the "
           "established group (no author with ≤ 2 works before the cutoff) and the newcomer group separately, and the two "
           "scores are averaged. `hard_active` uses ROC-AUC; `rank_hard_active` uses query AUC (count-independent, "
           "because group matching changes candidate counts). **Values** for learned models are the mean over 5 seeds with a "
           "**seed-mean bootstrap 95% CI** (each resample applied to every seed, seeds' metrics averaged); sd = seed-to-seed "
           "sd. Deterministic models: ordinary bootstrap CI. **Tests** (primary): one paired bootstrap per comparison on "
           "the seed-averaged scores, 5,000 resamples, Holm-corrected across the 24-comparison family defined in "
           "`docs/decisions.md` (\"Phase 7 design\"; each split corrected separately). The smallest attainable raw p is "
           "0.0002.", "",
           f"Best GNN = {best_gnn} (highest validation macro hard_active ROC-AUC). node2vec = reported configuration "
           f"(p = {n2v[0]:g}, q = {n2v[1]:g}, {n2v[2]} operator, chosen on validation).", "", sc_note, ""]
    for split in ("test", "val"):
        out += [f"## {split.capitalize()}", ""]
        for p, metric, title in (("hard_active", "roc_auc", "macro hard_active ROC-AUC"),
                                 ("rank_hard_active", "query_auc", "macro rank_hard_active query AUC")):
            tbl = pd.DataFrame({"model": [label.get(m, m) for m in models]})
            tbl[title] = [cell(m, split, p, "macro", metric) for m in models]
            tbl["established group"] = [cell(m, split, p, "established_matched", metric) for m in models]
            tbl["newcomer group"] = [cell(m, split, p, "newcomer_matched", metric) for m in models]
            tbl["vs Adamic-Adar (primary test)"] = [test_cell(m, split, p, metric) if m != "adamic_adar" else "—" for m in models]
            out += [f"### {title}", "", report.to_md(tbl), ""]
        tbl = pd.DataFrame({"comparison": [f"{hybrid} − {PF}", f"{PF} − lightgbm"]})
        for p, metric in (("hard_active", "roc_auc"), ("rank_hard_active", "query_auc")):
            tbl[f"{p} macro {metric}"] = [test_cell(hybrid, split, p, metric, PF), test_cell(PF, split, p, metric, "lightgbm")]
        out += ["### Other headline comparisons", "", report.to_md(tbl), ""]
        tp = t[(t.split == split) & t.model.str.contains("no_flagged_ids|two_stage", regex=True)]
        prow = []
        for (name, p, sub), d in tp.groupby(["model", "protocol", "subset"], sort=False):
            for m in d.metric[d.metric.str.endswith("_diff")]:
                g = lambda k: d[d.metric == f"{m}{k}"].iloc[0]
                prow.append({"comparison": name, "metric": f"{p} {sub} {m[:-5]}",
                             "difference [95% CI]": f"{g('').value:+.3f} [{g('').ci_low:+.3f}, {g('').ci_high:+.3f}]",
                             "p": f"{g('_p_value').value:.4f}", "Holm p": f"{g('_p_holm').value:.4f}"})
        if prow:
            out += ["### Phase 7 comparisons in the family (disambiguation sensitivity, two-stage)", "",
                    "Diff and raw p from `results/disambiguation_sensitivity_tests.csv` / `results/two_stage_tests.csv`; "
                    "Holm across all 24 tests. Model values: `results/disambiguation_sensitivity.csv`, "
                    "`results/two_stage.csv`, summarised in `results/final_summary.md`.", "",
                    report.to_md(pd.DataFrame(prow)), ""]
        tbl = pd.DataFrame({"model": [label.get(m, m) for m in models]})
        for p, sub, metric, lab in (("hard", "macro", "roc_auc", "hard macro ROC-AUC (includes activity)"),
                                    ("rank_hard", "macro", "query_auc", "rank_hard macro query AUC (includes activity)"),
                                    ("hard_active", "all", "roc_auc", f"hard_active ROC-AUC ({report.POOLED_NOTE})"),
                                    ("rank_hard_active", "full_candidates", "mrr", f"rank_hard_active MRR, full lists ({report.POOLED_NOTE})"),
                                    ("hard", "all", "roc_auc", f"hard ROC-AUC (includes activity; {report.POOLED_NOTE})"),
                                    ("random", "all", "roc_auc", "random ROC-AUC")):
            tbl[lab] = [cell(m, split, p, sub, metric) for m in models]
        out += ["### Secondary metrics", "", report.to_md(tbl), ""]
    (res / "final_comparison.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
