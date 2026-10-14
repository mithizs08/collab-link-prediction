"""Phase 4: logistic regression and LightGBM on engineered pair features.

  python scripts/run_feature_models.py --config configs/default.yaml

Steps (all selection on VALIDATION; test only scored at the end):
 1. negative-mix grid: hard fraction {0, .5, 1} x ratio {1:1, 1:3}, LR and LightGBM (default
    params, seed 0). Rule (fixed before running): highest val `hard` ROC-AUC among mixes whose
    val rank_random MRR >= 90% of the best mix's; ties -> fewer hard negatives, lower ratio.
 2. LightGBM search on the chosen mix (config grid), selected by val `hard` ROC-AUC.
 3. final models x 5 seeds, all protocols, val + test, bootstrap CIs:
    logreg, lightgbm, lightgbm_d2_specialist (distance-2 positives vs hard negatives only)
 4. paired tests on test vs the validation-best heuristic per protocol and vs Adamic-Adar
 5. ablation by feature family (LightGBM, chosen mix/params, 5 seeds; hard + rank_hard)

Outputs: results/{negative_mix_grid,lgbm_search,feature_models,feature_models_paired,ablation}.csv,
results/feature_models.md, figures/feature_models_*.png, report/features/importance.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkpred.config import load_config, set_seed  # noqa: E402
from linkpred.data.splits import load_split  # noqa: E402
from linkpred.eval import bootstrap as bs  # noqa: E402
from linkpred.eval import report  # noqa: E402
from linkpred.eval.protocols import PROTOCOL_ORDER, evaluate  # noqa: E402
from linkpred.features.build import load_features  # noqa: E402
from linkpred.features.pair_features import ALL_FEATURES, FAMILIES  # noqa: E402
from linkpred.models import feature_models as FM  # noqa: E402
from linkpred.results import write_results  # noqa: E402

log = logging.getLogger("feature_models")
GRID_PROTOCOLS = ("hard_active", "rank_hard_active", "hard", "rank_hard", "rank_random", "random")
PAIR_PROTOCOLS = ("hard_active", "rank_hard_active", "hard", "rank_hard")
ABLATIONS = {
    "structural": ["structural"], "temporal": ["temporal"], "text": ["text"],
    "structural+temporal": ["structural", "temporal"], "structural+text": ["structural", "text"],
    "all": ["structural", "temporal", "text"],
}
HEURISTICS = ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment", "katz"]


def cols_for(families) -> list[str]:
    return [c for f in families for c in FAMILIES[f]]


def headline(rows, split, protocol):
    subset, metric = report.HEADLINE[protocol]
    for r in rows:
        if r["split"] == split and r["protocol"] == protocol and r["subset"] == subset and r["metric"] == metric:
            return r["value"]
    raise KeyError((split, protocol))


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg["paths"]["splits"])
        self.splits = {s: load_split(root, s) for s in ("train", "val", "test")}
        self.feat = {s: load_features(cfg, s) for s in ("train", "val", "test")}
        self.X = {s: FM.eval_features(self.splits[s], self.feat[s], ALL_FEATURES) for s in ("val", "test")}
        self.k_full = cfg["negatives"]["ranking_candidates"]
        self.n_boot = cfg["evaluation"]["n_bootstrap"]
        self.seed = cfg["seed"]

    def Xsets(self, split, columns):
        return {k: v[columns] for k, v in self.X[split].items()}

    def fit(self, kind, columns, mix, seed, params=None, specialist=False, active=False):
        """active=True: pair-focused model (activity-matched training negatives, early stopping on
        validation hard_active AP). Otherwise early stopping on validation hard AP (Phase 4)."""
        X, y = FM.training_set(self.splits["train"], self.feat["train"], columns, mix[0], mix[1], specialist, active)
        if kind == "logreg":
            return FM.fit_logreg(X, y, columns, seed)
        Xv, yv = FM.hard_eval_set(self.splits["val"], self.Xsets("val", columns),
                                  "neg_hard_active" if active else "neg_hard")
        return FM.fit_lightgbm(X, y, Xv, yv, params, seed)

    def evaluate(self, model, columns, name, seed, split, n_boot, protocols=PROTOCOL_ORDER):
        scores = FM.score_sets(model, self.Xsets(split, columns))
        return evaluate(None, self.splits[split], split, name, seed, self.seed, n_boot,
                        protocols=protocols, k_full=self.k_full, precomputed=scores)

    def heuristic_raw(self, split):
        out = {}
        for h in HEURISTICS:
            pre = {k: v[h].values for k, v in self.X[split].items()}
            rows, raw = evaluate(None, self.splits[split], split, h, self.seed, self.seed, 0,
                                 k_full=self.k_full, precomputed=pre)
            out[h] = (rows, raw)
        return out


def select_mix(grid_rows, kind):
    df = pd.DataFrame([r for r in grid_rows if r["model"].startswith(kind + "|")])
    summ = []
    for m, d in df.groupby("model"):
        frac = float(m.split("hard=")[1].split("|")[0]); ratio = int(m.split("1:")[1])
        get = lambda p: d[(d.protocol == p) & (d.subset == report.HEADLINE[p][0]) & (d.metric == report.HEADLINE[p][1])]["value"].iloc[0]
        summ.append({"model": m, "frac": frac, "ratio": ratio, "hard": get("hard"), "rank_random": get("rank_random")})
    s = pd.DataFrame(summ)
    eligible = s[s["rank_random"] >= 0.9 * s["rank_random"].max()]
    best = eligible.assign(h=eligible["hard"].round(6)).sort_values(["h", "frac", "ratio"], ascending=[False, True, True]).iloc[0]
    return (float(best["frac"]), int(best["ratio"])), s


def paired(cfg, model_raw, ref_raw, protocol, label_a, label_b, seed, neg_type):
    subset, metric = report.HEADLINE[protocol]
    ra, rb = model_raw[protocol][subset], ref_raw[protocol][subset]
    rng = np.random.default_rng([cfg["seed"], 77, PROTOCOL_ORDER.index(protocol)])
    n = cfg["evaluation"]["n_bootstrap"]
    res = (bs.paired_classification(ra["pos"], ra["neg"], rb["pos"], rb["neg"], metric, rng, n) if "pos" in ra
           else bs.paired_mean(ra[metric], rb[metric], rng, n))
    base = {"model": f"{label_a} - {label_b}", "split": "test", "protocol": protocol, "subset": subset,
            "negative_type": neg_type, "seed": seed}
    return [{**base, "metric": f"{metric}_diff", "value": res["diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]},
            {**base, "metric": f"{metric}_diff_p_value", "value": res["p_value"]}]


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
    E = Experiment(cfg)
    lcfg = cfg["models"]["lightgbm"]
    default = {k: lcfg[k] for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree")}

    # 1. negative-mix grid ------------------------------------------------------------
    t0 = time.time()
    grid_rows = []
    mixes = list(itertools.product(cfg["models"]["training_mix"]["hard_fractions"], cfg["models"]["training_mix"]["ratios"]))
    for kind in ("logreg", "lightgbm"):
        for frac, ratio in mixes:
            m = E.fit(kind, ALL_FEATURES, (frac, ratio), cfg["seed"], default)
            r, _ = E.evaluate(m, ALL_FEATURES, f"{kind}|hard={frac:g}|1:{ratio}", cfg["seed"], "val", 0, GRID_PROTOCOLS)
            grid_rows += r
    write_results(grid_rows, res / "negative_mix_grid.csv", append=False)
    mix = {k: select_mix(grid_rows, k)[0] for k in ("logreg", "lightgbm")}
    log.info("mix grid done (%.0fs): %s", time.time() - t0, mix)

    # 2. LightGBM search ----------------------------------------------------------------
    search_rows, best_params, best_auc = [], None, -1
    sgrid = lcfg["search"]
    for nl, lr, mcs in itertools.product(sgrid["num_leaves"], sgrid["learning_rate"], sgrid["min_child_samples"]):
        p = {**default, "num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        m = E.fit("lightgbm", ALL_FEATURES, mix["lightgbm"], cfg["seed"], p)
        name = f"lightgbm|leaves={nl}|lr={lr:g}|min_child={mcs}"
        r, _ = E.evaluate(m, ALL_FEATURES, name, cfg["seed"], "val", 0, ("hard",))
        r = [x for x in r if x["subset"] == "all"]
        r.append({**r[0], "metric": "n_trees", "value": float(m.best_iteration_ or m.n_estimators)})
        search_rows += r
        auc = headline(r, "val", "hard")
        if round(auc, 6) > round(best_auc, 6):
            best_auc, best_params = auc, p
    # Pair-focused LightGBM: same grid, activity-matched training negatives (ratio of the chosen
    # general mix), early stopping on val hard_active AP, selected by val hard_active ROC-AUC.
    pf_params, pf_auc = None, -1
    for nl, lr, mcs in itertools.product(sgrid["num_leaves"], sgrid["learning_rate"], sgrid["min_child_samples"]):
        p = {**default, "num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        m = E.fit("lightgbm", ALL_FEATURES, mix["lightgbm"], cfg["seed"], p, active=True)
        name = f"lightgbm_pair_focused|leaves={nl}|lr={lr:g}|min_child={mcs}"
        r, _ = E.evaluate(m, ALL_FEATURES, name, cfg["seed"], "val", 0, ("hard_active",))
        r = [x for x in r if x["subset"] == "all"]
        r.append({**r[0], "metric": "n_trees", "value": float(m.best_iteration_ or m.n_estimators)})
        search_rows += r
        auc = headline(r, "val", "hard_active")
        if round(auc, 6) > round(pf_auc, 6):
            pf_auc, pf_params = auc, p
    write_results(search_rows, res / "lgbm_search.csv", append=False)
    log.info("search done: %s (val hard AUC %.4f); pair-focused %s (val hard_active AUC %.4f)",
             best_params, best_auc, pf_params, pf_auc)

    # 3. final models -------------------------------------------------------------------
    final_rows, raw = [], {}
    specs = {"logreg": ("logreg", mix["logreg"], None, False, False),
             "lightgbm": ("lightgbm", mix["lightgbm"], best_params, False, False),
             "lightgbm_d2_specialist": ("lightgbm", mix["lightgbm"], best_params, True, False),
             "lightgbm_pair_focused": ("lightgbm", mix["lightgbm"], pf_params, False, True)}
    importances = []
    for name, (kind, mx, params, spec, act) in specs.items():
        for s in seeds:
            m = E.fit(kind, ALL_FEATURES, mx, s, params, spec, act)
            if kind == "lightgbm":
                importances.append(pd.DataFrame({"model": name, "seed": s, "feature": ALL_FEATURES,
                                                 "gain": m.booster_.feature_importance("gain")}))
            for split in ("val", "test"):
                r, x = E.evaluate(m, ALL_FEATURES, name, s, split, E.n_boot)
                final_rows += r
                raw[(name, s, split)] = x
            log.info("final %s seed %d (%.0fs)", name, s, time.time() - t0)
    write_results(final_rows, res / "feature_models.csv", append=False)
    imp = pd.concat(importances)
    fam = {c: f for f, cs in FAMILIES.items() for c in cs}
    imp["family"] = imp["feature"].map(fam)
    Path(cfg["paths"]["report"], "features").mkdir(parents=True, exist_ok=True)
    imp.to_csv(Path(cfg["paths"]["report"]) / "features" / "importance.csv", index=False)

    # 4. paired tests on test -------------------------------------------------------------
    base = pd.read_csv(res / "baselines.csv")
    val_best = {}
    for p in GRID_PROTOCOLS:
        subset, metric = report.HEADLINE[p]
        d = base[(base.split == "val") & (base.protocol == p) & (base.subset == subset) & (base.metric == metric)]
        val_best[p] = d.sort_values(["value", "model"], ascending=[False, True])["model"].iloc[0]
    href = E.heuristic_raw("test")
    prow = []
    for name in specs:
        for s in seeds:
            for p in GRID_PROTOCOLS:
                neg_type = {"random": "random", "rank_random": "random", "hard_active": "hard_active",
                            "rank_hard_active": "hard_active"}.get(p, "hard")
                refs = {val_best[p]} | ({"adamic_adar"} if p in PAIR_PROTOCOLS else set())
                for h in sorted(refs):
                    prow += paired(cfg, raw[(name, s, "test")], href[h][1], p, name, h, s, neg_type)
            for p in PAIR_PROTOCOLS:
                nt = "hard_active" if p.endswith("active") else "hard"
                if name in ("lightgbm_d2_specialist", "lightgbm_pair_focused"):
                    prow += paired(cfg, raw[(name, s, "test")], raw[("lightgbm", s, "test")], p, name, "lightgbm", s, nt)
    write_results(prow, res / "feature_models_paired.csv", append=False)
    log.info("paired tests done (%.0fs)", time.time() - t0)

    # 5. ablation -------------------------------------------------------------------------
    abl = []
    for label, fams in ABLATIONS.items():
        columns = cols_for(fams)
        for s in seeds:
            m = E.fit("lightgbm", columns, mix["lightgbm"], s, best_params)
            for split in ("val", "test"):
                r, _ = E.evaluate(m, columns, f"lightgbm[{label}]", s, split, E.n_boot, PAIR_PROTOCOLS)
                abl += r
        log.info("ablation %s (%.0fs)", label, time.time() - t0)
    write_results(abl, res / "ablation.csv", append=False)

    from linkpred.eval.feature_report import write_feature_report  # noqa: E402
    write_feature_report(cfg, mix, best_params, val_best, pf_params)
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
