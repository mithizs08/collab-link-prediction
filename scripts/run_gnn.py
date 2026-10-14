"""Phase 6: GNN link predictors -- grid, final seeds, dot-decoder ablation, determinism check,
out-of-fold scores for the hybrid. Runs locally (CPU dry run) and on Colab (GPU), same code.

  python scripts/run_gnn.py --out <dir> [--device cuda] [--dry]

Every unit of work (one grid config, one final seed, ...) is written to its own folder under
--out as soon as it finishes, with a meta.json marker; a rerun skips completed units, so a
Colab disconnect loses at most the unit in progress. At the end, combined CSVs are written:
  <out>/gnn_grid.csv  <out>/gnn.csv  <out>/gnn_determinism.csv  <out>/versions.json
and per-run pair scores (<out>/final/<model>/seed<k>/scores_{val,test}.npz, <out>/oof/...) for
the local analysis (scripts/gnn_analysis.py: paired tests, hybrid LightGBM).

Selection (validation only):
  general       early stopping on val `hard` ROC-AUC; grid choice by val `hard` ROC-AUC
  pair_focused  early stopping on val `hard_active` AP; grid choice by val `hard_active` ROC-AUC
  best GNN overall (determinism check, hybrid): highest mean val `hard_active` ROC-AUC over seeds
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkpred.config import load_config  # noqa: E402
from linkpred.data.splits import load_split  # noqa: E402
from linkpred.eval.protocols import PairScores, evaluate  # noqa: E402
from linkpred.features.node_features import load_node_features  # noqa: E402
from linkpred.models import gnn as G  # noqa: E402
from linkpred.models.feature_models import training_pairs  # noqa: E402
from linkpred.results import RESULT_COLUMNS, validate_row  # noqa: E402

GRID_PROTOCOLS = ("hard_active", "rank_hard_active", "hard", "rank_random")
SELECT = {"general": ("hard", "roc_auc"), "pair_focused": ("hard_active", "roc_auc")}
EARLY_STOP = {"general": ("neg_hard", "roc_auc"), "pair_focused": ("neg_hard_active", "average_precision")}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Runner:
    def __init__(self, cfg, out: Path, device: str, dry: bool):
        self.cfg, self.out, self.device, self.dry = cfg, out, device, dry
        self.g = cfg["models"]["gnn"]
        if dry:   # heavily reduced setting for the local CPU dry run
            self.g = {**self.g, "max_epochs": 2, "patience": 2, "oof_folds": 2}
        self.seeds = [cfg["seed"]] if dry else cfg["seeds"]
        self.n_boot = 20 if dry else cfg["evaluation"]["n_bootstrap"]
        root, proc = Path(cfg["paths"]["splits"]), Path(cfg["paths"]["processed"])
        self.splits = {s: load_split(root, s) for s in ("train", "val", "test")}
        struct = {s: load_node_features(proc, s, "structural") for s in self.splits}
        mean, std = struct["train"].mean(0), struct["train"].std(0)   # train-split (pre-cutoff) statistics
        self.graphs = {}
        for fs in ("structural", "text"):
            for s in self.splits:
                text = load_node_features(proc, s, "text") if fs == "text" else None
                x = torch.from_numpy(G.make_node_features(struct[s], text, mean, std))
                self.graphs[(fs, s)] = G.GraphData(x, G.edge_index_from_adjacency(self.splits[s]["A"]))

    # ------------------------------------------------------------ pieces

    def sup_pairs(self, objective, exclude: set | None = None):
        tr = self.splits["train"]
        if objective == "general":
            frac, ratio = self.g["general_mix"]
            pairs, y = training_pairs(tr, frac, ratio)
        else:
            pairs, y = training_pairs(tr, 0.0, self.g["pair_focused_ratio"], active=True)
        u, v = pairs["u_idx"].values, pairs["v_idx"].values
        if exclude:
            keep = np.array([(min(a, b), max(a, b)) not in exclude for a, b in zip(u, v)])
            u, v, y = u[keep], v[keep], y[keep]
        G.assert_supervision_not_in_graph(tr["A"], u, v)
        return u, v, y

    def es_set(self, objective):
        val = self.splits["val"]
        neg_table, metric = EARLY_STOP[objective]
        d2 = (val["positives"]["distance"] == 2).values
        u = np.r_[val["positives"]["u_idx"].values[d2], val[neg_table]["u_idx"].values]
        v = np.r_[val["positives"]["v_idx"].values[d2], val[neg_table]["v_idx"].values]
        y = np.r_[np.ones(d2.sum()), np.zeros(len(val[neg_table]))]
        return (u, v), y, metric

    def fit(self, c, seed, exclude=None):
        G.set_determinism(seed)
        u, v, y = self.sup_pairs(c["objective"], exclude)
        uv_val, y_val, metric = self.es_set(c["objective"])
        gt = self.graphs[(c["features"], "train")].to(self.device)
        gv = self.graphs[(c["features"], "val")].to(self.device)
        model = G.LinkPredictor(c["encoder"], c["decoder"], gt.x.shape[1], c["hidden"], c["layers"], c["dropout"],
                                self.g["heads"])
        res = G.train(model, gt, u, v, y, gv, uv_val, y_val, metric=metric, lr=c["lr"],
                      weight_decay=self.g["weight_decay"], epochs=self.g["max_epochs"], patience=self.g["patience"],
                      device=self.device, log=log)
        return res

    def scores(self, model, c, split):
        g = self.graphs[(c["features"], split)].to(self.device)
        return {n: model.score(g, self.splits[split][t][a].values, self.splits[split][t][b].values)
                for n, (t, a, b) in PairScores.SOURCES.items()}

    def eval_rows(self, scores, split, name, seed, n_boot, protocols=None):
        kw = {"protocols": protocols} if protocols else {}
        rows, _ = evaluate(None, self.splits[split], split, name, seed, self.cfg["seed"], n_boot,
                           k_full=self.cfg["negatives"]["ranking_candidates"], precomputed=scores, **kw)
        return rows

    # ------------------------------------------------------------ stages

    def grid_configs(self):
        out = []
        gr = self.g["grid"]
        for objective, features, encoder in itertools.product(self.g["objectives"], self.g["node_features"], ("sage", "gat")):
            e = gr[encoder]
            combos = list(itertools.product(e["hidden"], e["layers"], e["dropout"], e["lr"]))
            if self.dry:
                combos = combos[:1]
            for h, L, d, lr in combos:
                out.append({"objective": objective, "features": features, "encoder": encoder, "decoder": "mlp",
                            "hidden": h, "layers": L, "dropout": d, "lr": lr})
        return out

    @staticmethod
    def cid(c):
        return f"{c['objective']}__{c['encoder']}__{c['features']}__{c['decoder']}__h{c['hidden']}_L{c['layers']}_d{c['dropout']:g}_lr{c['lr']:g}"

    @staticmethod
    def model_name(c):
        return f"gnn_{c['objective']}_{c['encoder']}_{c['features']}" + ("_dot" if c["decoder"] == "dot" else "")

    def save_unit(self, d: Path, rows, curve, meta, scores=None):
        d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            validate_row(r)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(d / "rows.csv", index=False)
        if curve is not None:
            pd.DataFrame(curve).to_csv(d / "curve.csv", index=False)
        for split, sc in (scores or {}).items():
            np.savez_compressed(d / f"scores_{split}.npz", **{k: v.astype(np.float32) for k, v in sc.items()})
        (d / "meta.json").write_text(json.dumps(meta, indent=2))       # written LAST = completion marker

    def run_grid(self):
        for c in self.grid_configs():
            d = self.out / "grid" / self.cid(c)
            if (d / "meta.json").exists():
                continue
            log(f"grid {self.cid(c)}")
            res = self.fit(c, self.cfg["seed"])
            rows = self.eval_rows(self.scores(res.model, c, "val"), "val", f"{self.model_name(c)}|{self.cid(c)}",
                                  self.cfg["seed"], 0, GRID_PROTOCOLS)
            self.save_unit(d, rows, res.curve, {**c, "best_epoch": res.best_epoch, "best_val": res.best_val,
                                                "seconds": res.seconds, "epochs_run": len(res.curve)})

    def selected(self):
        """Best grid config per (objective, encoder, features) by the objective's validation metric."""
        best = {}
        for d in sorted((self.out / "grid").glob("*/meta.json")):
            c = json.loads(d.read_text())
            rows = pd.read_csv(d.parent / "rows.csv")
            p, m = SELECT[c["objective"]]
            val = rows[(rows.protocol == p) & (rows.subset == "all") & (rows.metric == m)]["value"].iloc[0]
            key = (c["objective"], c["encoder"], c["features"])
            if key not in best or round(val, 6) > round(best[key][1], 6):
                best[key] = ({k: c[k] for k in ("objective", "features", "encoder", "decoder", "hidden", "layers",
                                                "dropout", "lr")}, val)
        return [b[0] for b in best.values()]

    def run_final(self, configs, tag="final"):
        for c in configs:
            name = self.model_name(c)
            for s in self.seeds:
                d = self.out / tag / name / f"seed{s}"
                if (d / "meta.json").exists():
                    continue
                log(f"{tag} {name} seed {s}")
                res = self.fit(c, s)
                sc = {sp_: self.scores(res.model, c, sp_) for sp_ in ("val", "test")}
                rows = [r for sp_ in ("val", "test") for r in self.eval_rows(sc[sp_], sp_, name, s, self.n_boot)]
                self.save_unit(d, rows, res.curve, {**c, "seed": s, "best_epoch": res.best_epoch,
                                                    "best_val": res.best_val, "seconds": res.seconds}, sc)

    def final_summary(self, tag="final"):
        rows = [pd.read_csv(f) for f in (self.out / tag).glob("*/seed*/rows.csv")]
        return pd.concat(rows) if rows else pd.DataFrame(columns=RESULT_COLUMNS)

    def best_overall(self):
        df = self.final_summary()
        d = df[(df.split == "val") & (df.protocol == "hard_active") & (df.subset == "all") & (df.metric == "roc_auc")]
        name = d.groupby("model")["value"].mean().round(6).sort_values(ascending=False, kind="stable").index[0]
        meta = json.loads(next((self.out / "final" / name).glob("seed*/meta.json")).read_text())
        return name, {k: meta[k] for k in ("objective", "features", "encoder", "decoder", "hidden", "layers", "dropout", "lr")}

    def run_dot_ablation(self):
        df = self.final_summary()
        configs = []
        for objective in self.g["objectives"]:
            p, m = SELECT[objective]
            d = df[(df.split == "val") & (df.protocol == p) & (df.subset == "all") & (df.metric == m)
                   & df.model.str.startswith(f"gnn_{objective}_")]
            name = d.groupby("model")["value"].mean().round(6).sort_values(ascending=False, kind="stable").index[0]
            meta = json.loads(next((self.out / "final" / name).glob("seed*/meta.json")).read_text())
            configs.append({**{k: meta[k] for k in ("objective", "features", "encoder", "hidden", "layers", "dropout", "lr")},
                            "decoder": "dot"})
        self.run_final(configs, tag="ablation")

    def run_determinism(self):
        d = self.out / "determinism"
        if (d / "meta.json").exists():
            return
        name, c = self.best_overall()
        s = self.cfg["seed"]
        log(f"determinism re-run {name} seed {s}")
        res = self.fit(c, s)
        sc = {sp_: self.scores(res.model, c, sp_) for sp_ in ("val", "test")}
        ref = {sp_: dict(np.load(self.out / "final" / name / f"seed{s}" / f"scores_{sp_}.npz")) for sp_ in ("val", "test")}
        ref_rows = pd.read_csv(self.out / "final" / name / f"seed{s}" / "rows.csv")
        rows = []
        for sp_ in ("val", "test"):
            diff = max(float(np.max(np.abs(sc[sp_][k].astype(np.float32) - ref[sp_][k]))) for k in sc[sp_])
            new = pd.DataFrame(self.eval_rows(sc[sp_], sp_, name, s, 0, ("hard_active", "hard")))
            base = {"model": f"determinism:{name}", "split": sp_, "subset": "all", "seed": s, "ci_low": None, "ci_high": None}
            rows.append({**base, "protocol": "hard_active", "negative_type": "hard_active",
                         "metric": "max_abs_score_diff", "value": diff})
            for p, nt in (("hard_active", "hard_active"), ("hard", "hard")):
                a = new[(new.protocol == p) & (new.subset == "all") & (new.metric == "roc_auc")]["value"].iloc[0]
                b = ref_rows[(ref_rows.split == sp_) & (ref_rows.protocol == p) & (ref_rows.subset == "all")
                             & (ref_rows.metric == "roc_auc")]["value"].iloc[0]
                rows.append({**base, "protocol": p, "negative_type": nt, "metric": "roc_auc_abs_diff", "value": abs(a - b)})
        self.save_unit(d, rows, res.curve, {"model": name, "seed": s, "device": self.device,
                                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})

    def run_oof(self):
        """Out-of-fold GNN scores for the hybrid's training pairs (pair-focused LightGBM set).

        Fold k's pairs are scored by a GNN trained (on the train split only) without them, so the
        hybrid never sees in-sample GNN scores. Val/test GNN scores come from the final run."""
        name, c = self.best_overall()
        tr = self.splits["train"]
        pairs, _ = training_pairs(tr, 0.0, self.cfg["models"]["gnn"]["pair_focused_ratio"], active=True)
        u, v = pairs["u_idx"].values, pairs["v_idx"].values
        K = self.g["oof_folds"]
        for s in self.seeds:
            d = self.out / "oof" / f"seed{s}"
            if (d / "meta.json").exists():
                continue
            fold = np.random.default_rng([s, 4242]).permutation(len(u)) % K
            oof = np.full(len(u), np.nan)
            for k in range(K):
                excl = {(min(a, b), max(a, b)) for a, b in zip(u[fold == k], v[fold == k])}
                res = self.fit(c, s, exclude=excl)
                g = self.graphs[(c["features"], "train")].to(self.device)
                oof[fold == k] = res.model.score(g, u[fold == k], v[fold == k])
                log(f"oof {name} seed {s} fold {k}")
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "train_pairs.npz", u=u, v=v, score=oof.astype(np.float32))
            (d / "meta.json").write_text(json.dumps({"model": name, "seed": s, "folds": K, **c}, indent=2))

    def write_combined(self):
        def cat(pattern):
            fs = sorted(self.out.glob(pattern))
            return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame(columns=RESULT_COLUMNS)
        cat("grid/*/rows.csv").to_csv(self.out / "gnn_grid.csv", index=False)
        parts = [t for t in (cat("final/*/seed*/rows.csv"), cat("ablation/*/seed*/rows.csv")) if len(t)]
        (pd.concat(parts) if parts else pd.DataFrame(columns=RESULT_COLUMNS)).to_csv(self.out / "gnn.csv", index=False)
        cat("determinism/rows.csv").to_csv(self.out / "gnn_determinism.csv", index=False)
        import torch_geometric
        ver = {"python": platform.python_version(), "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
               "device": self.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "numpy": np.__version__, "pandas": pd.__version__, "dry_run": self.dry}
        (self.out / "versions.json").write_text(json.dumps(ver, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry", action="store_true", help="2 epochs, 1 seed, 1 config per cell, 20 bootstrap draws")
    ap.add_argument("--stages", nargs="*", default=["grid", "final", "ablation", "determinism", "oof"])
    args = ap.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = Runner(cfg, out, device, args.dry)
    log(f"device {device}; loaded in {time.time() - t0:.0f}s")
    if "grid" in args.stages:
        R.run_grid()
        R.write_combined()
    if "final" in args.stages:
        R.run_final(R.selected())
        R.write_combined()
    if "ablation" in args.stages:
        R.run_dot_ablation()
        R.write_combined()
    if "determinism" in args.stages:
        R.run_determinism()
        R.write_combined()
    if "oof" in args.stages:
        R.run_oof()
    R.write_combined()
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
