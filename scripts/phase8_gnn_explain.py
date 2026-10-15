"""Phase 8, step 4: GNN explanations on validation predictions.

  python scripts/phase8_gnn_explain.py

The Colab runs saved scores but no weights, so the two models are retrained on CPU from their saved configs
(data/interim/gnn/final/<model>/seed0/meta.json), seed 0:
  gnn_pair_focused_sage_structural  the best GNN, retrained on its ORIGINAL training set (pair-focused, positives at
                                    any distance: models.gnn.pair_focused_d2_only = false for this run only), so the
                                    model explained is the shortcut-trained one that is reported;
  gnn_general_sage_text             the best general-objective GNN on validation hard.
Fidelity to the Colab models is checked (Spearman of validation scores against the saved Colab scores, validation
macro hard_active of both) and reported, since CPU and GPU training are not bit-identical.

Explanations on the 2-hop subgraph around each pair (exact for 2-layer message passing):
  * GNNExplainer (PyG; edge mask + node-feature mask, 200 epochs) -- reported, but uninformative here (see docs);
  * integrated gradients on node features (baseline = training-mean node, 64 steps; completeness error reported):
    the logit split into the pair's OWN node features vs their neighbours' features, and by feature;
  * edge occlusion: logit change when every u-w-v path through a common neighbour is removed. 8 validation predictions per model:
  2 confident true positives / 2 confident false positives / 2 missed positives  (hard_active pairs)
  2 far pairs: the highest-scored validation random negatives at distance >= 3 (non-collaborators with no common
    neighbour; a high score there is the distance shortcut, results/gnn_shortcut_check.csv)
Per explanation: share of edge-mask mass on edges through common neighbours (the u-w-v paths a link predictor
should use), share on the two endpoints' own edges, and node-feature mask by feature.

Outputs: results/gnn_explanations.csv, figures/gnn_explanations_val.png, data/interim/gnn/cpu_explain/*.pt (weights).
Explanations show what a model uses, not why collaborations happen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred.config import load_config  # noqa: E402
from linkpred.data.audit import GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from linkpred.eval.metrics import roc_auc  # noqa: E402
from linkpred.eval.protocols import evaluate  # noqa: E402
from linkpred.models import gnn as G  # noqa: E402
from linkpred.results import write_results  # noqa: E402
from gnn_analysis import load_scores  # noqa: E402
from run_gnn import Runner  # noqa: E402

MODELS = ("gnn_pair_focused_sage_structural", "gnn_general_sage_text")
STRUCT_NAMES = ["log1p_degree", "career_age", "log1p_recent_works", "is_uts"]
CASES = ("true_positive", "false_positive", "missed_positive", "far_pair")


class PairModel(torch.nn.Module):
    """LinkPredictor as f(x, edge_index, edge_label_index) -> logit, the signature torch_geometric.explain expects."""

    def __init__(self, lp):
        super().__init__()
        self.lp = lp

    def forward(self, x, edge_index, edge_label_index):
        z = self.lp.encoder(x, edge_index)
        return self.lp.decoder(z[edge_label_index[0]], z[edge_label_index[1]]).view(-1)


def row(model, protocol, subset, metric, value):
    return {"model": model, "split": "val", "protocol": protocol, "subset": subset,
            "negative_type": {"hard_active": "hard_active", "random": "random", "hard": "hard"}[protocol],
            "seed": 0, "metric": metric, "value": float(value), "ci_low": None, "ci_high": None}


def pick(scores, split, k=2):
    """Pair choices on validation (indices into the saved pair tables), from the model's own scores."""
    pos, d2 = split["positives"], (split["positives"]["distance"] == 2).values
    sp, sn = scores["positives"], scores["neg_hard_active"]
    ip = np.flatnonzero(d2)
    tp = ip[np.argsort(-sp[ip])[:k]]
    fn = ip[np.argsort(sp[ip])[:k]]
    fp = np.argsort(-sn)[:k]
    far = np.flatnonzero(np.asarray(split["neg_random_distance"]) >= 3)
    farp = far[np.argsort(-scores["neg_random"][far])[:k]]
    ref = np.r_[sp[ip], sn]           # hard_active pair scores: percentile reference
    out = []
    for case, tab, idx, protocol, lab in (("true_positive", "positives", tp, "hard_active", 1),
                                          ("false_positive", "neg_hard_active", fp, "hard_active", 0),
                                          ("missed_positive", "positives", fn, "hard_active", 1),
                                          ("far_pair", "neg_random", farp, "random", 0)):
        for j, i in enumerate(idx):
            u, v = int(split[tab]["u_idx"].iloc[i]), int(split[tab]["v_idx"].iloc[i])
            s = scores[tab][i]
            out.append({"case": case, "k": j, "u": u, "v": v, "label": lab, "protocol": protocol, "score": s,
                        "pct": float((ref < s).mean())})
    return out


def explain_pair(lp, g, A, u, v, layers, seed, feat_names):
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    nodes, ei, mapping, _ = k_hop_subgraph(torch.tensor([u, v]), layers, g.edge_index, relabel_nodes=True,
                                           num_nodes=g.n)
    x = g.x[nodes]
    torch.manual_seed(seed)
    model = PairModel(lp).eval()
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200), explanation_type="model",
                          edge_mask_type="object", node_mask_type="attributes",
                          model_config=dict(mode="binary_classification", task_level="edge", return_type="raw"))
    eli = mapping.view(2, 1)
    with torch.no_grad():
        full_logit = model(g.x, g.edge_index, torch.tensor([[u], [v]])).item()
        sub_logit = model(x, ei, eli).item()
    exp = explainer(x, ei, edge_label_index=eli)
    em = exp.edge_mask.detach().numpy()
    nm = exp.node_mask.detach().numpy()               # (n_sub, n_features)
    src, dst = ei.numpy()
    gl = nodes.numpy()
    uu, vv = int(mapping[0]), int(mapping[1])
    nbr_u = set(dst[src == uu]) | set(src[dst == uu])
    nbr_v = set(dst[src == vv]) | set(src[dst == vv])
    common = nbr_u & nbr_v
    on_path = np.array([(a in (uu, vv) and b in common) or (b in (uu, vv) and a in common) for a, b in zip(src, dst)])
    endpoint = np.array([a in (uu, vv) or b in (uu, vv) for a, b in zip(src, dst)]) & ~on_path
    tot = em.sum() if em.sum() > 0 else 1.0
    fimp = nm.mean(axis=0)            # per input dimension; the 384 text dimensions are AVERAGED, not summed
    names = feat_names + (["text_per_dim"] if nm.shape[1] > len(feat_names) else [])
    fv = list(fimp[:len(feat_names)]) + ([fimp[len(feat_names):].mean()] if nm.shape[1] > len(feat_names) else [])
    fv = np.array(fv) / max(np.sum(fv), 1e-12)
    # Integrated gradients on node features (baseline 0 = the training-mean node after standardisation), 64 steps.
    steps = 64
    xs = x.detach()
    grads = torch.zeros_like(xs)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (a * xs).requires_grad_(True)
        out = model(xi, ei, eli).sum()
        grads += torch.autograd.grad(out, xi)[0]
    ig = (xs * grads / steps).numpy()                          # (n_sub, n_features)
    with torch.no_grad():
        base = model(torch.zeros_like(xs), ei, eli).item()
    completeness = ig.sum() - (sub_logit - base)
    own = ig[[uu, vv]].sum()
    nbr = ig.sum() - own
    ig_feat = ig.sum(axis=0)
    ig_names = feat_names + (["text"] if ig.shape[1] > len(feat_names) else [])
    ig_vals = list(ig_feat[:len(feat_names)]) + ([ig_feat[len(feat_names):].sum()] if ig.shape[1] > len(feat_names) else [])
    # Edge occlusion: remove every u-w and w-v edge through a common neighbour w (both directions).
    keep = ~on_path
    with torch.no_grad():
        occl = model(x, ei[:, torch.from_numpy(keep)], eli).item()
    top = np.argsort(-em)[:15]
    edges = [(int(gl[src[t]]), int(gl[dst[t]]), float(em[t])) for t in top]
    return {"n_sub_nodes": len(gl), "n_sub_edges": int(ei.shape[1]), "n_common": len(common),
            "mask_share_common_neighbour_paths": float(em[on_path].sum() / tot),
            "mask_share_endpoint_edges": float(em[endpoint].sum() / tot),
            "logit_full_graph": full_logit, "logit_subgraph": sub_logit,
            "feature_mask": dict(zip(names, fv)), "top_edges": edges, "common": [int(gl[c]) for c in common],
            "ig_logit_minus_baseline": sub_logit - base, "ig_completeness_error": float(completeness),
            "ig_share_pair_own_features": float(own / (abs(own) + abs(nbr) + 1e-12)),
            "ig_pair_own": float(own), "ig_neighbours": float(nbr), "ig_feature": dict(zip(ig_names, ig_vals)),
            "logit_change_without_common_neighbour_paths": occl - sub_logit}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    res, fig = Path(cfg["paths"]["results"]), Path(cfg["paths"]["figures"])
    wdir = Path(cfg["paths"]["interim"]) / "gnn" / "cpu_explain"
    wdir.mkdir(parents=True, exist_ok=True)
    R = Runner(cfg, wdir, "cpu", dry=False)
    R.g = {**R.g, "pair_focused_d2_only": False}       # reproduce the saved (shortcut-trained) Colab model
    val = R.splits["val"]
    from linkpred.data.splits import positive_distances
    val["neg_random_distance"] = positive_distances(val["A"], val["neg_random"])
    rows, drawn = [], {}
    for name in MODELS:
        c = json.loads((Path("data/interim/gnn/final") / name / "seed0" / "meta.json").read_text())
        wf = wdir / f"{name}_seed0.pt"
        lp = G.LinkPredictor(c["encoder"], c["decoder"], R.graphs[(c["features"], "train")].x.shape[1], c["hidden"],
                             c["layers"], c["dropout"], R.g["heads"])
        if wf.exists():
            lp.load_state_dict(torch.load(wf))
        else:
            lp = R.fit(c, 0).model
            torch.save(lp.state_dict(), wf)
        lp.eval()
        sc = R.scores(lp, c, "val")
        colab = load_scores(Path("data/interim/gnn/final") / name / "seed0", "val")
        rho = np.mean([spearmanr(sc[k], colab[k]).statistic for k in sc])
        rows.append(row(name, "hard_active", "fidelity=cpu_retrain_vs_colab", "mean_spearman_val_scores", rho))
        for tag, s_ in (("cpu_retrain", sc), ("colab", colab)):
            r, _ = evaluate(None, val, "val", name, 0, cfg["seed"], 0, protocols=("hard_active",),
                            k_full=cfg["negatives"]["ranking_candidates"], precomputed=s_)
            v_ = next(x["value"] for x in r if x["subset"] == "macro" and x["metric"] == "roc_auc")
            rows.append(row(name, "hard_active", f"fidelity={tag}", "macro_roc_auc", v_))
            # shortcut check on validation: far random negatives vs distance-2 positives (> 0.5 = shortcut)
            d2 = (val["positives"]["distance"] == 2).values
            far = val["neg_random_distance"] >= 3
            rows.append(row(name, "random", f"fidelity={tag}", "shortcut_far_negatives_vs_distance2_positives_auc",
                            roc_auc(s_["neg_random"][far], s_["positives"][d2])))
        g = R.graphs[(c["features"], "val")]
        for p in pick(sc, val):
            e = explain_pair(lp, g, val["A"], p["u"], p["v"], c["layers"], cfg["seed"], STRUCT_NAMES)
            sub = f"case={p['case']}|k={p['k']}"
            dist = 2.0 if p["protocol"] == "hard_active" else float(val["neg_random_distance"][
                np.flatnonzero((val["neg_random"]["u_idx"].values == p["u"]) & (val["neg_random"]["v_idx"].values == p["v"]))[0]])
            dist = 99.0 if not np.isfinite(dist) else dist        # 99 = disconnected in the input graph
            for metric, value in (("score_logit", p["score"]), ("score_percentile_in_hard_active", p["pct"]),
                                  ("label", p["label"]), ("distance", dist), ("n_common_neighbours", e["n_common"]),
                                  ("subgraph_nodes", e["n_sub_nodes"]), ("subgraph_edges", e["n_sub_edges"]),
                                  ("mask_share_common_neighbour_paths", e["mask_share_common_neighbour_paths"]),
                                  ("mask_share_endpoint_edges", e["mask_share_endpoint_edges"]),
                                  ("logit_subgraph_minus_full", e["logit_subgraph"] - e["logit_full_graph"])):
                rows.append(row(name, p["protocol"], sub, metric, value))
            rows += [row(name, p["protocol"], sub, f"gnnexplainer_feature_mask_{f}", v_) for f, v_ in e["feature_mask"].items()]
            for metric in ("ig_logit_minus_baseline", "ig_completeness_error", "ig_pair_own", "ig_neighbours",
                           "logit_change_without_common_neighbour_paths"):
                rows.append(row(name, p["protocol"], sub, metric, e[metric]))
            rows += [row(name, p["protocol"], sub, f"ig_feature_{f}", v_) for f, v_ in e["ig_feature"].items()]
            drawn[(name, p["case"], p["k"])] = (p, e)
            print(name, sub, f"score {p['score']:.2f} pct {p['pct']:.2f} common {e['n_common']} "
                  f"path-share {e['mask_share_common_neighbour_paths']:.2f} IG own {e['ig_pair_own']:.2f} "
                  f"nbr {e['ig_neighbours']:.2f} occl {e['logit_change_without_common_neighbour_paths']:.2f} "
                  f"compl {e['ig_completeness_error']:.3f}", flush=True)
    write_results(rows, res / "gnn_explanations.csv", append=False)
    draw(drawn, fig / "gnn_explanations_val.png")


def draw(drawn, path):
    f_, axes = plt.subplots(2, 8, figsize=(18, 5.4), facecolor=SURFACE)
    for r_, name in enumerate(MODELS):
        for c_, (case, k) in enumerate([(c, k) for c in CASES for k in (0, 1)]):
            ax = axes[r_, c_]
            ax.set_facecolor(SURFACE)
            ax.axis("off")
            p, e = drawn[(name, case, k)]
            Gx = nx.Graph()
            for a, b, w in e["top_edges"]:
                Gx.add_edge(a, b, w=w)
            Gx.add_nodes_from([p["u"], p["v"]])
            pos = nx.spring_layout(Gx, seed=0)
            ws = np.array([Gx[a][b]["w"] for a, b in Gx.edges()]) if Gx.number_of_edges() else np.array([])
            nx.draw_networkx_edges(Gx, pos, ax=ax, width=0.5 + 3 * (ws / ws.max() if len(ws) else ws), edge_color=INK_2,
                                   alpha=0.7)
            col = [SERIES[1] if n in (p["u"], p["v"]) else (SERIES[2] if n in e["common"] else GRID) for n in Gx.nodes()]
            nx.draw_networkx_nodes(Gx, pos, ax=ax, node_size=28, node_color=col, linewidths=0)
            ax.plot([pos[p["u"]][0], pos[p["v"]][0]], [pos[p["u"]][1], pos[p["v"]][1]], color=SERIES[1], linestyle=":",
                    linewidth=1)
            ax.set_title(f"{case.replace('_', ' ')}\nscore pct {p['pct']:.2f} · common {e['n_common']}\n"
                         f"Δlogit w/o shared paths {e['logit_change_without_common_neighbour_paths']:+.2f}",
                         fontsize=7, color=INK)
        axes[r_, 0].text(-0.15, 0.5, "pair-focused GNN" if "pair" in name else "general GNN", rotation=90,
                         transform=axes[r_, 0].transAxes, va="center", fontsize=9, color=INK)
    f_.suptitle("GNNExplainer on validation pairs: top-15 edges by mask (orange = the pair, green = common neighbours, "
                "dotted = predicted link). Δlogit = change when every u–w–v path through a common neighbour is removed (occlusion).", x=0.01, ha="left",
                fontsize=10, color=INK)
    f_.tight_layout()
    f_.savefig(path, dpi=150)
    plt.close(f_)


if __name__ == "__main__":
    main()
