"""Data audit: yearly counts, degree distribution, missingness, split candidates.

Reads data/processed/*.parquet and the cleaning log; writes
  report/data_audit/*.csv   every number quoted in docs/data_audit.md
  figures/audit_*.png
  docs/data_audit.md        (only the block between the GENERATED markers)

The audit is descriptive and uses all years. Nothing here feeds a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
YEAR_COLS = {"year", "cutoff", "window_end"}

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- loading

def load(processed: Path) -> dict[str, pd.DataFrame]:
    return {
        "works": pd.read_parquet(processed / "works.parquet",
                                 columns=["work_id", "year", "doi", "title", "type",
                                          "n_authors", "has_abstract"]),
        "auth": pd.read_parquet(processed / "authorships.parquet"),
        "authors": pd.read_parquet(processed / "authors.parquet"),
        "edges": pd.read_parquet(processed / "edges.parquet"),
    }


def uts_first_year(auth: pd.DataFrame, works: pd.DataFrame) -> pd.Series:
    """Year each author first appears with a UTS affiliation (time-respecting UTS status)."""
    a = auth[auth["is_uts"]].merge(works[["work_id", "year"]], on="work_id")
    return a.groupby("author_id")["year"].min()


def pair_first_year(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.groupby(["author_u", "author_v"], as_index=False)["year"].min()


# ---------------------------------------------------------------- tables

def yearly_table(d: dict, max_authors: int) -> pd.DataFrame:
    works, auth, edges = d["works"], d["auth"], d["edges"]
    aw = auth.merge(works[["work_id", "year"]], on="work_id")
    ufy = uts_first_year(auth, works)
    first_seen = aw.groupby("author_id")["year"].min()
    fy = pair_first_year(edges)
    fy["uts_uts"] = (
        fy["author_u"].map(ufy).le(fy["year"]) & fy["author_v"].map(ufy).le(fy["year"])
    )

    t = pd.DataFrame(index=sorted(works["year"].unique()))
    t.index.name = "year"
    t["works"] = works.groupby("year").size()
    t["works_over_cap"] = works[works["n_authors"] > max_authors].groupby("year").size()
    t["share_over_cap"] = t["works_over_cap"] / t["works"]
    t["share_with_abstract"] = works.groupby("year")["has_abstract"].mean()
    t["authorships"] = aw.groupby("year").size()
    t["active_authors"] = aw.groupby("year")["author_id"].nunique()
    t["active_uts_authors"] = aw[aw["is_uts"]].groupby("year")["author_id"].nunique()
    t["new_authors"] = first_seen.value_counts()
    t["edge_rows"] = edges.groupby("year").size()
    t["distinct_pairs"] = edges.drop_duplicates(["author_u", "author_v", "year"]).groupby("year").size()
    t["new_pairs"] = fy.groupby("year").size()
    t["new_pairs_uts_uts"] = fy[fy["uts_uts"]].groupby("year").size()
    return t.fillna(0).reset_index()


def split_candidates(d: dict, first_cutoff: int = 2017) -> pd.DataFrame:
    """Eligible positives for every (cutoff, window length).

    Positive = pair whose FIRST co-authored work falls in [cutoff, cutoff+L-1]
    with both authors already in the graph (edges with year < cutoff).
    UTS-UTS additionally needs both authors to have had a UTS authorship
    before the cutoff - i.e. UTS status is itself computed leak-free.
    """
    works, auth, edges = d["works"], d["auth"], d["edges"]
    ufy = uts_first_year(auth, works)
    fy = pair_first_year(edges)
    last = int(works["year"].max())
    rows = []
    for c in range(first_cutoff, last + 1):
        before = edges[edges["year"] < c]
        seen = pd.Index(pd.unique(before[["author_u", "author_v"]].values.ravel()))
        uts_seen = seen[seen.map(ufy).fillna(9999) < c]
        for L in (1, 2, 3):
            if c + L - 1 > last:
                continue
            win = fy[fy["year"].between(c, c + L - 1)]
            u_seen, v_seen = win["author_u"].isin(seen), win["author_v"].isin(seen)
            u_uts, v_uts = win["author_u"].isin(uts_seen), win["author_v"].isin(uts_seen)
            rows.append({
                "cutoff": c, "window_end": c + L - 1, "window_years": L,
                "graph_nodes": len(seen), "graph_edges_distinct": len(before.drop_duplicates(["author_u", "author_v"])),
                "uts_nodes": len(uts_seen),
                "new_pairs_in_window": len(win),
                "pos_both_seen": int((u_seen & v_seen).sum()),
                "pos_uts_uts": int((u_uts & v_uts).sum()),
                "pos_uts_any_both_seen": int((u_seen & v_seen & (u_uts | v_uts)).sum()),
            })
    out = pd.DataFrame(rows)
    out["share_both_seen"] = out["pos_both_seen"] / out["new_pairs_in_window"]
    return out


def degree_table(edges: pd.DataFrame, nodes: pd.Series | None = None) -> pd.DataFrame:
    e = edges.drop_duplicates(["author_u", "author_v"])
    if nodes is not None:
        e = e[e["author_u"].isin(nodes) & e["author_v"].isin(nodes)]
    deg = pd.concat([e["author_u"], e["author_v"]]).value_counts()
    return deg.value_counts().sort_index().rename_axis("degree").rename("n_authors").reset_index()


def graph_summary(edges: pd.DataFrame, nodes: pd.Series | None = None) -> dict:
    e = edges.drop_duplicates(["author_u", "author_v"])
    if nodes is not None:
        e = e[e["author_u"].isin(nodes) & e["author_v"].isin(nodes)]
    ids = pd.Index(pd.unique(e[["author_u", "author_v"]].values.ravel()))
    u, v = ids.get_indexer(e["author_u"]), ids.get_indexer(e["author_v"])
    m = coo_matrix((np.ones(len(e)), (u, v)), shape=(len(ids), len(ids)))
    n_comp, lab = connected_components(m, directed=False)
    deg = np.bincount(np.concatenate([u, v]), minlength=len(ids))
    return {
        "nodes": len(ids), "edges": len(e), "components": int(n_comp),
        "giant_component_share": float(np.bincount(lab).max() / len(ids)),
        "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)),
        "max_degree": int(deg.max()),
        "density": float(2 * len(e) / (len(ids) * (len(ids) - 1))),
    }


def missingness_table(d: dict, log: dict) -> pd.DataFrame:
    works, auth = d["works"], d["auth"]
    n_auth_total = len(auth) + log["authorships_without_author_id_dropped"]
    no_inst = auth["institution_ids"].map(len) == 0
    rows = [
        ("work", "publication_year", log["missing_year_dropped"], log["raw_works"], "dropped"),
        ("work", "doi", int(works["doi"].isna().sum()), len(works), "kept; title+year dedup only"),
        ("work", "title (empty)", int((works["title"] == "").sum()), len(works), "kept"),
        ("work", "abstract", int((~works["has_abstract"]).sum()), len(works), "kept; has_abstract=False"),
        ("authorship", "author_id", log["authorships_without_author_id_dropped"], n_auth_total, "dropped"),
        ("authorship", "institution_ids (empty)", int(no_inst.sum()), len(auth), "kept"),
        ("work", "no UTS-affiliated identified author", log["works_without_uts_authorship"], len(works), "kept"),
    ]
    t = pd.DataFrame(rows, columns=["level", "field", "n_missing", "n_total", "handling"])
    t["share_missing"] = t["n_missing"] / t["n_total"]
    return t


# ---------------------------------------------------------------- figures

def _style(ax, title, ylabel, xlabel="Publication year"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _fig(w=7, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _lines(ax, x, series: dict):
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=label)
        ax.annotate(label, (x.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_xticks(x)
    ax.set_xlim(x.min() - 0.5, x.max() + 2.2)
    ax.set_ylim(bottom=0)


def make_figures(t: pd.DataFrame, deg_all: pd.DataFrame, deg_uts: pd.DataFrame, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = []
    x = t["year"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, facecolor=SURFACE)
        plt.close(fig)
        names.append(name)

    fig, ax = _fig()
    ax.bar(x, t["works"], color=SERIES[0], width=0.7)
    _style(ax, "Works per year (after cleaning)", "Works")
    ax.set_xticks(x)
    save(fig, "audit_works_per_year.png")

    fig, ax = _fig()
    _lines(ax, x, {"All authors": t["active_authors"], "UTS-affiliated": t["active_uts_authors"]})
    _style(ax, "Active authors per year", "Distinct authors")
    save(fig, "audit_authors_per_year.png")

    # Two populations differ ~10x in scale -> small multiples, not a shared axis.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, col, title, c in ((axes[0], "new_pairs", "New co-author pairs: all authors", SERIES[0]),
                              (axes[1], "new_pairs_uts_uts", "New co-author pairs: UTS–UTS", SERIES[1])):
        ax.bar(x, t[col], color=c, width=0.7)
        _style(ax, title, "Pairs linking for the first time")
        ax.set_xticks(x)
        ax.tick_params(axis="x", rotation=45)
    fig.text(0.01, 0.005, "2015–16 counts include pre-2015 collaborations (data start is left-censored).",
             fontsize=7, color=INK_2)
    save(fig, "audit_new_edges_per_year.png")

    fig, ax = _fig(6, 4)
    for i, (label, dd) in enumerate((("All authors", deg_all), ("UTS–UTS subgraph", deg_uts))):
        ax.scatter(dd["degree"], dd["n_authors"], s=10, color=SERIES[i], label=label,
                   edgecolors=SURFACE, linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Degree distribution (all years, log–log)", "Number of authors", "Degree (distinct co-authors)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, "audit_degree_loglog.png")

    fig, ax = _fig()
    ax.bar(x, 100 * t["share_over_cap"], color=SERIES[0], width=0.7)
    _style(ax, "Works above the author cap (excluded from edges)", "% of works")
    ax.set_xticks(x)
    save(fig, "audit_share_over_cap.png")
    return names


# ---------------------------------------------------------------- markdown

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and not float(v).is_integer():
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int, np.integer, float, np.floating)):
                # Years are labels, not quantities: no thousands separator.
                cells.append(str(int(v)) if c in YEAR_COLS else f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def configured_splits(cands: pd.DataFrame, splits_cfg: dict) -> pd.DataFrame:
    """Candidate rows for the splits actually written in the config."""
    rows = []
    for name in ("train", "val", "test"):
        s = splits_cfg.get(name) or {}
        if s.get("cutoff") is None:
            continue
        r = cands[(cands["cutoff"] == s["cutoff"]) & (cands["window_end"] == s["window_end"])]
        if len(r):
            rows.append({"split": name, "graph_years": f"<{s['cutoff']}",
                         "label_window": f"{s['cutoff']}-{s['window_end']}", **r.iloc[0].to_dict()})
    cols = ["split", "graph_years", "label_window", "graph_nodes", "graph_edges_distinct", "uts_nodes",
            "pos_uts_uts", "pos_uts_any_both_seen", "pos_both_seen", "new_pairs_in_window"]
    out = pd.DataFrame(rows, columns=cols)
    num = cols[3:]
    out[num] = out[num].astype(int)
    return out


def render_markdown(log, t, miss, cands, g_all, g_uts, figs, conf=None) -> str:
    L = log
    steps = pd.DataFrame([
        ("Raw works in snapshot", L["raw_works"], ""),
        ("Missing publication_year", L["missing_year_dropped"], "dropped"),
        ("Excluded types (" + ", ".join(f"{k} {v}" for k, v in L.get("excluded_type_counts", {}).items()) + ")",
         L.get("excluded_type_dropped", 0), "dropped"),
        ("Duplicate OpenAlex IDs", L["duplicate_work_ids"], "dropped"),
        ("Duplicate DOI", L["duplicate_doi"], "dropped (preferred copy kept)"),
        ("Duplicate normalised title + year (shared author, ≥5-word title)", L["duplicate_title_year"], "dropped"),
        ("Works kept", L["works"], ""),
        (f"Works with > {L['max_authors_per_work']} authors", L["hyper_works"], "kept in works, no edges"),
        ("  of which authorship list capped at 100 by the API", L["hyper_works_truncated_at_100"], ""),
        ("Would-be edge rows from hyper-authored works", L["hyper_would_be_edges"], "not created (lower bound)"),
        ("Authorships without author ID", L["authorships_without_author_id_dropped"], "dropped"),
        ("  works affected", L["works_affected_by_missing_author_id"], ""),
        ("Self-loops (same author twice on a byline)", L["self_loops_removed"], "removed"),
        ("Duplicate (pair, work) rows", L["duplicate_pair_work_rows_removed"], "removed"),
    ], columns=["step", "count", "handling"])
    out = [
        BEGIN,
        "_Generated by `linkpred.data.audit` from `report/data_audit/`; do not edit by hand._",
        "",
        "## Final dataset",
        _md_table(pd.DataFrame([
            ("works", L["works"]), ("authorships (identified)", L["authorships"]),
            ("authors", L["authors"]), ("UTS-affiliated authors (ever)", L["uts_authors"]),
            ("edge rows (pair × work)", L["edge_rows"]), ("distinct co-author pairs", L["distinct_pairs"]),
        ], columns=["table", "rows"])),
        "",
        "## Cleaning steps",
        _md_table(steps),
        "",
        "## Author disambiguation (flagged, not merged)",
        "Same normalised display name with more than one author ID at the same institution.",
        "",
        _md_table(pd.DataFrame([
            ("all institutions", L["disambig_groups"], L["disambig_author_ids"]),
            ("UTS only", L["disambig_groups_uts"], L["disambig_author_ids_uts"]),
        ], columns=["scope", "name×institution groups", "author IDs involved"])),
        "",
        "## Missingness",
        _md_table(miss),
        "",
        "## Yearly counts",
        _md_table(t),
        "",
        "## Graph summary (all years, distinct pairs)",
        _md_table(pd.DataFrame([{"graph": "all authors", **g_all}, {"graph": "UTS–UTS", **g_uts}]), "{:.4g}"),
        "",
        "## Configured splits (target: UTS–UTS pairs)",
        "`pos_uts_uts` are the prediction targets: pairs first co-authoring in the label window, both "
        "in the graph before the cutoff and both with a UTS authorship before the cutoff. The other "
        "positive counts are context only.",
        "",
        _md_table(conf) if conf is not None and len(conf) else "_No splits configured._",
        "",
        "## Split candidates",
        "Positives = pairs first co-authoring inside the window whose two authors are both already "
        "in the graph before the cutoff. `pos_uts_uts` also requires both to have had a UTS "
        "affiliation before the cutoff.",
        "",
        _md_table(cands[["cutoff", "window_end", "graph_nodes", "uts_nodes", "new_pairs_in_window",
                         "pos_both_seen", "share_both_seen", "pos_uts_any_both_seen", "pos_uts_uts"]]),
        "",
        "## Figures",
        *[f"![{f}](../figures/{f})" for f in figs],
        END,
    ]
    return "\n".join(out)


def write_markdown(path: Path, generated: str, title: str = "Data audit") -> None:
    if path.exists():
        text = path.read_text()
        if BEGIN in text and END in text:
            text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: generated, text, flags=re.S)
            path.write_text(text)
            return
    path.write_text(f"# {title}\n\n" + generated + "\n")


# ---------------------------------------------------------------- pipeline

def run(cfg: dict) -> dict:
    paths = cfg["paths"]
    out = Path(paths["report"]) / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    log = json.loads((out / "cleaning_log.json").read_text())
    d = load(Path(paths["processed"]))
    max_authors = cfg["data"]["max_authors_per_work"]

    t = yearly_table(d, max_authors)
    cands = split_candidates(d)
    miss = missingness_table(d, log)
    uts_ever = d["authors"].loc[d["authors"]["n_uts_works"] > 0, "author_id"]
    deg_all, deg_uts = degree_table(d["edges"]), degree_table(d["edges"], uts_ever)
    g_all, g_uts = graph_summary(d["edges"]), graph_summary(d["edges"], uts_ever)
    types = d["works"]["type"].value_counts().rename_axis("type").rename("works").reset_index()

    t.to_csv(out / "yearly_counts.csv", index=False)
    cands.to_csv(out / "split_candidates.csv", index=False)
    miss.to_csv(out / "missingness.csv", index=False)
    deg_all.to_csv(out / "degree_distribution_all.csv", index=False)
    deg_uts.to_csv(out / "degree_distribution_uts.csv", index=False)
    types.to_csv(out / "work_types.csv", index=False)
    (out / "graph_summary.json").write_text(json.dumps({"all": g_all, "uts_uts": g_uts}, indent=2) + "\n")

    conf = configured_splits(cands, cfg["splits"])
    conf.to_csv(out / "configured_splits.csv", index=False)

    figs = make_figures(t, deg_all, deg_uts, Path(paths["figures"]))
    write_markdown(Path("docs/data_audit.md"), render_markdown(log, t, miss, cands, g_all, g_uts, figs, conf))
    return {"yearly": t, "candidates": cands, "configured": conf, "missingness": miss, "graph": {"all": g_all, "uts_uts": g_uts}}


def main(cfg: dict) -> None:
    run(cfg)
