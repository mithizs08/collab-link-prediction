# Decision log

One entry per decision, newest at the bottom. Format: **Decision**, **Alternatives**, **Reason**.

## 2026-10-14 — Project scaffold and pinned environment
- **Decision:** Python 3.11, `src/` layout (package `linkpred`), pinned `requirements.txt`
  around torch 2.4.1 / torch_geometric 2.6.1 with numpy 1.26.4.
- **Alternatives:** latest unpinned versions; numpy 2.x.
- **Reason:** reproducibility; numpy 1.26 avoids ABI friction with shap/lightgbm/PyG extensions
  at these versions. PyG's Node2Vec random-walk backend (pyg-lib / torch-cluster) is installed
  from the PyG wheel index because it must match the torch build.

## 2026-10-14 — Drop works with more than 25 authors
- **Decision:** `data.max_authors_per_work: 25`.
- **Alternatives:** keep all works; down-weight edges by 1/(n_authors-1).
- **Reason:** hyper-authored papers create large cliques that dominate degree and
  common-neighbour statistics without reflecting genuine collaboration ties. Threshold to be
  sanity-checked against the author-count distribution during EDA.

## 2026-10-14 — Predict only *new* links between already-seen authors
- **Decision:** positives for period T are pairs whose first co-authored work falls in T's window;
  both authors must appear in the graph before the cutoff (`require_both_nodes_seen: true`).
- **Alternatives:** include repeat collaborations; include cold-start authors.
- **Reason:** repeat links are trivially predictable from the existing edge, and cold-start
  authors have no graph signal, so both would blur the comparison between methods.

## 2026-10-14 — Results as long-format CSV with an enforcing writer
- **Decision:** single writer `linkpred.results.write_results` validates the schema
  `model, split, negative_type, seed, metric, value`.
- **Alternatives:** per-model ad hoc CSV/JSON.
- **Reason:** one schema makes cross-model aggregation trivial and makes "every reported number
  is in a results file" checkable.

## 2026-10-14 — torch-cluster (not pyg-lib) as Node2Vec backend; llvm-openmp for LightGBM
- **Decision:** install `torch-cluster` from the PyG wheel index; `conda install llvm-openmp`.
- **Alternatives:** pyg-lib; Homebrew `libomp`.
- **Reason:** on macOS + conda, the pyg-lib wheel links to the python.org framework Python and
  fails to load. LightGBM's macOS wheel needs `libomp.dylib`; installing it inside the conda env
  keeps the environment self-contained instead of depending on Homebrew.

## 2026-10-14 — Build from the existing OpenAlex snapshot; no re-pull
- **Decision:** read `uts-research-intelligence/data/raw/uts_works_2015_2025.json` (63,514 works,
  2015–2025) read-only via `data.snapshot_path`; stream it with `ijson`.
- **Alternatives:** re-pull all UTS works from the API with `fetch_openalex.py`.
- **Reason:** the snapshot has every required field except `doi`, and DOIs are recoverable from
  `primary_location` for 96.7% of kept works. Its authorships are capped at 100 by the list API,
  which only touches works already excluded by the 25-author cap. Streaming keeps peak memory low
  on an 8 GB machine. `fetch_openalex.py` (cursor paging, mailto, retries, page cache) is written
  and unit-tested against a fake session but has not been run against the API.
- UTS institution ID `I114017466` verified against `/institutions/I114017466` (display name
  "University of Technology Sydney", ROR `03f0f6041`, no child institutions in lineage).

## 2026-10-14 — Deduplication rules
- **Decision:** drop works with the same normalised DOI; then drop works with the same normalised
  title + year that share at least one author ID, only for titles of ≥ 5 words. The preferred copy
  is a non-preprint, then has an abstract, then has more authors, then the lowest work ID.
- **Alternatives:** DOI only; title + year without guards.
- **Reason:** inspecting the 1,497 unguarded title + year matches showed mostly true duplicates
  (preprint vs. published, Angewandte German/English editions, multi-journal consensus
  statements, copies without DOI) but also false ones on generic titles ("Letter from the
  editors" ×23, "Test" ×12, "Introduction" ×10). The 5-word guard removed those (1,367 remain).
  Preprints and published versions in *different* years are deliberately not merged: the
  collaboration was public from the earlier year.

## 2026-10-14 — Author cap counts the full byline
- **Decision:** `n_authors` = number of authorships including authors without an OpenAlex ID;
  works with `n_authors > 25` stay in `works.parquet` but produce no edges.
- **Reason:** the cap describes how a paper was written, not what OpenAlex resolved.

## 2026-10-14 — authors.parquet is descriptive only
- **Decision:** `authors.parquet` (first/last year, n_works, primary institution) aggregates all
  years and must never be used as a model feature.
- **Reason:** it contains future information. Per-period author features will be recomputed
  from works with year < cutoff. Likewise, UTS status in the audit is time-respecting ("had a UTS
  authorship before the cutoff").

## 2026-10-14 — Recommended split cutoffs (PENDING REVIEW, not yet in config)
- **Recommendation:** one-year label windows ending in the most recent data:
  - train: cutoff 2023, window 2023 (graph = works 2015–2022)
  - val: cutoff 2024, window 2024
  - test: cutoff 2025, window 2025
- **Eligible positives** (`report/data_audit/split_candidates.csv`, both authors in the graph
  before the cutoff):

  | period | all pairs both seen | ≥1 UTS author | UTS–UTS |
  |---|---|---|---|
  | train 2023 | 10,867 | 6,491 | 2,250 |
  | val 2024 | 10,894 | 6,779 | 2,229 |
  | test 2025 | 11,724 | 7,238 | 2,539 |

- **Reason:** (1) Every window has more than 2,000 positives even in the strictest (UTS–UTS)
  population, enough for stable AUC/AP: for an AUC near 0.9 with 1:1 negatives the standard
  error is about 0.005, so differences of a few hundredths between models are detectable. (2) Equal one-year horizons
  keep train, validation and test the same task. (3) Late cutoffs give the longest observed
  history (8 years before the train cutoff) and avoid the left-censored 2015–2016 period.
  (4) Yearly new-pair counts are stable from 2022 on (73–80k all-author, 5.7–6.5k UTS–UTS), so
  the three windows have similar difficulty.
- **Alternatives considered:** a two-year test window 2024–2025 (more positives: 4,109 UTS–UTS,
  but a different horizon from training); earlier cutoffs such as 2021/2022/2023 (keeps 2024–2025
  unused or needs a horizon mismatch, and gives less history).
- **Caveats:** 2025 may still be slightly incomplete in OpenAlex (pulled Aug 2026). If more
  training signal is needed, Phase 2 can stack earlier one-year training windows (cutoffs
  2019–2022) without touching val/test; that is a separate decision.
- **Open (for review):** the primary prediction population, UTS–UTS vs. all pairs with both
  authors seen (see `docs/data_audit.md`, caveat 2).
