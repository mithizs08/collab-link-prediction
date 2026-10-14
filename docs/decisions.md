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
