# Who Collaborates Next? — Temporal Link Prediction on a University Co-authorship Network

Portfolio research project. Rigor and honesty matter more than headline numbers.

**Task:** given the co-authorship graph built from works published before a cutoff year,
predict which author pairs co-author for the first time in the following window.
**Models compared:** graph heuristics, feature-engineered LightGBM, node2vec, GraphSAGE, GAT.
**Evaluation:** leak-free temporal split, random and hard (2-hop) negatives reported separately.

## Layout
- `src/linkpred/` — package (`config.py`: config + `set_seed`; `results.py`: the only results writer; `cli.py`: pipeline stages)
- `src/linkpred/data/` — `build.py` (raw OpenAlex → `data/processed/*.parquet`), `audit.py` (→ `report/data_audit/`, `figures/audit_*`, `docs/data_audit.md`), `fetch_openalex.py` (optional re-pull)
- `data/processed/authors.parquet` aggregates all years: descriptive only, never a feature (rule 1)
- `configs/default.yaml` — all paths, split years, sampling ratios, seeds, hyperparameters
- `data/{raw,interim,processed}` — git-ignored
- `results/` — CSV outputs; `figures/`, `report/` — generated from `results/` only
- `docs/decisions.md` — dated decision log
- Pipeline: `make data splits baselines features train evaluate report`; tests: `make test`

## Standing rules (apply in every session)

1. **No future information.** Anything used to make a prediction for period T (graph edges,
   node features, degrees, embeddings, normalisation statistics, author metadata, negative
   candidates) must come only from works published before T's cutoff. Every new feature or
   model must be explicitly checked against this, ideally with a test that would fail on leakage.
2. **One source of randomness.** All randomness goes through `linkpred.config.set_seed`, driven
   by `seed` / `seeds` in the config. Learned models run with seeds `[0, 1, 2, 3, 4]`; report
   mean ± std across seeds.
3. **Results live in files.** All results are written to `results/` as CSV via
   `linkpred.results.write_results`, with columns
   `model, split, negative_type, seed, metric, value`. Never report a number (in chat, README,
   report, or figures) that isn't in a results file.
4. **Honest reporting.** If a simple baseline matches or beats a complex model, say so plainly.
   Never tune on the test set: hyperparameters, early stopping, thresholds and model selection
   use the validation split only. Test is evaluated once per final configuration.
5. **Phase gate.** Every phase ends with `make test` passing and a short summary covering:
   what was done, what was decided and why, and any problems found.
6. **Document decisions.** Explain non-obvious decisions in code comments and in
   `docs/decisions.md`, one dated entry per decision (format: date, decision, alternatives,
   reason).
