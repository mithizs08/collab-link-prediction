# Phase 8 progress (explainability and error analysis; completes RQ3)

Resume file for a fresh session. Python: `/opt/anaconda3/envs/linkpred/bin/python` (conda env `linkpred`), run from the
repo root with `PYTHONPATH=src`. Rule 8: every analysis on validation first; test only for final figures.
The task list is the Phase 8 request (steps 1–7); decisions go in `docs/decisions.md`.

## Setup facts (checked 2026-10-15)
- `shap` 0.46.0 and `torch_geometric` 2.6.1 are installed; `captum` is not (GNNExplainer from PyG is used).
- The Colab GNN runs saved scores only (`data/interim/gnn/final/<model>/seed*/scores_*.npz`, `meta.json`), **no
  weights**: step 4 retrains on CPU from the saved configs.
- `data/processed/works.parquet` has no topic / field columns. The raw snapshot
  (`/Users/zareenshyma/Documents/uts-research-intelligence/data/raw/uts_works_2015_2025.json`, read-only) has
  OpenAlex `topics` with `field`; step 5 extracts them with a separate script (no existing table is modified).

## Done
- Steps 1–3 on validation: `PYTHONPATH=src python scripts/phase8_shap.py --split val` →
  `results/shap_importance.csv`, `results/activity_detector.csv`, `report/shap/*.csv`, `figures/shap_*_val.png`,
  `figures/activity_detector_val.png`. Findings so far: the headline model has 2-leaf trees → additive, all SHAP
  interactions 0; mean |SHAP| is not discrimination (constant features like `distance` on distance-2 pairs get large
  |SHAP|), so `shap_auc` (ROC-AUC of SHAP contributions as a score within a pair set) is reported too.
- Step 2 exploratory block order on validation: `python scripts/rq2_text_ablation.py --block-order --splits val`.

## Also done
- Steps 1–3 test repeat: `python scripts/phase8_shap.py --split test`; block order val+test:
  `python scripts/rq2_text_ablation.py --block-order --splits val test` → `results/rq2_block_order_exploratory*.csv`.
- Step 4: `python scripts/phase8_gnn_explain.py` (CPU retrain, weights cached in `data/interim/gnn/cpu_explain/`) →
  `results/gnn_explanations.csv`, `figures/gnn_explanations_val.png`. GNNExplainer masks were uninformative (near-
  uniform); integrated gradients + edge occlusion added. Key result: removing the shared-neighbour paths RAISES the
  pair-focused GNN's logit (shortcut seen directly); far/disconnected pairs get its top scores.
- Step 5 input: `python scripts/extract_work_fields.py` → `data/processed/work_fields.parquet` (git-ignored; rerun
  needed on a fresh machine), `report/fields/coverage.csv`.
- Step 5 validation: `python scripts/phase8_errors.py --split val` → `results/error_analysis.csv`,
  `figures/error_analysis_val.png`.
- Step 6 validation: `python scripts/phase8_calibration.py --splits val`. Family extended to 34 in decisions.md
  ("Phase 8 design") and `final_comparison.py` (FAMILY_SIZE = 34) BEFORE the test run.

- Step 5/6 test: `python scripts/phase8_calibration.py --splits val test`, `python scripts/phase8_errors.py --split
  test` (censoring rows: test has no later year). `python scripts/final_comparison.py --holm-only` → 34 tests.
- Step 7: `python scripts/phase8_report.py` → `docs/explainability.md`; `python scripts/final_summary.py` (RQ3 complete).
  Makefile target `make phase8` runs the whole phase.

## In progress
- (none) — Phase 8 committed and pushed as f7e3dbc; make test 196 passed; fresh clone 146 passed, 50 skipped.

## Next
- Nothing in Phase 8 after the push. Do not start Phase 9 without the project owner.
