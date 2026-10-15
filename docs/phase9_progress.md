# Phase 9 progress (write-up)

Resume file for a fresh session. Python: `/opt/anaconda3/envs/linkpred/bin/python`, run from the repo root with
`PYTHONPATH=src`. Phase 9 uses only committed code, results files and figures (no data folder needed).

## Environment (checked 2026-10-15)
- `pandoc` is available (`/opt/anaconda3/bin/pandoc`); no PDF engine (no xelatex / pdflatex / wkhtmltopdf /
  weasyprint). The PDF is therefore not built here; the command is in `report/README_build.md`.

## Done
- Step 0, cross-field check (from committed files): the same-field / cross-field subsets restrict BOTH classes.
  `results/error_analysis.csv` `field_pair=*` `n_negatives` sum to the full activity-matched negative set
  (val 575 + 857 = 1,432; test 677 + 1,007 = 1,684, equal to `results/baselines.csv` hard_active `n_negatives`),
  and `scripts/phase8_errors.py` (lines ~166-174) filters positives and negatives by the pair's own field relation.
  Not positives-only; no local rerun needed for this question.

- Step 3: `python scripts/final_figures.py` → `figures/final/fig1..fig6` (Okabe-Ito, 6.3 in wide, checked visually).

- Step 1: `report/report.md` (about 3,700 words, 5 figures, 19 references all marked [VERIFY]); PDF build command in
  `report/README_build.md` (no PDF engine here).
- Step 2: `README.md` rewritten.
- Step 4: `report/numbers.yaml` + `scripts/check_numbers.py` (in `make test` via `tests/test_writeup.py`); passes.

## In progress
- `make test`; fresh-clone check; commit; push.

## Next
- Author checks every [VERIFY] reference; builds the PDF locally.
