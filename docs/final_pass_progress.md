# Final pass progress (before the repo goes public)

Resume file. Python: `/opt/anaconda3/envs/linkpred/bin/python`, repo root, `PYTHONPATH=src`. Rule 8: new analyses on
validation first. Task list: the final-pass request (1 wording, 2 checks a-d, 3 references, 4 numbers + tests + push).
Reviewer notes that motivate it: `docs/reviewer_pass.md`.

## Done
- `git pull` (up to date at fceb919).

- Step 2a: no new compute; validation rows already in `results/final_comparison.csv`.
- Step 2b/2c/2d on validation: `PYTHONPATH=src python scripts/final_pass_checks.py --splits val` →
  `results/pf_logreg_exploratory.csv`, `final_pass_tests.csv`, `node2vec_sanity.csv`, `clustered_bootstrap.csv`.
  Val: pair-focused LR 0.831 vs pair-focused LightGBM 0.824 (diff -0.006, p 0.07) -> gain is the training pairs;
  clustered CIs ~1.8-2.0x wider; PF - AA stays significant; Jaccard - AA (hard_active) loses significance.

- Step 2 test: `python scripts/final_pass_checks.py --splits val test`. Test: pair-focused LR 0.808 vs LightGBM 0.799
  (LGBM - LR -0.008, p 0.016); clustered CIs ~1.9-2.1x wider; headline gains hold; Jaccard - AA loses significance
  on both metrics.

- Step 1 wording and step 3 references done in README.md and report/report.md (no [VERIFY] left).
- Step 4: new tokens in report/numbers.yaml; python scripts/check_numbers.py passes (264 occurrences).
- Decision entry "Final pass" in docs/decisions.md; make final-pass target.

- make test: 197 passed. Fresh clone: 147 passed, 50 skipped; number check passes in the clone.
- Committed locally as fa271ea (plus this progress update).

## In progress
- (none) Pushed as 3b76b68 (2026-10-28); origin/main matches local main.

## Next
- Nothing else in the final pass.
