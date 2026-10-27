# Consistency check: README and report against the pair-focused logistic regression result

Date: 2026-10-28. Report only; nothing in `README.md`, `report/report.md` or any other file was changed.

**What is being checked.** The final pass found that a logistic regression trained on the same distance-2,
activity-matched pairs as the pair-focused LightGBM scores at least as well. The numbers below are from
`results/final_comparison.csv`, `results/pf_logreg_exploratory.csv` and `results/final_pass_tests.csv`.

- On macro `hard_active` (test), the logistic regression scores 0.808 and the LightGBM 0.799. LightGBM minus logistic
  regression is -0.008 [-0.015, -0.002], with a raw p of 0.016.
- On macro `rank_hard_active` (test), the scores are 0.796 and 0.792. The difference is -0.003 [-0.009, +0.003], with
  a p of 0.38.
- On validation, the scores are 0.831 against 0.824 and 0.803 against 0.800.

The logistic regression is exploratory and outside the Holm family. The supported reading is that the gain comes
from the training pairs, not the learner; the logistic regression is not shown to be the better model. The
pair-focused LightGBM stays the pre-specified headline model, but it should not be called the best model.

## 1. Places that call the LightGBM the best or headline model without that context

| # | Where | Text | Problem | Suggested fix |
|---|---|---|---|---|
| 1 | report, Abstract, line 13 | "the best model is a LightGBM trained on the same kind of pairs it is evaluated on" | This contradicts the abstract's own sentence five lines later ("A logistic regression ... does at least as well (0.808)") and Table 1, where 0.808 > 0.799. **This is the most important fix.** | "the headline model, chosen before the final checks, is a LightGBM trained on ..." or "models trained on the same kind of pairs they are evaluated on do best; the headline LightGBM reaches ..." |
| 2 | report, Table 1, line 188 | The pair-focused LightGBM row is bold. | Bold reads as "best", but two rows score higher: the pair-focused logistic regression (0.831 / 0.808, 0.803 / 0.796) and the hybrid (0.828 / 0.808, 0.804 / 0.795). | Mark it "(headline)" instead of bolding it, or say in the table caption that bold marks the pre-specified headline model. |
| 3 | README, results table, line 53 | Same bolding. | Same as row 2. This is the first table a reader sees, and the higher row sits directly below the bold one. | Same as row 2. |
| 4 | README, figure under the table (`fig1_headline.png`) and report Figure 1 | The figure shows prefer-newer, Adamic-Adar, logistic regression, general LightGBM, two-stage and pair-focused LightGBM; its legend reads "Blue = headline model". | The pair-focused logistic regression is not in the figure. The figure alone shows the LightGBM as the clear top model, which is the reading the table now argues against. | Add the exploratory pair-focused logistic regression to the figure (marked exploratory), or say in the caption that it is omitted and scores 0.808 / 0.796. |
| 5 | report, §6 RQ3, line 212; §8, lines 254, 270, 274, 278, 287 | "the headline model" | The term is never defined in the report. After the final pass a reader could take it to mean the best-scoring model. | Define it once, for example in §4 or at the start of §6: "headline model = pair-focused LightGBM, fixed before the final-pass checks". |
| 6 | README, Finding 1, line 65; Limitations, line 133 | "The pair-focused model beats ...", "the pair-focused model still beats Adamic-Adar" | There are now two pair-focused models, so "the pair-focused model" is ambiguous. The disambiguation result (0.780 against 0.710) is for the LightGBM only. | Write "pair-focused LightGBM". |

## 2. Statements that contradict or strain the finding

| # | Where | Text | Problem |
|---|---|---|---|
| 7 | report, §6, lines 160-161 | "slightly above the LightGBM (difference -0.008 [-0.015, -0.002] ...)" | The sign is the reverse of the sentence. "Logistic regression above LightGBM" is followed by a negative number, which is LightGBM minus logistic regression. A reader will think the sign is wrong. Fix: "(LightGBM minus logistic regression: -0.008 ...)" or give +0.008. The sentence also does not say that the ranking difference is not significant (p 0.38) or that the hard_active one is uncorrected and exploratory (p 0.016). Without that, "slightly above" may read as a claim that logistic regression is better. |
| 8 | report, §6 RQ3, lines 212-215, and §8 "What the models use" | The model's drivers are described from SHAP on the LightGBM only. | This is not a contradiction, because RQ3 is about the headline model. But since a linear model on the same pairs does as well, the reader may ask whether its coefficients show the same drivers. One sentence of scope would help: the drivers were measured on the LightGBM and not checked on the logistic regression. The LightGBM's 2-leaf trees make it additive, so the two are likely similar, but that has not been checked. |
| 9 | report, §6, lines 178-179 | The hybrid "adds +0.009 [+0.005, +0.014] over the pair-focused LightGBM" | This is consistent with the data, but missing context. The hybrid's 0.808 equals the pair-focused logistic regression's 0.808, so a shortcut-trained GNN adds no more than switching the learner does. That supports treating the hybrid as secondary; one clause would make the point. |
| 10 | `results/final_summary.md`, line 140 (linked from the README as "Summary of all results") | "The pair-focused LightGBM is the best model" | This file is outside the two documents checked, but the README sends readers there. It predates the final pass and contradicts it. It also still writes "Holm p = 0.0068" rather than the floor wording the README and report now use. It is generated by `scripts/final_summary.py`, so the fix belongs in that script, not in the markdown. |
| 11 | `docs/explainability.md`, line 111 | "The headline model is also the best newcomer model" | This was true among the models in `final_comparison.csv`. The pair-focused logistic regression scores higher on test newcomer pairs: 0.798 against 0.783 on `hard_active`, newcomer_matched (`results/pf_logreg_exploratory.csv`). The two documents checked here do not repeat the claim; report §8 line 275 says only that every model does worse on newcomers, which still holds. |

The rest is consistent with the finding:
- The README summary, lines 8-10: "a logistic regression trained that way does as well as the LightGBM, so the gain comes from the training pairs".
- README Finding 1's "at least as well".
- The abstract's sentence at lines 17-19.
- The report's "Is it the model or the training pairs?" paragraph.
- The conclusion: "the training pairs matter more than the learner".
- Report §8's note that the LightGBM is additive.

No finding claims that LightGBM's flexibility or its interactions explain the gain.

## 3. README read top to bottom as a first-time reader

What works:
- The order is right: summary, caveats, questions, results, findings, transfer questions, reproduce, map,
  limitations.
- The caveats are visible before any number.
- The terms paragraph comes before the table.
- The summary's last two sentences state the main result and the negative results plainly.

Where a first-time reader stumbles, in reading order:

1. **Summary, lines 4-7.** The second sentence stacks three conditions ("Under a leak-free temporal split,
   negatives that share a co-author and are both still publishing, and scores computed separately ...") before the
   verb, and the first condition (a split) is not parallel with the second (negatives). It is hard to parse on
   first read. Splitting it into two sentences would help, for example "We evaluate under ... . Under that evaluation,
   much of the apparent skill ... disappears."
2. **Summary, line 9.** "does as well as the LightGBM": "the LightGBM" has not been introduced yet (the first
   LightGBM mention uses the definite article).
3. **Caveats, lines 16-24, come before the terms.** They use "headline protocol", "macro"-style group scoring,
   "general LightGBM" and "the pair model" before any of these are defined (the terms paragraph is at line 38).
   "The pair model" in caveat 2 is never defined anywhere in the README; it is the pair-focused LightGBM, used as
   stage 2 of the two-stage model (`scripts/two_stage.py`). One option is to keep the caveats on top and define the
   two or three terms they need in place.
4. **Caveat 1, line 19: "Validation-year (2024) numbers are given next to every test number below."** This is not
   true. Validation numbers appear in the table only. Finding 1's gains (+0.074, +0.077, +0.093), the logistic
   regression's 0.808 in the text, the text gain (+0.023), Finding 2 (0.749, 0.488), Finding 3 (0.775) and the
   disambiguation result (0.780 against 0.710) are test-only. The report has the same overclaim at line 28
   ("validation numbers are reported beside test numbers throughout"); §6 RQ2, §7 and §8 are mostly test-only.
   Fix by narrowing the wording ("in the results table") or by adding the validation values.
5. **Research questions, then findings.** The three findings are not labelled by research question. Finding 1
   answers RQ1, Finding 1's last sentence answers RQ2, and Findings 2 and 3 are emergent findings (the report's §7).
   RQ3 has no answer in the README. A reader who has just read the three questions will look for three answers.
   Labelling each finding ("RQ1", "RQ2", "Not pre-specified") and adding a one-line RQ3 answer would close that gap.
6. **Finding 1 carries too much.** It holds two comparators, a p-value floor, an exploratory flag, the logistic
   regression result, the clustered bootstrap and the RQ2 text result, in one paragraph. The text result (+0.023)
   is a separate question and is hard to find at the end. The earlier reviewer pass also noted that it is given
   without scale (it is about a third of the model's gain over Adamic-Adar).
7. **Finding 3, line 75.** The line is much longer than the others in the source (a wrapping slip). It does not
   affect the rendered page.
8. **Reproduce, lines 103-111.** `make final-pass` is missing. The pair-focused logistic regression row, the
   Jaccard comparison and the clustered intervals come from `scripts/final_pass_checks.py` (`make final-pass`). A
   reader who runs the listed commands will not reproduce them. `scripts/check_numbers.py` still passes, because
   the CSVs are committed.
9. **Repository map, line 121.** The `scripts/` row lists stages up to Phase 8, without the final-pass script.
   This is minor.

## 4. Summary

The two documents agree with the finding in their prose: the summary, the "model or training pairs" paragraph and
the conclusion all say the training pairs matter, not the learner. Four things still present the LightGBM as the
best model:
- one sentence in the report abstract;
- the bold rows in both results tables;
- the headline figure, which leaves the logistic regression out;
- `results/final_summary.md`, which is linked from the README.

The sign in report §6 lines 160-161 is likely to confuse a reader. For a first-time reader of the README, the
fixes most worth making are:
- the false "every test number" claim in caveat 1;
- the undefined "pair model";
- the missing mapping from findings to research questions;
- the missing `make final-pass` step.
