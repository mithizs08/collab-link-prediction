# Reviewer pass: README and report (simulated internship screening)

Date: 2026-10-15. Perspective: senior researcher at a university telecom research institute, screening internship
applicants; 2 minutes on `README.md`, 15 minutes on `report/report.md`. Report only, nothing changed. Numbers quoted
below are from the README/report, plus `results/baselines.csv` and `docs/decisions.md` where noted.

## 1. After two minutes on the README

**What I would remember**

- "Predicting first-time co-authorships at UTS; the point is the evaluation, not the model."
- A boosted-tree model beats Adamic-Adar by about 0.07-0.09 AUC; node2vec and a GNN lose to the heuristic.
- The candidate found that easy protocols measure "who keeps publishing", and that a GNN learned a distance shortcut.
- The tone is unusually honest for an applicant: negative results up front, a limitations section, a statement that
  the test year was reused.

**Would I read further?** Yes. The opening paragraph signals someone who distrusts their own numbers, which is rarer
than a high score. But I nearly stopped at the table: the column headers (`macro hard_active ROC-AUC`,
`macro rank_hard_active query AUC`) mean nothing before the definitions, which come *below* the table. A reader with
less patience reads the first paragraph, sees jargon, and moves on.

**What did not land in two minutes:** what "pair-focused" means, why AUC 0.80 is good or bad for this task, and the
telecom paragraph (read as a generic add-on, see 2.7).

## 2. Claims that overreach the evidence or sound like hype

1. **"Explanations show [the shortcut] directly in a graph neural network"** (abstract; README finding 3). The
   explanation runs on a CPU *retrain* of the GNN, not the evaluated model (score correlation 0.955 for one of the two
   models), and the logit changes quoted (+0.54 to +0.77) are four predictions. The README turns this into a general
   statement ("raises its score"). Say: "in a CPU replica, removing shared-co-author paths raised the score of each of
   its 4 most confident predictions", and say n.

2. **"Representation learning did not help"** (report §6, RQ1). The GNN was trained with a known bug (the distance
   shortcut) and not retrained; the report itself says "its ceiling is unknown". So RQ1 is answered only for
   node2vec and for a flawed GNN. The conclusion's wording ("the GNN tested here") is right; §6's is not. Relatedly,
   §9 says the GNN results "understate what a correctly trained GNN could do": that is an untested claim in the
   other direction. "May understate" is what the evidence supports.

3. **node2vec at 0.561 is below the reference that uses no pair information (0.578).** The report presents this as
   "representation learning did not help" without remarking that it is worse than a score that knows nothing about
   the pair. A reviewer's first thought is a pipeline problem (Procrustes alignment, edge operator, retraining per
   split), not a property of node2vec. This needs one sentence of diagnosis, or it undermines trust in the rest.

4. **Choice of comparator.** All headline gains are against Adamic-Adar. On `rank_hard_active`, Jaccard (0.716) and
   Katz (0.707) beat Adamic-Adar (0.699) (`results/baselines.csv`). Against the strongest heuristic per metric, the
   ranking gain is +0.076, not +0.093. Either justify Adamic-Adar as the pre-chosen reference or report the gap to
   the best heuristic.

5. **"Holm p = 0.0068" reported as if exact.** This is the smallest attainable value (5,000 resamples x 34 tests);
   every headline comparison sits on that floor. §5 explains it, the README does not. Write "p ≤ 0.0068 (resolution
   limit)". Also, the bootstrap resamples pairs, but pairs share authors; if resampling ignores this clustering, the
   intervals are too narrow. If authors are resampled, say so.

6. **"Three research questions were fixed in the project plan before any data was seen"** (§1). The decision log
   (2026-10-15) says the RQs were not in the repository until Phase 7, and that Phase 7 first wrote *different* RQs,
   later replaced by the ones from an external plan document. The claim may well be true, but nothing in the repo
   lets a reader check it. Either add the original plan (or its date) to the repo or soften to "set out in the
   original project plan".

7. **The telecom paragraph** ("How the methods transfer"). "The same evaluation problems apply when predicting new
   links, faults or demand over time" is asserted, not shown; fault and demand forecasting are not link prediction.
   To a telecom reader this reads as tailoring. Better: one concrete, correct mapping (e.g. "predicting new
   inter-cell neighbour relations or peering links: random negatives are trivially far apart; 'is this cell still
   carrying traffic?' is the activity confound"), or drop the paragraph.

8. **Novelty claim** (§2): activity-matched negatives and within-group scoring are "two controls we did not find in
   that work". This is scoped to the two cited papers, which is honest, but the related-work search is two
   evaluation papers deep. Temporal link prediction has a literature on inactive nodes and node "birth/death"; a
   reviewer would ask whether this was checked.

9. **Small effects stated without scale.** "Text features add +0.023" and "hybrid +0.009" are significant but tiny;
   the report does say so for the hybrid. Fine as written in the report; the README lists +0.023 as a headline
   "finding" without saying it is about a third of the model's gain over the heuristic.

Not overreaching, and worth keeping: the "one in five comparisons wrong" translation of AUC 0.80, the calibration
section ("use for ranking only"), "some errors are early rather than wrong" (hedged, with the 4/100 vs 0.7% base
rate), and the admission that test was reused.

## 3. Unclear, jargon-heavy, or buried

**Should move up front**

- **The test year shaped the headline metric.** Activity matching and macro within-group scoring (i.e. the headline
  protocol itself) were adopted after diagnostics on the test year. This is the single most important caveat on the
  headline number and it sits in the last section of both documents. One sentence beside the results table:
  "the headline protocol was chosen after looking at test-year diagnostics; validation numbers are X".
- **Validation numbers for the headline.** Neither document gives the headline models' validation scores. Given the
  point above, these are the cleanest evidence available and should be in Table 1.
- **The disambiguation result.** Removing flagged split profiles drops test positives from 2,533 to 1,546 (39%).
  The README says "a large share". That a third or more of "new collaborations" involve possibly duplicated
  identities is a data-validity issue a reviewer would want in the Data section and the README in numbers.
- **The "LightGBM" is an additive model.** §8 reveals the pair-focused LightGBM uses 2-leaf trees (boosted stumps,
  no interactions). That changes how to read RQ1: the win comes from training on the right pairs with good features,
  not from a flexible learner. Logistic regression (0.763) beats the general LightGBM (0.751) and is missing from the
  README table. Together these suggest the key ingredient is the training distribution, and the README should say so.
- **The headline task conditions on the future.** `hard_active` keeps only negatives whose authors both publish in
  the prediction year, which a deployed recommender cannot know. This is a correct *diagnostic* design, but the
  README presents it as "the task". Say explicitly that it measures pair signal given activity, and point to the
  two-stage result for the deployable version.

**Jargon to define on first use (README especially)**

`hard_active`, `rank_hard_active`, "query AUC", "macro", "pair-focused", "prefer-newer", "distance 2", "distance
shortcut", "in logit", "Holm". A plain-language row header ("Future collaborators vs. active non-collaborators who
share a co-author, scored within newcomer/established groups") would carry the table on its own.

**Smaller clarity points**

- Report all ranking results as AUC; for a recommender reader, MRR or hits@10 is more intuitive (Adamic-Adar MRR is
  about 0.32 in `results/baselines.csv`; the headline model's is not in the report).
- "Newcomer" (≤ 2 prior works) and the size of each group should be given in §4, with counts.
- The shortcut figure is referenced in §7 but placed in §8.
- Every reference carries `[VERIFY]` and the list ends with "All entries must be checked by the author before
  submission." As shipped, this reads as an unfinished, partly machine-generated draft. Verify and remove the tags
  before anyone outside sees it. (The README's note that the code was written with Claude Code is fine and honest;
  unverified citations are a different matter.)

## 4. Three interview questions

1. **"Your headline protocol was designed after looking at the test year. If you had to defend the 0.799 to a
   sceptical reviewer, what would you show them?"** Looking for: validation-year results for the same comparison,
   an honest account of which decisions were contaminated, and ideally a plan for a clean held-out year (2026 data).

2. **"Your best model is boosted stumps trained on distance-2, activity-matched pairs. Logistic regression beats the
   general LightGBM. How much of the gain is the model, and how much is choosing the training pairs?"** Looking for:
   whether they ran (or would run) a pair-focused logistic regression and a pair-focused Adamic-Adar-only model, and
   whether they understand that their main contribution is about training/evaluation distribution matching.

3. **"node2vec scores below a reference that uses no information about the pair. What did you check before
   concluding that node2vec doesn't work here, and how would you rule out a bug?"** Looking for: debugging habits
   (a sanity check on the `random` protocol, alignment diagnostics, an unaligned per-split baseline), and willingness
   to say "I'm not sure it's not a bug".

Backup question if time allows: "39% of your test positives involve possibly split author profiles. How do you know
your model isn't partly predicting OpenAlex disambiguation errors?"

## Overall

Strong candidate on judgement and honesty; the evaluation work is the real contribution and is ahead of what most
applicants send. The write-up undersells its most important caveat (test-year contamination of the headline
protocol), oversells two secondary claims (GNN explanation, "representation learning did not help"), and the
unverified citations would hurt the first impression. All are fixable in an afternoon.
