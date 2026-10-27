# Who Collaborates Next?

Predicting which researchers at the University of Technology Sydney (UTS) will co-author for the first time next
year, from the co-authorship graph and publications before a cutoff. The main work is in the evaluation. Models are
scored under a leak-free temporal split, against non-collaborators who already share a co-author and are both still
publishing, and separately for established and newcomer authors. Under that evaluation, much of the apparent skill
on easier protocols disappears. Several results that looked strong turned out to measure who keeps publishing, who
is new, or how far apart two people are in the graph, not who will collaborate. The main positive result is about
training data, not the learner: models trained on the same kind of pairs they are scored on do best, and a LightGBM
and a logistic regression trained on those pairs score about the same. node2vec and a graph neural network trained
with a known flaw do not beat simple heuristics.

Full write-up: [`report/report.md`](report/report.md). Summary of all results: [`results/final_summary.md`](results/final_summary.md).

**Read these two caveats first.**

1. **The test year shaped the headline protocol.** Matching negatives on activity and scoring within newcomer and
   established groups were adopted after diagnostics on the test year (2025), before a validation-first rule was in
   place. The test numbers are therefore not a clean confirmation of those two choices. Validation-year (2024)
   numbers are given next to the test numbers in the results table; the numbers quoted in the findings are test-year.
2. **The headline task assumes both authors publish in the prediction year.** That is known only afterwards. The
   headline protocol (defined under Results) measures how well a model separates collaborators from
   non-collaborators *given* that both are active; it is a diagnostic, not a deployable system. A deployed
   recommender would need a shortlisting step first. A model that predicts whether both authors will be active
   reaches a test ROC-AUC of 0.881, but multiplying its score with the pair-focused LightGBM's (defined under Results)
   scored below a LightGBM trained against random pairs, so that step is not solved here.

## Research questions

Set in the project plan before any data work began; they entered the repository in Phase 7, after a reconstructed
set had briefly been recorded (marked superseded in [`docs/decisions.md`](docs/decisions.md)).

1. How much do learned models (GNNs, embeddings, feature models) improve over simple graph heuristics for predicting
   new collaborations?
2. Do topical signals (what researchers publish on) add predictive value beyond network structure?
3. Which features drive predictions, and where do models fail (e.g. newcomers, cross-discipline pairs)?

## Results

**Terms.** `hard_active`: future collaborators against non-collaborators who already share a co-author (graph
distance 2) and who both publish in the prediction year; scored by ROC-AUC. `rank_hard_active`: each future
collaborator ranked against up to 100 such non-collaborators of the same author; scored by the share of candidates
ranked below the collaborator (query AUC). **Macro**: each score is computed separately for pairs that include a
newcomer (an author with at most two earlier works) and for established pairs, then the two are averaged, so the
mix of newcomers cannot drive the result. **General** models (general logistic regression, general LightGBM) are
trained on the training year's new collaborations against random pairs. **Pair-focused** models (pair-focused
LightGBM, pair-focused logistic regression) are trained on exactly the kind of pairs they are scored on: new
collaborators who already share a co-author, against active non-collaborators who share a co-author. The **headline
model** is the pair-focused LightGBM. It was fixed as the headline before the final pass added the pair-focused
logistic regression as a check, and it stays the headline; it is not the highest-scoring row.

| Model | macro hard_active ROC-AUC, validation / test | macro rank_hard_active query AUC, validation / test |
|---|---|---|
| prefer-newer (reference, uses no pair information) | 0.576 / 0.578 | 0.561 / 0.551 |
| Adamic-Adar (validation-best heuristic on hard_active) | 0.741 / 0.725 | 0.722 / 0.699 |
| Jaccard (validation-best heuristic on rank_hard_active) | 0.692 / 0.691 | 0.729 / 0.716 |
| general logistic regression | 0.782 / 0.763 | 0.767 / 0.763 |
| general LightGBM | 0.766 / 0.751 | 0.761 / 0.756 |
| pair-focused LightGBM (pre-specified headline model) | 0.824 / 0.799 | 0.800 / 0.792 |
| pair-focused logistic regression (exploratory, final pass) | 0.831 / 0.808 | 0.803 / 0.796 |
| node2vec | 0.580 / 0.561 | 0.579 / 0.572 |
| best GNN (GraphSAGE, trained with the distance shortcut) | 0.637 / 0.640 | 0.610 / 0.619 |

Mean of 5 seeds. Comparisons use one paired bootstrap per comparison on the seed-averaged scores, with Holm
correction across the predefined family of 34 comparisons. The pair-focused logistic regression and the comparisons
against Jaccard are exploratory, outside that family.

![Headline results](figures/final/fig1_headline.png)

The figure (test year) also shows a two-stage model (see caveat 2) and a hybrid that adds the GNN's score to the
pair-focused LightGBM; the hybrid is secondary because it is built on the shortcut-trained GNN.

## Findings, by research question

1. **RQ1: the training pairs matter, not the learner.** On test the headline model beats the validation-best
   heuristic by +0.074 ROC-AUC (Adamic-Adar, hard_active) and +0.077 ranking AUC (Jaccard, rank_hard_active;
   +0.093 against Adamic-Adar), both p < 0.01 (the smallest value this test can show; the Jaccard comparison is
   exploratory). A logistic regression trained on the same pairs scores 0.808 and 0.796. LightGBM minus logistic
   regression is -0.008 [-0.015, -0.002] (raw p 0.016) and -0.003 [-0.009, +0.003] (raw p 0.38); this check is
   exploratory and outside the Holm family. The two learners are roughly tied, so the gain is not specific to
   LightGBM. Resampling authors rather than pairs (pairs share authors) roughly doubles the interval widths and
   leaves the gains over the heuristics clearly above zero. node2vec and the shortcut-trained GNN score below
   Adamic-Adar.
2. **RQ2: topical signals add a little.** Adding four text features to the structural ones raises the headline
   model from 0.754 to 0.777 on macro hard_active (+0.023, p < 0.01), about a third of its gain over Adamic-Adar.
3. **RQ3: drivers and failures.** The headline model relies mostly on recency-weighted shared co-authors and topical
   similarity; it fails most on newcomer pairs (0.783 against 0.815 for established pairs on test) and on
   collaborations with no network or topical precursor; cross-field pairs are not a weak spot.
4. **Not pre-specified: easy protocols reward predicting who stays active.** When negatives are not matched on
   publishing activity, a single "recent works" feature scores 0.749 ROC-AUC; once activity is matched it scores
   0.488.
5. **Not pre-specified: the GNN learned a distance shortcut.** The evaluated GNN ranks far-apart non-collaborators
   above distance-2 collaborators (shortcut check 0.775; above 0.5 means it uses the shortcut). On a CPU retrain of
   it, whose validation scores correlate 0.955 to 0.987 with the evaluated GNNs, removing the paths through a shared
   co-author raised the score of each of its 4 most confident predictions examined (+0.54 to +0.77 in logit).

## Questions I would check first on a new network dataset

These are evaluation questions, not results; nothing here says the models or numbers carry over to other networks.

- Is the split temporal, with every feature and candidate built only from data before the cutoff?
- Are the negatives as hard as the cases that matter (for example, nodes that are already two hops apart), or are
  they trivially far apart?
- Is there an activity-style confound (is this node still active at prediction time?), and does matching on it
  change the ranking of models?
- Does a reference score that uses no information about the pair do well? If so, the protocol rewards something else.
- Are the training negatives drawn so that a single feature such as distance separates the classes?
- Do confidence intervals account for observations that share a node?

## Reproduce

```bash
conda env create -f environment.yml && conda activate linkpred
make test                                   # data-dependent tests skip until the data is built
```

`data/` is not in git. Build it from an OpenAlex snapshot of UTS works 2015 to 2025
(`export LINKPRED_SNAPSHOT_PATH=/path/to/uts_works_2015_2025.json`) or download it
(`export OPENALEX_MAILTO=you@example.org; python -m linkpred.data.fetch_openalex`), then:

```bash
make data splits baselines features train   # data, splits, heuristics, feature models, node2vec
make gnn-bundle                             # GNNs train on a GPU via notebooks/06_gnn_colab.ipynb
make gnn-import gnn-analysis                # after downloading the Colab results
make phase7 final                           # robustness checks, Holm-corrected comparisons, final summary
make phase8                                 # explainability and error analysis
make final-pass                             # exploratory checks: pair-focused logistic regression, clustered bootstrap
python scripts/final_figures.py             # figures/final/
python scripts/check_numbers.py             # every number in README and report traced to a results file
```

GNN results were produced with torch 2.11, CUDA 13.0 and a T4 GPU (`results/gnn_versions.json`). Rebuilding from a
fresh OpenAlex download will differ slightly because OpenAlex keeps indexing.

## Repository map

| Path | Contents |
|---|---|
| `src/linkpred/` | package: data build and splits, features, models, evaluation protocols, bootstrap, results writer |
| `scripts/` | one script per experiment stage (baselines, feature models, node2vec, GNN, Phase 7 and 8 analyses, final-pass checks) |
| `configs/default.yaml` | all paths, split years, sampling ratios, seeds and hyperparameters |
| `results/` | every reported number, as validated CSVs (`final_summary.md`, `final_comparison.md`) |
| `figures/final/` | report figures, drawn from `results/` only |
| `report/` | paper (`report.md`), number provenance (`numbers.yaml`), script-produced diagnostics |
| `docs/` | decision log, data audit, split format, explainability, independent audit |
| `tests/` | unit and leakage tests (`make test`) |

## Limitations

- Test-year reuse for two design decisions (see the caveats at the top).
- Author identities come from OpenAlex. Removing every author ID flagged as a possible split profile cuts the test
  positives from 2,533 to 1,546; the pair-focused LightGBM still beats Adamic-Adar on that subset (0.780 against
  0.710), but merged profiles cannot be detected this way.
- 2025 works are likely under-indexed; pairs counted as negatives may collaborate later (right-censoring).
- One institution, UTS-UTS pairs only.
- The GNNs were trained before the distance shortcut was found and were not retrained, so the GNN results answer the
  research question only for that flawed training.

## What I would do with more time

- Retrain the GNNs without the distance shortcut (distance-2 positives only; the code already does this by default)
  and re-run the comparison and explanations on the retrained models.
- Evaluate on a clean held-out year (2026) that played no part in any design decision.
- Design and test a deployable shortlisting step for the activity condition.
- Use a finer discipline definition (subfield or topic overlap) for the cross-field analysis.
- Re-run on a later OpenAlex snapshot to reduce indexing lag and to check right-censored negatives.
- Repeat the full protocol at a second institution.
- Evaluate a recommender setting directly: rank all candidates for a researcher, not sampled pairs.

## Notes

The code was implemented with the help of Claude Code; design decisions and their reasoning are recorded in
[`docs/decisions.md`](docs/decisions.md). Project rules are in [`CLAUDE.md`](CLAUDE.md).
