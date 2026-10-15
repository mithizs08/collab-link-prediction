# Who Collaborates Next?

Predicting which researchers at the University of Technology Sydney (UTS) will co-author for the first time next
year, from the co-authorship graph and publications before a cutoff. The main work is in the evaluation. Under a
leak-free temporal split, negatives that share a co-author and are both still publishing, and scores computed
separately for established and newcomer authors, much of the apparent skill on easier protocols disappears. Several
results that looked strong turned out to measure who keeps publishing, who is new, or how far apart two people are
in the graph, not who will collaborate. A LightGBM trained on the same kind of pairs it is scored on is the best
model; it beats the Adamic-Adar heuristic by a clear but modest margin. node2vec and the graph neural network tested
here do not beat the heuristic.

Full write-up: [`report/report.md`](report/report.md). Summary of all results: [`results/final_summary.md`](results/final_summary.md).

## Results (test year 2025)

| Model | macro hard_active ROC-AUC | macro rank_hard_active query AUC |
|---|---|---|
| prefer-newer (reference, uses no pair information) | 0.578 | 0.551 |
| Adamic-Adar | 0.725 | 0.699 |
| general LightGBM | 0.751 | 0.756 |
| **pair-focused LightGBM** | **0.799** | **0.792** |
| node2vec | 0.561 | 0.572 |
| best GNN (GraphSAGE, trained with the distance shortcut) | 0.640 | 0.619 |

Mean of 5 seeds. `hard_active`: future collaborators against non-collaborators who share a co-author and both publish
in 2025. Macro: scored separately for established and newcomer pairs, then averaged. All comparisons use one paired
bootstrap per comparison with Holm correction.

![Headline results](figures/final/fig1_headline.png)

## Three findings

1. The pair-focused LightGBM beats Adamic-Adar by +0.074 ROC-AUC and +0.093 ranking AUC (Holm p = 0.0068), and text
   features add +0.023 beyond network structure.
2. When negatives are not matched on publishing activity, a single "recent works" feature scores 0.749 ROC-AUC, and it
   falls to 0.488 once activity is matched: easy protocols reward predicting who stays active.
3. A training set whose negatives all share a co-author taught the GNN a distance shortcut; removing the paths through
   a shared co-author raises its score (+0.54 to +0.77 in logit for its confident predictions).

## How the methods transfer

Telecom, power and transport networks are graphs too, and the same evaluation problems apply when predicting new
links, faults or demand over time. A random split leaks the future into training. Random negatives are far easier
than the cases that matter. A confound like "is this node still active?" can carry most of a score. Explanations can
reveal a shortcut that aggregate metrics hide. The specific models and numbers here do not transfer; the checks do:
temporal splits, negatives matched on the confound, scores within comparable groups, reference scores that use no
pair information, and Holm-corrected comparisons.

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
python scripts/final_figures.py             # figures/final/
python scripts/check_numbers.py             # every number in README and report traced to a results file
```

GNN results were produced with torch 2.11, CUDA 13.0 and a T4 GPU (`results/gnn_versions.json`). Rebuilding from a
fresh OpenAlex download will differ slightly because OpenAlex keeps indexing.

## Repository map

| Path | Contents |
|---|---|
| `src/linkpred/` | package: data build and splits, features, models, evaluation protocols, bootstrap, results writer |
| `scripts/` | one script per experiment stage (baselines, feature models, node2vec, GNN, Phase 7 and 8 analyses) |
| `configs/default.yaml` | all paths, split years, sampling ratios, seeds and hyperparameters |
| `results/` | every reported number, as validated CSVs (`final_summary.md`, `final_comparison.md`) |
| `figures/final/` | report figures, drawn from `results/` only |
| `report/` | paper (`report.md`), number provenance (`numbers.yaml`), script-produced diagnostics |
| `docs/` | decision log, data audit, split format, explainability, independent audit |
| `tests/` | unit and leakage tests (`make test`) |

## Limitations

- Two design changes followed diagnostics computed on the test year, before the validation-first rule; the test year
  is not a clean confirmation for those two decisions.
- Author identities come from OpenAlex; a heuristic flag for split profiles removes a large share of positives when
  applied, and merged profiles cannot be detected.
- 2025 works are likely under-indexed; pairs counted as negatives may collaborate later (right-censoring).
- One institution, UTS-UTS pairs only.
- The GNNs were trained before the distance shortcut was found and were not retrained.

## What I would do with more time

- Retrain the GNNs on distance-2 positives and re-run the comparison and explanations.
- Use a finer discipline definition (subfield or topic overlap) for the cross-field analysis.
- Re-run on a later OpenAlex snapshot to reduce indexing lag and to check right-censored negatives.
- Repeat the full protocol at a second institution.
- Evaluate a recommender setting directly: rank all candidates for a researcher, not sampled pairs.

## Notes

The code was implemented with the help of Claude Code; design decisions and their reasoning are recorded in
[`docs/decisions.md`](docs/decisions.md). Project rules are in [`CLAUDE.md`](CLAUDE.md).
