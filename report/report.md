---
title: "Who Collaborates Next? Leak-free evaluation of new-collaboration prediction in a university co-authorship network"
date: "October 2026"
geometry: a4paper, margin=2.2cm
fontsize: 10pt
---

# Abstract

We predict which pairs of University of Technology Sydney (UTS) researchers will co-author for the first time in the
next year, using only the co-authorship graph and publications from before a cutoff. The main contribution is the
evaluation. Under a leak-free temporal split with negatives that share a co-author and are both still publishing,
and with scores computed separately for established and newcomer authors, the best model is a LightGBM trained on
the same kind of pairs it is evaluated on. On the 2025 test year it reaches a macro ROC-AUC of 0.799 [0.781, 0.816]
and a macro ranking AUC of 0.792 [0.780, 0.804], against 0.725 and 0.699 for the Adamic-Adar heuristic (differences
+0.074 and +0.093, Holm-corrected p = 0.0068). node2vec (0.561) and the best graph neural network (0.640) score below
Adamic-Adar. Text features add +0.023 beyond network structure. Three findings come from the evaluation itself:
protocols that do not match negatives on publishing activity reward predicting who keeps publishing, pooled scores
reward newcomer status, and a training set whose negatives are all at distance 2 teaches models a distance shortcut,
which explanations show directly in a graph neural network.

# 1. Introduction

Research offices, funders and researchers use collaboration recommendations to find partners for grants, to plan
hiring and to spot groups that could work together. Whether such recommendations are useful depends on how well new
collaborations can be predicted, and on whether reported accuracy reflects the task or the evaluation. Link
prediction in co-authorship networks is a standard benchmark [Liben-Nowell and Kleinberg 2007] [VERIFY], and simple
neighbourhood scores such as Adamic-Adar remain hard to beat [Adamic and Adar 2003] [VERIFY]. At the same time,
evaluation choices such as random splits and easy random negatives can inflate scores [Yang et al. 2015] [VERIFY];
[Li et al. 2023] [VERIFY].

We study one institution, UTS, using OpenAlex data from 2015 to 2025 [Priem et al. 2022] [VERIFY]. Three research
questions were fixed in the project plan before any data was seen:

1. **RQ1.** How much do learned models (GNNs, embeddings, feature models) improve over simple graph heuristics for
   predicting new collaborations?
2. **RQ2.** Do topical signals (what researchers publish on) add predictive value beyond network structure?
3. **RQ3.** Which features drive predictions, and where do models fail (e.g. newcomers, cross-discipline pairs)?

Along the way the evaluation design changed several times, each change recorded with its reason in the project's
decision log. Those changes produced the findings we consider most useful, and we report them separately from the
research questions (Section 7).

# 2. Related work

**Heuristics.** Liben-Nowell and Kleinberg framed link prediction in co-authorship networks and compared
neighbourhood and path scores [Liben-Nowell and Kleinberg 2007] [VERIFY]. Adamic-Adar weights common neighbours by the
inverse log of their degree [Adamic and Adar 2003] [VERIFY]; resource allocation uses the inverse degree [Zhou et al.
2009] [VERIFY]; the Katz index counts paths of all lengths with decaying weights [Katz 1953] [VERIFY]. Co-authorship
networks are known to be highly clustered, which is why common-neighbour scores work well [Newman 2001] [VERIFY].

**Learned representations.** node2vec learns node embeddings from biased random walks [Grover and Leskovec 2016] [VERIFY]. GraphSAGE [Hamilton et al. 2017] [VERIFY] and graph attention networks [Velickovic et al. 2018] [VERIFY]
aggregate neighbour features and apply to unseen graphs. We use gradient-boosted trees as implemented in LightGBM
[Ke et al. 2017] [VERIFY] for engineered pair features.

**Explanations and statistics.** SHAP attributes a prediction to features using Shapley values [Lundberg and Lee 2017] [VERIFY], with an exact algorithm for tree ensembles [Lundberg et al. 2020] [VERIFY]. For graph models we use
GNNExplainer [Ying et al. 2019] [VERIFY] and integrated gradients [Sundararajan et al. 2017] [VERIFY]. Confidence
intervals use the bootstrap [Efron and Tibshirani 1993] [VERIFY]; multiple comparisons are corrected with Holm's
procedure [Holm 1979] [VERIFY]; probability calibration uses Platt scaling [Platt 1999] [VERIFY].

**Evaluation pitfalls.** Yang et al. show that link-prediction results depend strongly on how test pairs and negatives
are sampled and on temporal leakage [Yang et al. 2015] [VERIFY]. Li et al. report that common GNN link-prediction
benchmarks use easy negatives and inconsistent settings, and that harder, more realistic negatives change model
rankings [Li et al. 2023] [VERIFY]. Our protocols follow this line and add two controls we did not find in that
work: matching negatives on publishing activity in the prediction year, and scoring newcomer and established
authors separately.

# 3. Data

The input is the set of OpenAlex works with at least one UTS-affiliated author, 2015 to 2025: 63,514 raw records and
61,493 works after cleaning (errata, peer reviews and paratext removed, duplicates merged). Works with more than 25
authors are kept for author statistics but produce no edges, since large consortia create cliques that do not reflect
working relationships. Abstracts are missing for 32% of works; text features use titles and, where present, abstracts.

Author identities are OpenAlex author IDs. We flag possible split profiles (same normalised name, more than one ID at
the same institution): 1,011 such groups at UTS involve 2,689 IDs. These are flagged, not merged; Section 9 reports a
sensitivity analysis that removes them.

# 4. Method

**Temporal splits.** Each split has an input graph built only from works before a cutoff year and a one-year label
window: train predicts 2023 from works before 2023, validation predicts 2024, and test predicts 2025. The test input
graph has 88,562 authors and 536,306 distinct co-author pairs. Every feature, degree, embedding, normalisation
statistic and negative candidate is computed from the input graph and pre-cutoff works only, and unit tests fail if
future information enters.

**Targets.** A positive is a pair of UTS authors, both present before the cutoff, who co-author for the first time in
the label window (any team size counts as a joint work for this purpose). Test has 2,533 positives and validation
2,225.

**Negatives and protocols.** All models are scored on the same saved pairs.

- `random`: all positives against an equal number of random unconnected UTS pairs. Easy: most random pairs are far
  apart in the graph.
- `hard`: positives at distance 2 (sharing at least one co-author) against non-collaborating pairs at distance 2.
- `hard_active`: as `hard`, but both authors of every negative also publish in the prediction year. This removes the
  question "is this person still active?" from the task. Prediction-year activity is used only to choose evaluation
  pairs and training labels, never as a model input. Test has 1,684 positives and 1,684 negatives.
- `rank_hard_active`: each distance-2 positive ranked against up to 100 active, distance-2 candidates for the same
  author, scored by the per-query AUC.

**Macro within-group scores.** Activity matching makes newcomers (authors with at most two works before the cutoff)
rare among negatives, while they are common among positives. A pooled score then rewards any model that ranks
newcomers high. The headline metric therefore compares positives with negatives (or candidates) from the same group,
established or newcomer, and averages the two groups' scores. Pooled numbers are kept and labelled.

**Models.** Heuristics: common neighbours, Jaccard, Adamic-Adar, resource allocation, preferential attachment and
truncated Katz. A reference score, `prefer-newer`, ranks pairs by how few prior works the less experienced author has;
it uses no pair information and shows how much a protocol rewards newcomer status. Feature models use 27 pair
features in three blocks: structural (neighbourhood counts, Katz, degrees, distance, communities, shared institutions),
temporal (career age, recent output, years since last work, recency-weighted common neighbours and Adamic-Adar) and
text (cosine similarity of the authors' mean title and abstract embeddings, and of each author to the other's
co-authors). Logistic regression and a general LightGBM are trained on train-split positives against random
negatives. The **pair-focused LightGBM** is trained on distance-2 positives against activity-matched distance-2
negatives from the train window, the same kind of pairs it is evaluated on. node2vec embeddings are aligned across
splits with orthogonal Procrustes. GraphSAGE and GAT models were trained on a GPU with a general and a pair-focused
objective. A hybrid adds the best GNN's out-of-fold score to the pair-focused LightGBM.

# 5. Experimental setup

All hyperparameters, negative-mix choices and early stopping use the validation split. Learned models run with five
seeds; values are the mean over seeds with a seed-mean bootstrap 95% interval. The primary test for each comparison is
one paired bootstrap on seed-averaged scores with 5,000 resamples, and all primary comparisons form one family of 34
tests corrected with Holm's procedure. The family was extended in steps, each time in the decision log before the new
comparison was run on test. With 5,000 resamples the smallest attainable raw p is 0.0002, so the smallest attainable
Holm-adjusted p in a family of 34 is 0.0068.

# 6. Results by research question

![Headline results on the test year. Mean of 5 seeds with 95% intervals; the dotted line is Adamic-Adar.](../figures/final/fig1_headline.png){width=100%}

**RQ1: learned models against heuristics.** Table 1 and Figure 1 give the test results. The pair-focused LightGBM is
the best model. It beats Adamic-Adar by +0.074 [+0.058, +0.091] on macro `hard_active` and +0.093 [+0.081, +0.106] on
macro `rank_hard_active` (Holm p = 0.0068 for both), and it beats the general LightGBM by +0.048. The general feature
models gain much less over the heuristic: +0.038 for logistic regression and +0.027 [+0.009, +0.044] for the general
LightGBM. Representation learning did not help. node2vec scores 0.561 and the best GNN 0.640, both significantly below
Adamic-Adar. The GNN was trained with the distance shortcut described in Section 7, so its ceiling is unknown. The
hybrid adds +0.009 [+0.005, +0.014] over the pair-focused LightGBM, a small gain from a shortcut-trained model, and we
treat it as secondary.

| Model | macro hard_active ROC-AUC | macro rank_hard_active query AUC |
|---|---|---|
| prefer-newer (reference) | 0.578 | 0.551 |
| Adamic-Adar | 0.725 | 0.699 |
| Jaccard | 0.691 | 0.716 |
| logistic regression | 0.763 | 0.763 |
| general LightGBM | 0.751 | 0.756 |
| pair-focused LightGBM | **0.799** | **0.792** |
| node2vec | 0.561 | 0.572 |
| best GNN (shortcut-trained) | 0.640 | 0.619 |
| hybrid (secondary) | 0.808 | 0.795 |

: Test results (2025 predictions), mean of 5 seeds for learned models.

An AUC of 0.80 means that in about one in five comparisons between a future collaborator and an active
non-collaborator at the same distance, the model ranks them the wrong way round. That is useful for ordering
candidates, not for predicting individual collaborations.

![RQ2: text ablation of the pair-focused LightGBM by group (test). Same parameters for each feature set.](../figures/final/fig3_rq2_text.png){width=100%}

**RQ2: topical signals.** We trained the pair-focused LightGBM on structural features only, on structural plus text,
and on all features, with the same parameters (Figure 2). Adding the four text features raises macro `hard_active`
from 0.754 to 0.777 (+0.023 [+0.013, +0.033], Holm p = 0.0068) and macro `rank_hard_active` from 0.740 to 0.767
(+0.027). Validation shows the same direction. The gain holds in both groups for ranking and for established pairs
on `hard_active`; for newcomer pairs on `hard_active` it is +0.019 [+0.002, +0.036], not significant after correction
(Holm p = 0.0536). Adding the temporal block on top adds a further +0.022. An exploratory run that adds the blocks in
the other order gives text a similar increment after the temporal block (+0.017 on test), so the answer does not
depend on order. Topical similarity helps, by about as much as the temporal block, but structure alone already gives
0.754 of the final 0.799.

**RQ3: drivers and failures.** Section 8 gives the evidence. In short, the headline model relies mostly on
recency-weighted shared neighbourhoods and on topical similarity, and it fails most on newcomer pairs and on
collaborations that have no network or topical precursor. Cross-field pairs, under a coarse field definition, are not
a weak spot.

# 7. Emergent findings

These results answer none of the three questions but shaped the evaluation.

![Left: ROC-AUC on pooled hard (hatched, not matched on activity) and macro hard_active. Right: one activity feature used directly as a score.](../figures/final/fig2_activity.png){width=100%}

**Activity confound.** When negatives are not matched on activity, part of the task is predicting who keeps
publishing. The general LightGBM scores 0.864 on pooled `hard` but 0.751 on macro `hard_active`. A single feature,
the smaller of the two authors' recent work counts, scores 0.749 on pooled `hard` and 0.488 on macro `hard_active`
(Figure 3). This is why `hard_active` is the headline protocol.

**Newcomer imbalance.** The `prefer-newer` reference uses no pair information. It scores 0.647 on pooled
`hard_active` but 0.578 macro: the pooled number rewards newcomer composition alone.

**Distance shortcut.** An early version of the pair-focused LightGBM was trained on activity-matched negatives, which
are all at distance 2, against positives at any distance. Distance alone then separates the classes. On the test
`random` protocol, that model ranked far random non-collaborators above distance-2 collaborators with ROC-AUC 1.000;
after retraining on distance-2 positives the same check gives 0.015. The best GNN was trained the same way before the
problem was found and was not retrained; its check is 0.775 (Figure 4).

**Two-stage model.** A model of the form P(both authors active) times P(collaborate given both active) scores 0.854
on pooled `hard` against 0.864 for the general LightGBM (-0.010 [-0.016, -0.005], Holm p = 0.0068), although its
activity component is accurate on its own task (test activity ROC-AUC 0.881). Calibrating both components on
training-split out-of-fold predictions does not close the gap (-0.013). The general model learns activity and the
pair signal jointly; the product of two separately trained models loses that interaction.

# 8. Explainability and error analysis

![Left: shortcut check per model (above 0.5 means far non-collaborators are ranked above near collaborators). Right: change in the GNN logit when every path through a shared co-author is removed (validation).](../figures/final/fig5_shortcut.png){width=100%}

**What the models use.** We computed exact TreeSHAP values for seed-0 models. The pair-focused LightGBM uses 2-leaf
trees, so it is additive and has no interactions. Grouped by what the features measure (test), recency-weighted
shared structure takes 0.281 of mean absolute SHAP, text similarity 0.281, plain structure 0.248 and author activity
or career 0.190. Filed by code block it would look like a "temporal" model (0.471), because the recency-weighted
common-neighbour features sit in that block. Mean absolute SHAP measures distance from the training average, not
discrimination, so we also score each group's SHAP sum as a predictor inside each pair set (Figure 5). For the
general LightGBM, the activity group discriminates on pooled `hard` (0.742) and points the wrong way on `hard_active`
(0.406). For the headline model the pattern is reversed (0.309 and 0.627): trained among active authors, it uses
activity only to rank within them.

![SHAP discrimination by feature group: ROC-AUC of each group's summed SHAP contribution used as a score (test).](../figures/final/fig4_attribution.png){width=100%}

**The shortcut, seen directly.** No GNN weights were saved during GPU training, so we retrained both GNNs on CPU from
their saved configurations (validation score correlation with the original models 0.987 and 0.955). For the
pair-focused GNN, removing every path through a shared co-author raises the score of its confident predictions
(+0.71, +0.74, +0.77 and +0.54 in logit): it treats a shared co-author as evidence against a link. Its most confident
validation predictions are disconnected pairs with no common neighbour (logits 8.2 and 7.8, against 0.89 for its best
true positive). For the general GNN, the same deletion lowers or leaves the score (-0.50 to -0.00). GNNExplainer masks
were close to uniform and did not support conclusions; integrated gradients and edge deletion did.

**Errors (headline model, validation).** Among established pairs, the most confident false positives share
more co-authors (median 5 against 1 for all negatives) and are topically closer (text similarity 0.669 against 0.362);
4 of the top 100 co-authored in the following year, against 0.7% of all negatives, so some errors are early rather
than wrong. Missed collaborations have lower text similarity and are more often cross-field than other positives in
both groups. Newcomer pairs are harder for every model: on test the headline model scores 0.783 for newcomer pairs and
0.815 for established ones, and no model does better on newcomers once negatives are group-matched.

**Cross-field pairs.** Each author's field is the most common OpenAlex field of their works before the cutoff
(26 fields), and both positives and negatives are split by the pair's own field relation. On test the headline model
scores 0.830 on cross-field pairs and 0.792 on same-field pairs. This is consistent with the error result above:
cross-field pairs make up 47% of new collaborations but 60% of non-collaborations, so being cross-field counts
against a pair and cross-field collaborators are more often missed, yet within the cross-field subset the model
separates collaborators from non-collaborators well. One field per author out of 26 is coarse, and a finer definition
could change this.

**Calibration.** Neither model's scores are probabilities. On test `hard_active` pairs the general LightGBM has a
Brier score of 0.417, worse than the 0.250 of always predicting the base rate, because it predicts about 0.93 for
pairs of which half collaborate. The headline model scores 0.230 and under-predicts. The evaluation sets are sampled
about 1:1, while real new collaborations are a tiny fraction of all pairs, so the scores should be used for ranking
only.

# 9. Limitations

- **Test-set reuse.** Two design changes (activity matching and newcomer-matched scoring) followed diagnostics that
  were computed on the test year before the validation-first rule was adopted. The test year is therefore not a
  clean confirmation for those two decisions. All later decisions used validation only.
- **Disambiguation.** The split-profile flag is a heuristic and also catches correctly identified authors who share a
  name. Removing all flagged IDs keeps the main result (pair-focused LightGBM 0.780 against Adamic-Adar 0.710,
  difference +0.071, Holm p = 0.0068) but reduces the test positives from 2,533 to 1,546, so it checks a smaller, different
  population. Merged profiles cannot be detected this way.
- **OpenAlex coverage and indexing lag.** The data reflect OpenAlex at the time of the snapshot. Works from 2025 are
  likely to be under-indexed, which affects the test labels; a fresh download would differ slightly.
- **Right-censoring.** A pair counted as a negative may collaborate later. This can be checked on validation for one
  later year only, and not at all for test.
- **Single institution.** Only collaborations between UTS authors are targets, and the graph covers UTS works.
  Results may not transfer to other institutions or to cross-institution collaborations.
- **GNN not retrained.** The GNNs were trained before the distance shortcut was found and were not retrained, so the
  GNN results understate what a correctly trained GNN could do, and the hybrid is built on a shortcut-trained model.

# 10. Conclusion

Under a strict evaluation, new UTS collaborations are moderately predictable. A LightGBM trained on the same kind of
pairs it is scored on beats Adamic-Adar by about 0.07 to 0.09 AUC; node2vec and the GNN tested here do not beat the
heuristic; and topical similarity adds a small, consistent gain beyond structure. The larger lesson is about
evaluation: unmatched activity, pooled newcomer composition and distance-matched training negatives each produce
convincing numbers that do not measure the task, and each was found only by building controls and reference scores
into the evaluation.

# References

All entries must be checked by the author before submission.

- [VERIFY] Adamic, L. A., and Adar, E. (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211-230.
- [VERIFY] Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- [VERIFY] Grover, A., and Leskovec, J. (2016). node2vec: Scalable feature learning for networks. *Proceedings of KDD 2016*, 855-864.
- [VERIFY] Hamilton, W. L., Ying, R., and Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- [VERIFY] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- [VERIFY] Katz, L. (1953). A new status index derived from sociometric analysis. *Psychometrika*, 18(1), 39-43.
- [VERIFY] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- [VERIFY] Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang, J., and Yin, D. (2023). Evaluating graph neural networks for link prediction: Current pitfalls and new benchmarking. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*.
- [VERIFY] Liben-Nowell, D., and Kleinberg, J. (2007). The link-prediction problem for social networks. *Journal of the American Society for Information Science and Technology*, 58(7), 1019-1031.
- [VERIFY] Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- [VERIFY] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., and Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
- [VERIFY] Newman, M. E. J. (2001). Clustering and preferential attachment in growing networks. *Physical Review E*, 64, 025102.
- [VERIFY] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press.
- [VERIFY] Priem, J., Piwowar, H., and Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.
- [VERIFY] Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of ICML 2017*.
- [VERIFY] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., and Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations (ICLR 2018)*.
- [VERIFY] Yang, Y., Lichtenwalter, R. N., and Chawla, N. V. (2015). Evaluating link prediction methods. *Knowledge and Information Systems*, 45(3), 751-782.
- [VERIFY] Ying, R., Bourgeois, D., You, J., Zitnik, M., and Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
- [VERIFY] Zhou, T., Lu, L., and Zhang, Y.-C. (2009). Predicting missing links via local information. *The European Physical Journal B*, 71, 623-630.
