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

## 2026-10-14 — Build from the existing OpenAlex snapshot; no re-pull
- **Decision:** read `uts-research-intelligence/data/raw/uts_works_2015_2025.json` (63,514 works,
  2015–2025) read-only via `data.snapshot_path`; stream it with `ijson`.
- **Alternatives:** re-pull all UTS works from the API with `fetch_openalex.py`.
- **Reason:** the snapshot has every required field except `doi`, and DOIs are recoverable from
  `primary_location` for 96.7% of kept works. Its authorships are capped at 100 by the list API,
  which only touches works already excluded by the 25-author cap. Streaming keeps peak memory low
  on an 8 GB machine. `fetch_openalex.py` (cursor paging, mailto, retries, page cache) is written
  and unit-tested against a fake session but has not been run against the API.
- UTS institution ID `I114017466` verified against `/institutions/I114017466` (display name
  "University of Technology Sydney", ROR `03f0f6041`, no child institutions in lineage).

## 2026-10-14 — Deduplication rules
- **Decision:** drop works with the same normalised DOI; then drop works with the same normalised
  title + year that share at least one author ID, only for titles of ≥ 5 words. The preferred copy
  is a non-preprint, then has an abstract, then has more authors, then the lowest work ID.
- **Alternatives:** DOI only; title + year without guards.
- **Reason:** inspecting the 1,497 unguarded title + year matches showed mostly true duplicates
  (preprint vs. published, Angewandte German/English editions, multi-journal consensus
  statements, copies without DOI) but also false ones on generic titles ("Letter from the
  editors" ×23, "Test" ×12, "Introduction" ×10). The 5-word guard removed those (1,367 remain).
  Preprints and published versions in *different* years are deliberately not merged: the
  collaboration was public from the earlier year.

## 2026-10-14 — Author cap counts the full byline
- **Decision:** `n_authors` = number of authorships including authors without an OpenAlex ID;
  works with `n_authors > 25` stay in `works.parquet` but produce no edges.
- **Reason:** the cap describes how a paper was written, not what OpenAlex resolved.

## 2026-10-14 — authors.parquet is descriptive only
- **Decision:** `authors.parquet` (first/last year, n_works, primary institution) aggregates all
  years and must never be used as a model feature.
- **Reason:** it contains future information. Per-period author features will be recomputed
  from works with year < cutoff. Likewise, UTS status in the audit is time-respecting ("had a UTS
  authorship before the cutoff").

## 2026-10-14 — Recommended split cutoffs (superseded: accepted, see below)
- **Recommendation:** one-year label windows ending in the most recent data:
  - train: cutoff 2023, window 2023 (graph = works 2015–2022)
  - val: cutoff 2024, window 2024
  - test: cutoff 2025, window 2025
- **Eligible positives** (`report/data_audit/split_candidates.csv`, both authors in the graph
  before the cutoff):

  | period | all pairs both seen | ≥1 UTS author | UTS–UTS |
  |---|---|---|---|
  | train 2023 | 10,867 | 6,491 | 2,250 |
  | val 2024 | 10,894 | 6,779 | 2,229 |
  | test 2025 | 11,724 | 7,238 | 2,539 |

- **Reason:** (1) Every window has more than 2,000 positives even in the strictest (UTS–UTS)
  population, enough for stable AUC/AP: for an AUC near 0.9 with 1:1 negatives the standard
  error is about 0.005, so differences of a few hundredths between models are detectable. (2) Equal one-year horizons
  keep train, validation and test the same task. (3) Late cutoffs give the longest observed
  history (8 years before the train cutoff) and avoid the left-censored 2015–2016 period.
  (4) Yearly new-pair counts are stable from 2022 on (73–80k all-author, 5.7–6.5k UTS–UTS), so
  the three windows have similar difficulty.
- **Alternatives considered:** a two-year test window 2024–2025 (more positives: 4,109 UTS–UTS,
  but a different horizon from training); earlier cutoffs such as 2021/2022/2023 (keeps 2024–2025
  unused or needs a horizon mismatch, and gives less history).
- **Caveats:** 2025 may still be slightly incomplete in OpenAlex (pulled Aug 2026). If more
  training signal is needed, Phase 2 can stack earlier one-year training windows (cutoffs
  2019–2022) without touching val/test; that is a separate decision.
- **Open (for review):** the primary prediction population, UTS–UTS vs. all pairs with both
  authors seen (see `docs/data_audit.md`, caveat 2).

## 2026-10-14 — Split cutoffs accepted and written to config
- **Decision (reviewed by project owner):** one-year windows. Train: graph 2015–2022 → predict
  2023 (`cutoff: 2023`); validation: graph 2015–2023 → predict 2024 (`cutoff: 2024`); test: graph
  2015–2024 → predict 2025 (`cutoff: 2025`).
- **Reason:** as in the recommendation above. Counts after the work-type exclusion are in
  `report/data_audit/configured_splits.csv` (UTS–UTS positives: 2,244 / 2,216 / 2,533).
- 2025 indexing lag is listed under Limitations in `docs/data_audit.md`.

## 2026-10-14 — Prediction targets are UTS–UTS pairs only
- **Decision (project owner):** positives and negatives are restricted to pairs of UTS authors.
  A UTS author for a split has at least one UTS-affiliated authorship in a work with
  year < cutoff, i.e. at or before the split's last graph year (2022 / 2023 / 2024). Affiliations
  from the label window itself are never used: that would select authors on future information.
  Both authors must also be in the input graph before the cutoff (`require_both_nodes_seen`).
  The input graph (heuristics, features, message passing) still uses ALL authors and edges with
  year < cutoff.
- **Alternatives:** all pairs with both authors seen; pairs with at least one UTS author.
- **Reason:** the dataset contains every work with a UTS author, so any collaboration between two
  UTS-affiliated authors is guaranteed to be observed, which makes both positive and negative labels
  reliable. For pairs involving a non-UTS author, collaborations on non-UTS papers are invisible,
  so negatives would contain hidden false negatives. All-pairs positive counts stay in the audit
  for context.

## 2026-10-14 — Drop erratum, peer-review and paratext works
- **Decision (project owner):** works of these types are removed before deduplication and edge
  building and counted in the cleaning log (`excluded_type_dropped`, 658: 283 errata, 242
  paratext, 133 peer reviews). Editorials and letters are kept.
- **Reason:** these types repeat or annotate other works rather than record a collaboration.
  Removing them before deduplication also guarantees an erratum is never kept as the canonical copy.
- **Side effect:** 118 of the previously over-cap works were of these types, so over-cap works
  fall from 623 to 505.

## 2026-10-14 — Planned sensitivity check (Phase 7): disambiguation
- **Plan:** rerun the best model and the best heuristic with every author ID flagged in
  `data/processed/disambiguation_flags.parquet` removed (from the input graph and from the
  positive/negative pairs), on the same splits and seeds, and report both results side by side.
- **Reason:** 1,011 (name, UTS) groups map to more than one ID (2,689 IDs), and several top hubs are
  flagged. Split profiles can turn continuing collaborations into false "new" positives; merged
  profiles create artificial hubs that favour degree-based methods. The check shows whether
  conclusions depend on these profiles. The flag is a heuristic (same name, same institution), so
  removal also drops some correctly identified authors; this will be stated with the result.

## 2026-10-14 — Cutoff interpretation approved
- **Decision (project owner):** a split's input graph and all node attributes, including UTS
  status, use only works published before that split's cutoff year.

## 2026-10-14 — Split construction details (Phase 2)
- **Isolated authors are nodes.** The input graph contains every author with a work before the
  cutoff, including authors with no co-author yet (solo or only hyper-authored works), as
  specified. 12–27 positives per split have such an endpoint; they count as disconnected.
  *Alternative:* only authors with ≥ 1 edge (the Phase 1 audit's "seen"). Effect: 2,257 / 2,225 /
  2,533 positives instead of the audit's 2,244 / 2,216 / 2,533.
- **"First co-authorship" counts works of any size.** Edges come from ≤25-author works only, but a
  pair whose first joint work was a hyper-authored paper before the cutoff is *not* a positive
  (14–21 pairs per split), and any pair with a joint hyper-authored work in the prediction year is
  excluded from negatives. *Reason:* otherwise a collaboration that already happened would be
  labelled "new", or a real collaboration labelled "none".
- **Input-graph edge weight** = number of distinct joint ≤25-author works before the cutoff.
- **Hard negatives are uniform over the full distance-2 pool** (480k–782k eligible pairs, enumerated
  exactly via a sparse product), not weighted by common neighbours. *Alternative:* sampling via
  random 2-step walks, which over-represents hubs.
- **Ranking head** is the positive's endpoint with the smaller author ID: fixed and unrelated to
  any attribute, so effectively random. Candidates exclude u's input-graph neighbours and anyone
  with a joint work with u in the prediction year (which includes all of u's positive partners).
- **Seeding:** each split uses `SeedSequence([cfg.seed, split_index]).spawn(3)`, one independent
  stream for random negatives, hard negatives and ranking candidates, so changing one sampler never
  changes the others. Negatives are a fixed evaluation set built once from `seed`; the five model
  seeds affect training only. This keeps rule 2 (all randomness from the config seed).
- **Leakage tests were mutation-checked:** deliberately reintroducing each leak (future edges and
  nodes, label-year UTS status, window collaborators as negatives, unfiltered hard negatives,
  connected positives, unseeded sampling) in a scratch copy made `tests/test_leakage.py` fail.

## 2026-10-14 — Evaluation protocols (fixes a distance artefact)
- **Problem:** all hard negatives are at distance exactly 2, but 34–41% of positives are at
  distance ≥ 3. In "all positives vs hard negatives", distance ≥ 3 identifies a positive
  perfectly, so a model given shortest-path distance (or anything correlated with it) would learn
  an artefact of the evaluation design rather than the task.
- **Decision (project owner):** every model from now on is evaluated with five protocols on the
  same saved pairs (`linkpred.eval.protocols`):
  - `hard`: distance-2 positives vs hard negatives (like-for-like). **Headline.**
  - `rank_random`: Hits@20/50/100 and MRR for each positive against its 100 random ranking candidates.
  - `rank_hard`: each distance-2 positive (u, v) against up to 100 candidates (u, w), w a UTS
    author at distance exactly 2 from u in the input graph, not connected to u and with no joint
    work with u in the prediction year. Seeded (fourth independent stream; the first three, and
    therefore all previously built pairs, are unchanged). Built in `splits.py`, file
    `ranking_hard.parquet`, covered by `tests/test_leakage.py` and mutation-checked.
  - `random`: all positives vs random negatives (easy; comparability with the literature).
  - `diagnostic_all_vs_hard`: all positives vs hard negatives, appendix only, with a warning.
- **rank_hard with fewer than 100 candidates:** 18–22% of distance-2 positives have fewer than 100
  eligible candidates (one train query has none and is dropped; counts in `docs/splits.md`). A
  positive ranked among fewer candidates gets a better rank by construction, so MRR / Hits@K are
  not comparable across candidate counts. **Choice:** rank against however many exist and report
  strata: `full_candidates` (exactly 100; the headline rank_hard numbers), `partial_candidates`
  (reported separately), and `all` with only `query_auc` = the fraction of a query's candidates
  the positive beats (ties ½), which does not depend on the candidate count.
  *Alternatives:* drop partial queries (loses information and biases toward well-connected heads);
  pad with random candidates (mixes in easy negatives); normalised MRR (no standard definition).

## 2026-10-14 — Tie handling in metrics
- **Decision:** scores are snapped to ~12 significant digits so float-rounding differences count
  as ties. ROC-AUC counts tied pairs as ½ (average ranks). AP treats tied scores as one threshold.
  Hits@K and MRR use the expectation under uniformly random tie-breaking: with g candidates above
  and e tied, the rank is uniform on {g+1, …, g+e+1}.
- **Alternatives:** optimistic ranks (g+1) or pessimistic ranks (g+e+1).
- **Reason:** heuristics give many pairs a score of exactly 0. Optimistic ranking would reward a
  model for scoring everything 0; pessimistic ranking would punish it for ties it cannot break.
  With the expectation, a constant scorer gets exactly the random-ranker values (unit-tested).

## 2026-10-14 — Results schema extended
- **Decision:** results CSVs now have
  `model, split, protocol, subset, negative_type, seed, metric, value, ci_low, ci_high`.
  `protocol` is one of the five protocols; `subset` is `all` or a stratum (`distance=3`,
  `cold_start`, `full_candidates`, …); `ci_low`/`ci_high` are bootstrap 95% intervals (empty when
  not computed). Deterministic models (heuristics) record the config seed, i.e. the seed that
  fixed the evaluation pairs. Counts (`n_positives`, `n_negatives`, `n_queries`) are rows too.
- **Reason:** the original six columns could not hold the protocol, strata or intervals. CLAUDE.md
  rule 3 updated accordingly.

## 2026-10-14 — Bootstrap design
- **Decision:** 1,000 percentile-bootstrap resamples. Classification protocols resample positives
  and negatives separately (stratified, fixed class sizes); ranking protocols resample queries.
  Resample indices depend on (seed, protocol, split) but not on the model, so all models see
  identical resamples. Paired tests apply the same indices to both models; two-sided p-value =
  2·min(P(Δ ≤ 0), P(Δ ≥ 0)), floored at 1/1,000.
- **Reporting the "best" heuristic:** the best heuristic per protocol is chosen on **validation**
  by the protocol's headline metric, then reported on test and compared (paired) with every other
  heuristic on test. Picking the winner on test would be a mild form of tuning on test.

## 2026-10-14 — Truncated Katz beta selected on validation
- **Decision:** walks of length 1–3 on the binary input-graph adjacency; beta chosen from
  {1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5} by validation `hard` ROC-AUC (ties toward the smaller beta).
- **Note:** for the unconnected pairs we score, Katz ranks by A² + β·A³, so beta only trades
  common neighbours against length-3 walks. Result in `results/katz_beta_selection.csv`.

## 2026-10-14 — Phase 4 note: training negatives
- Training negatives must not let distance alone separate the classes (e.g. training only on
  random negatives, which are almost never at distance 2, while positives mostly are). The
  training negative mix (random / hard / both) will be chosen on validation under the `hard`
  protocol, not fixed in advance.

## 2026-10-14 — Phase 4 reporting change
- **Decision (project owner):** rank_hard reports Hits@10/20/50 and MRR; Hits@100 is dropped
  (near-trivial with ≤ 101 items). rank_random keeps Hits@20/50/100. rank_hard also gains
  `cold_start` / `established` strata (query_auc only, comparable across candidate counts).
  Baselines were re-run with this reporting; earlier headline values are unchanged.

## 2026-10-14 — Pair features (Phase 4)
- **Decision:** three families computed per split from works published before its cutoff only
  (`linkpred.features.pair_features`); all symmetric in (u, v) (per-author values enter as min/max).
  - *Structural:* CN, Jaccard, AA, RA, PA, truncated Katz (beta 0.005 from Phase 3), degree min/max,
    distance capped at 5 (computed with a 5-hop limited search) + `disconnected` (different connected
    component), same Louvain community (networkx, weighted by joint works, seed = config seed),
    shared institutions (from pre-cutoff authorships; always ≥ 1 for UTS–UTS pairs because UTS
    itself counts), common neighbours split into UTS / non-UTS.
  - *Temporal:* career age (years since first work in the data; left-censored at 2015), works in
    the two years before the cutoff, years since last work; recency-weighted CN / AA with weight
    0.5^((c − last_uw)/h)·0.5^((c − last_vw)/h); years since the most recent shared collaboration,
    defined as c − max_w min(last_uw, last_vw) (the most recent time both ties to a shared
    co-author were active), 11 when there is none.
  - *Text:* author vector = mean of unit embeddings (all-MiniLM-L6-v2, title + abstract or title
    only) of the author's pre-cutoff works; features = cosine of author vectors, similarity of each
    author to the mean vector of the *other* author's co-authors (min/max), both-have-text flag.
    Interpretation note: "similarity to the other's shared co-authors" was implemented as the
    other author's full co-author neighbourhood, because common neighbours do not exist for pairs
    at distance ≥ 3.
- **Half-life h** = 2 years, chosen on validation from {0.5, 1, 2, 3, 5, 10} by the `hard` ROC-AUC of
  recency-weighted AA alone (`results/half_life_selection.csv`).
- **Leakage guard:** `FeatureContext` receives the full tables and applies the only time filter
  itself; `tests/test_features.py` recomputes every feature after deleting all works in or after
  the cutoff (and scrambling future works' embeddings) and requires identical output. Three
  injected leaks (label-year authorships, unfiltered recent-work counts, future works in author
  text vectors) each made it fail.

## 2026-10-14 — Features considered and rejected for leakage
- `authors.parquet` aggregates (first/last year, n_works, primary institution): computed over all
  years, so they contain the future. Replaced by per-split pre-cutoff recomputation.
- OpenAlex `cited_by_count`, `counts_by_year`: measured at snapshot time (2026), after every cutoff.
- OpenAlex `last_known_institution` / current affiliation: describes the author's state at snapshot time.
- OpenAlex topics / concepts per work: assigned by OpenAlex classifiers trained recently; not used
  (the sentence embedding carries the same caveat, documented as a limitation, but is computed by us
  from the work's own text only).
- Degrees, communities, distances or embeddings computed on the full 2015–2025 graph: would encode
  the label-window edges. All computed on the split's input graph.
- Normalisation statistics (StandardScaler) fitted on validation or test pairs: fitted on training
  pairs only, inside the model pipeline.
- Activity inside the label window (e.g. "both authors published in the prediction year", window
  co-authors, triadic closure within the window): direct label leakage.
- Author UTS status from any work in or after the cutoff: already excluded by the split definition.
- Any hyperparameter (Katz beta, half-life, LightGBM settings, negative mix) chosen on test.

## 2026-10-14 — Training negative mix and model selection (Phase 4)
- **Grid:** hard fraction {0, 0.5, 1} × ratio {1:1, 1:3}, from train-split pools of 3 × positives per
  type (new seed streams 5 and 6; evaluation pairs unchanged).
- **Rule, fixed before running:** highest validation `hard` ROC-AUC among mixes whose validation
  rank_random MRR is ≥ 90% of the best mix's; ties go to fewer hard negatives, then the lower ratio.
  *Reason:* a hard-heavy mix teaches the model that sharing a co-author signals a negative, which
  can still score well on `hard` while collapsing on random candidates.
- **LightGBM:** early stopping on validation `hard` AP; search over num_leaves × learning_rate ×
  min_child_samples by validation `hard` ROC-AUC. The grid was widened once (adding 3 leaves and
  min_child 300) after a dry run picked the most regularised corner of the original grid. Decided
  on validation only, before any test evaluation.
- **Distance-2 specialist:** LightGBM with the same settings trained on distance-2 positives vs hard
  negatives only (ratio as the chosen general mix), compared with the general model on `hard` and
  `rank_hard`.
- **Logistic regression:** standardised features with log1p on heavy-tailed counts, C = 1, no search.
  lbfgs is deterministic, so its five seeds give identical results (sd 0).
- **Paired tests:** each final model and seed vs the validation-best heuristic per protocol, plus
  Adamic-Adar on `hard` and `rank_hard`; summarised as the number of seeds with p < 0.05.

## 2026-10-14 — Finding: much of the `hard` gain is author activity
- **Observation:** a positive requires both authors to publish in the prediction year; a hard
  negative does not, and only 18% of test hard negatives (467 / 2,533; 424 / 2,225 on validation)
  have both authors publishing anything in that year. Activity features (recent works, years since
  last work) therefore separate classes partly by predicting *who keeps publishing*.
- **Diagnostic** (`scripts/activity_diagnostic.py`, `results/activity_diagnostic*.csv`): `hard`
  re-scored against only the hard negatives whose two authors are both active in the prediction
  year. LightGBM's gain over Adamic-Adar shrinks from +0.125 to +0.042 ROC-AUC on test (still
  significant in 5/5 seeds; +0.021 on validation, 4/5). The temporal-only model's gain vanishes
  (−0.002, p ≈ 0.8), and the structural-only LightGBM becomes significantly *worse* than
  Adamic-Adar (−0.025).
- **Status:** the subset uses label-window information to choose evaluation pairs (never as a
  feature), so it is an analysis, not a protocol. The headline `hard` numbers stand as defined,
  but must always be reported together with this diagnostic. Whether to make an activity-matched
  protocol standard for Phase 5+ is an open question for the project owner.
- **Interpretation:** activity is legitimately known at the cutoff and is useful for forecasting,
  but "who will still be publishing" is a different and easier question than "which pairs will
  collaborate". The defensible pair-level gain of the feature models is about +0.02 to +0.04 AUC.

## 2026-10-14 — Note on feature importance
- LightGBM gain importance is measured on the training data, where negatives are random and
  structure (Katz, distance) separates classes easily, so it overstates structural features
  relative to their value on the `hard` protocol. The ablation, not gain importance, is the
  evidence for which family matters.
