# Independent audit, Phases 0–6

_Audit date: 2026-10-15. Scope: everything committed up to `6ea999a`, plus the local data it was built
from. Method: intent was read from CLAUDE.md, `docs/` and `results/*.md`, then **verified against code
and data with independently written scripts** (kept outside the repo, as instructed; the essential
logic and outputs are quoted below). Nothing in the repo was changed except this file._

Severity: **critical** = changes a headline result or is leakage; **major** = changes a reported number
or a conclusion; **minor** = docs / cosmetic / latent.

## Summary

| ID | Severity | Finding | Headline affected? |
|---|---|---|---|
| — | critical | **None found.** No leakage anywhere; the splits, every pair and node feature and every model input re-derive exactly from pre-cutoff data. | — |
| M1 | major | The pair-focused training set lets distance alone separate the classes (40.5% of positives at distance ≥ 3, 0% of negatives); this, not "sharing a co-author", causes the collapse on random protocols. | No (lead metric unchanged; secondary numbers and a conclusion change) |
| M2 | major | Newcomer composition distorts every protocol and every cold-start breakdown in Phases 3–6, not only `hard_active`. | GNN headline; all cold-start conclusions |
| M3 | major | Test set reused for design decisions (activity and newcomer diagnostics computed on test, then protocols and model families chosen in response). | Not quantifiable; validation agrees directionally |
| M4 | major | node2vec conclusion is wrong: alignment works; the weak results come from the L1 operator, which won validation `hard_active` but transfers proximity poorly. | node2vec secondary numbers |
| M5 | major | "Significant in k/5 seeds" counts 5 correlated tests on the same pairs, with no multiplicity control; one stated conclusion flips with a single seed-averaged test. | Hybrid `rank_hard_active` verdict |
| m1 | minor | Stale or inconsistent numbers in `docs/decisions.md`. | — |
| m2 | minor | Training-log overfitting range misstated (0.065–0.105 should be 0.065–0.177). | — |
| m3 | minor | Untraceable numbers: Colab `run.log` not imported; node2vec quality CSV hand-transcribed. | — |
| m4 | minor | "Seed-averaged CI" = mean of per-seed bootstrap endpoints: mislabelled, numerically close to a proper seed-mean CI. | — |
| m5 | minor | 8 config keys are decorative (never read; values hard-coded). | — |
| m6 | minor | CLAUDE.md rules 2 and 3 do not describe what the code does. | — |
| m7 | minor | Data and GNN results are not reproducible from a fresh clone. | — |
| m8 | minor | Latent bug: heuristic context cache keyed on `id()`. | — |
| m9 | minor | Two different "activity-matched" gains (+0.042 / +0.033) quoted without explaining the different negative sets. | — |
| m10 | minor | Cutoff years were chosen using test-window label counts (data volume, not performance). | — |

---

## Major findings

### M1 — Pair-focused training data let distance alone separate the classes
**What is wrong.** `FM.training_pairs(..., active=True)` (`src/linkpred/models/feature_models.py`)
pairs **all** train positives with activity-matched negatives, which are distance 2 by construction.
40.5% of positives but 0% of negatives are at distance ≥ 3, so the model learns "distance ≥ 3 ⇒
positive". That is exactly what the Phase 4 decision forbids: "Training negatives must not let distance
alone separate the classes" (`docs/decisions.md`, "Phase 4 note: training negatives"). The decision
log's explanation of the collapse ("it learned that sharing [a co-author] signals no link") describes
the symptom, not the mechanism. The GNN `pair_focused` objective uses the same `training_pairs` call,
so it almost certainly has the same artefact (its random-protocol AUC is 0.28); this was not verified
separately.

**Evidence** (audit script `check_pf.py`, seed 0, test):
```
PF training data: positives n 2257 share at distance >= 3: 0.405 | negatives n 6771 share at distance >= 3: 0.0
test positives        {'d=2': (0.307, 1684), 'd=3': (0.995, 417), 'd=4': (0.995, 210), 'd=5': (0.995, 222)}
test random negatives {'d=2': (0.146, 25),   'd=3': (0.995, 123), 'd=4': (0.994, 435), 'd=5': (0.994, 1950)}
training label check: mean predicted prob on training positives vs negatives: 0.611 0.13   # no label/sign flip
```
Every pair at distance ≥ 3 scores ≈ 0.995 whatever its label, hence random ROC-AUC 0.229 and
rank_random MRR 0.043.

**Suggested fix.** Train the pair-focused models on distance-2 positives only (or add random negatives).
Effect estimate (audit, same hyperparameters, 5 seeds, no re-tuning): hard_active **0.8126** (reported
0.813), established-matched 0.8147, newcomer-matched 0.7808, **random 0.954** (reported 0.229).

**Headline impact.** None on `hard_active`. The reported pair-focused random / rank_random numbers and
the conclusion "it is a within-friend-of-friend ranker, not a general recommender" are artefacts of the
training set. A related but separate point: the Phase 4 rule chose random-only negatives (hard fraction
0), which lets distance separate the classes in the *other* direction; the decision note is not enforced
by any rule or test.

### M2 — Newcomer composition distorts every protocol and every cold-start breakdown (Phases 3–6)
**What is wrong.** Phase 6 found this for `hard_active` only. It is general: the share of pairs with a
newcomer (≤ 2 works before the cutoff) differs between positives and negatives under *every* protocol,
in different directions. Separately, the reported `cold_start` / `established` strata restrict only the
**positives** and keep all negatives, so they measure newcomer composition as much as model quality.

**Evidence** (audit script `check_newcomer.py`):

| test | positives (all) | positives (d2) | neg_random | neg_hard | neg_hard_active |
|---|---|---|---|---|---|
| share of pairs with a newcomer | 0.383 | 0.347 | 0.779 | 0.480 | 0.167 |

| ranking (test) | true partner is a newcomer | candidate is a newcomer |
|---|---|---|
| ranking (rank_random) | 0.255 | 0.537 |
| ranking_hard | 0.230 | 0.262 |
| ranking_hard_active | 0.231 | 0.086 |

A trivial score `−career_age_min` (prefer newer authors) reaches, on test: hard **0.615**, hard_active
**0.646**, rank_hard query_auc 0.575, rank_hard_active query_auc **0.599** (validation: 0.617, 0.646,
0.567, 0.596). Adamic-Adar alone reaches 0.706 on hard_active.

Cold-start breakdown, Adamic-Adar under `hard`, test: reported cold-start **0.675** vs established
**0.773** (gap 0.098). With negatives restricted to the same stratum: **0.706** vs **0.748** (gap
0.042). Validation: reported 0.712 vs 0.787, matched 0.744 vs 0.764.

**Suggested fix.** Report stratified results with *both* classes restricted to the stratum (as
`scripts/stratified_hard_active.py` does for `hard_active`) for every protocol, or sample negatives that
match the positives' newcomer composition. Re-word the cold-start conclusions in `results/baselines.md`,
`results/feature_models.md`, `results/node2vec.md` and the Phase 3–5 summaries.

**Headline impact.** The pair-focused LightGBM's lead holds (matched strata 0.815 / 0.781, +0.088 /
+0.057 over Adamic-Adar, `results/hard_active_stratified*.csv`). The best GNN's 0.687 is not
interpretable: it is mostly newcomer signal (Spearman with career age −0.73, with Adamic-Adar −0.02 on
hard_active pairs; audit script `check_gnn_general.py`), and it is significantly worse than Adamic-Adar
in both matched strata. Every "cold start is much harder" statement (e.g. Phase 3, AA 0.675 vs 0.773)
overstates the gap by about 2×.

### M3 — Test set reused for design decisions
**What is wrong.** No fitting, early stopping or hyperparameter selection reads test data (verified, see
"Verified" below). But the **evaluation design and model families** were chosen after test analyses:
- Phase 4: `scripts/activity_diagnostic.py` writes test rows; the decision that made `hard_active` the
  lead metric quotes the test result as its justification ("LightGBM's gain over Adamic-Adar shrank from
  +0.125 to +0.042 ROC-AUC", `docs/decisions.md`, "Activity-matched evaluation is standard").
- Phase 5: the pair-focused LightGBM was introduced in response to that finding.
- Phase 6: the newcomer diagnostic (`scripts/stratified_hard_active.py`) was computed on test and drives
  the open question about the protocol.

The test set is therefore no longer a clean confirmatory set for the claim "the pair-focused model beats
every other model on the lead metric".

**Suggested fix.** State this explicitly in the final report. Base future design decisions on
validation only (the validation counterparts exist and agree: activity-matched gain +0.021, 4/5 seeds;
validation newcomer shares 31% vs 16%). If possible, confirm the final ranking on a holdout never looked
at, e.g. the 2026 window once OpenAlex has indexed it.

**Headline impact.** Not quantifiable. The direction of every diagnostic matches on validation, so a
large reversal is unlikely, but the size of the lead should be read as optimistic.

### M4 — node2vec: alignment works; the operator choice explains the weak results
**What is wrong.** `docs/decisions.md` ("Phase 5 findings") states: "Raw cosine beats the aligned
classifier everywhere, so the Procrustes + logistic-regression pipeline loses information". The audit
finds the alignment code correct (`align_to`: `orthogonal_procrustes(emb[anchors], ref[anchors])` then
`emb @ R` maps the target split into the train space; anchors = authors with degree > 0 in both graphs,
67,287 for val and test). A Hadamard classifier **transfers almost losslessly**. The weak numbers come
from the **L1 operator**. It won the validation `hard_active` selection legitimately (at p = q = 2:
L1 0.605, L2 0.603, Hadamard 0.571, average 0.410, `results/node2vec_grid.csv`; a 0.034 margin over
Hadamard), but it captures proximity poorly, so it does badly on every other protocol.

**Evidence** (audit scripts `check_n2v*.py`, seed 0, p = q = 2, validation):
```
train->val transferred via Procrustes, hadamard: val random AUC 0.905
train->val transferred via Procrustes, l1:       val random AUC 0.665
within-val 5-fold LR (no transfer), raw hadamard 0.901 | raw l1 0.676 | L2-normalised l1 0.902
cosine (no classifier) 0.915
```
Per-node alignment is only approximate (median cosine between an author's aligned val vector and train
vector 0.21; relative residual 1.26), but that does not prevent transfer for the Hadamard operator.

**Suggested fix.** Correct the decision text. The selection was done properly on validation, so it should
not be overturned on test; but report all operators' validation results alongside, and consider
L2-normalising embeddings before the operators (within validation, normalised L1 reaches 0.902 on random).
Selection must stay on validation.

**Headline impact.** node2vec's `hard_active` conclusion is unaffected: no operator beats cosine on
validation (0.64), and all lose to Adamic-Adar. The reported secondary numbers (random 0.675,
rank_random MRR 0.158) reflect the L1 choice; Hadamard transfers at about 0.90 on validation random, but
it was not the validation-selected operator, so this is an explanation, not a correction.

### M5 — Significance reporting: "k/5 seeds" and multiplicity
**What is wrong.** Paired tests are run once per seed on the **same** test pairs and summarised as
"significant in k/5 seeds" (`feature_report.py`, `run_node2vec.write_report`, `gnn_analysis.write_report`).
The five tests share the test-sampling noise, so 5/5 is not five independent confirmations, and dozens of
comparisons are made per phase without multiplicity control. (The counting itself matches its
description: p < 0.05 and the stated sign, `feature_report.py`.)

**Evidence** (audit script `check_ci.py`, test, one paired bootstrap on seed-averaged scores, 1,000 resamples):
```
hybrid - lightgbm_pair_focused  rank_hard_active MRR: +0.0087 CI [+0.0007, +0.0173] p=0.040   (reported: "2/5 seeds, not reliable")
lightgbm - adamic_adar          rank_hard_active MRR: +0.0118 CI [-0.0015, +0.0258] p=0.080   (reported: 1/5; consistent)
```

**Suggested fix.** One paired bootstrap per comparison on seed-averaged scores (or a two-level bootstrap
over seeds and pairs), with Holm correction across the family of comparisons; keep per-seed results as
a robustness table.

**Headline impact.** The hybrid's `rank_hard_active` gain becomes nominally significant (p = 0.04,
which would not survive Holm across the Phase 6 family). Clear-cut results (p ≤ 0.001 in every seed) do
not change.

---

## Minor findings

**m1 — Stale or inconsistent numbers in `docs/decisions.md`.**
- Line 61: "(1,367 remain)" — the current cleaning log has `duplicate_title_year` 1,356 (after the later
  type exclusion).
- Lines 85–89: the superseded cutoff table (10,867 / 6,491 / 2,250 …) predates the type exclusion; the
  section is marked "superseded" but the numbers are not.
- Line 112: "UTS–UTS positives: 2,244 / 2,216 / 2,533" are the audit's counts (authors with ≥ 1 edge);
  the actual split positives are 2,257 / 2,225 / 2,533 (`data/processed/splits/*/meta.json`). Two
  "positive" counts are used without saying so.
- Line 171: `SeedSequence(...).spawn(3)`; the code now uses `spawn(9)` (stated in later entries only).
- Line 333: "The defensible pair-level gain of the feature models is about +0.02 to +0.04 AUC" is
  superseded by the pair-focused model (+0.107) without a note.
*Fix:* mark superseded values inline or update them.

**m2 — Training log misstates the overfitting range.** `docs/training_log.md` (Phase 6 Colab section):
"at lr 0.005 training loss falls to 0.065–0.105 (h64/h128, 2–3 layers)". Curves in `report/gnn/curves`:
h64_L2 0.065, h128_L2 0.081, h128_L3 0.105, **h64_L3 0.177**. *Fix:* "0.065–0.177", or restrict the
claim to the 2-layer configs.

**m3 — Untraceable or hand-transcribed numbers.** The Colab `run.log` was not imported, so the wall time
(5,361 s, 22:13:27 → 23:42:47) in `docs/training_log.md` and `decisions.md` cannot be checked from the
repo. `report/node2vec/quality.csv` was written by hand from a log, not by `scripts/node2vec_quality.py`
(as the Phase 5 summary disclosed). *Fix:* import `run.log` (into `report/gnn/`); regenerate the quality
CSV by running the script.

**m4 — "Seed-averaged bootstrap CIs" are mislabelled.** Learned-model intervals in
`results/feature_models.md`, `node2vec.md` and `gnn.md` are the **mean of the five per-seed percentile
endpoints**. Each per-seed interval captures test-sampling noise for one fitted model only; averaging
endpoints is not a valid interval for the seed-mean performance (it ignores seed variance, and
percentile endpoints do not average in general). Here the error is negligible because seed variance is
tiny: pair-focused LightGBM hard_active, reported [0.7981, 0.8261] vs a proper seed-mean bootstrap
[0.7988, 0.8281]; general LightGBM [0.7225, 0.7543] vs [0.7229, 0.7553]. *Fix:* bootstrap the seed-mean
(average the seeds' metric inside each resample) and report seed sd separately, as is already done.

**m5 — Decorative config keys.** Never read by name in `src/` or `scripts/`, with values hard-coded:
`splits.require_both_nodes_seen`, `splits.target_population`, `evaluation.headline`,
`features.distance_cap` (`DIST_CAP = 5`), `models.heuristics.katz.max_path_length` (walks of length 3),
`models.logreg.C` (`C=1.0` default), `models.lightgbm.early_stopping_rounds` (`patience=50` default),
`models.node2vec.operators` (`N2V.OPERATORS`). Values agree today; changing them does nothing. *Fix:*
read them or delete them.

**m6 — CLAUDE.md rules 2 and 3 do not describe the code.** Rule 2: "All randomness goes through
`linkpred.config.set_seed`". In fact splits use `SeedSequence`, LightGBM `random_state`, GNNs
`set_determinism`, node2vec `srand` and bootstraps `default_rng`. All of these derive from config seeds,
so the intent holds but the wording does not. Rule 3: GNN rows were written with pandas on Colab and only
re-validated through `write_results` at import. *Fix:* reword both rules.

**m7 — Not reproducible end to end from a fresh clone.** `configs/default.yaml` points `snapshot_path`
to an absolute path in another private project; `fetch_openalex.py` has never been run against the API;
the README does not mention `environment.yml` or how to obtain data. GNN results were produced with
torch 2.11 / numpy 2.1.3 on Colab, not the pinned 2.4.1 / 1.26.4 (recorded in
`results/gnn_versions.json`). *Fix:* document data acquisition, and note the GNN versions in the
README / report.

**m8 — Latent bug: heuristic context cache keyed on `id()`.** `linkpred/models/heuristics.py::context`
caches the `GraphContext` under `id(split["A"])`. If a split is garbage-collected and a new adjacency
reuses the id, stale degrees would be used silently. Not triggered in current scripts (baselines
re-derive bit-identically, see below). *Fix:* key on the object, or pass the context explicitly.

**m9 — Two "activity-matched" gains without explanation.** Phase 4 quotes LightGBM − AA = +0.042
(`neg_both_active` = the subset of the saved `neg_hard` whose authors are active,
`results/activity_diagnostic_paired.csv`); Phase 5 quotes +0.033 (`neg_hard_active`, a fresh 1:1 sample
from the full active pool, `results/feature_models_paired.csv`). Both are correct for their sets, but the
docs do not say why they differ.

**m10 — Cutoff years were chosen using test-window label counts.** `linkpred.data.audit.split_candidates`
counted positives in every window, including the eventual test window, to recommend cutoffs. This uses
data volume, not model performance, so it is acceptable, but it should be stated.

---

## Verified (no finding)

**Leakage, re-derived from scratch** (audit script `recompute_splits.py`; does not import
`linkpred.data.splits`):
- For each split, the following match an independent recomputation from `works`, `authorships` and
  `edges` exactly: the input-graph node set, UTS flags (UTS authorship with year < cutoff),
  `n_works_before`, degrees, every edge's weight / first year / last year (max edge year 2022 / 2023 /
  2024), the positives, and the distance-2 flags.
- Every pair in `neg_random`, `neg_hard`, `neg_hard_active` and the three train pools is UTS–UTS,
  unconnected, has no joint work of any size in the window, is not a positive and is not duplicated;
  hard pairs are at distance exactly 2; active pairs have both authors active in the window. 0
  violations.
- Pool sizes recomputed: hard 480,626 / 627,556 / 781,611 and active 104,933 / 127,941 / 141,009, equal
  to `meta.json`.
- Ranking sets: every head is the positive's `author_u`; `ranking` has exactly 100 eligible candidates
  per query; every `ranking_hard` / `ranking_hard_active` query has exactly min(100, independently
  recomputed pool) candidates, all from the pool; rank_hard queries are distance-2 positives. 0 mismatches.
- Hard negatives are a uniform sample of the pool: KS test of `n_common`, p = 1.000 / 0.998 / 0.533.

**Model inputs use only pre-cutoff works:**
- Pair features: 65 pairs (20 per split + 5 pair-focused training negatives; 35 with common neighbours)
  hand-computed from raw tables for all 27 features, max relative error ≤ 2.3e-7 (float32). Pair-focused
  training negatives' features use the 2023 cutoff.
- GNN node features: 20 nodes per split hand-computed, structural max |diff| ≤ 8.6e-8, text ≤ 6.1e-5
  (float16 storage).
- node2vec: `train_embeddings(A, …)` receives only the split's adjacency; the test-graph embedding
  retrained from scratch is bit-identical to the cache.
- GNN: `x` comes from the saved node features (standardised with train-split statistics) and
  `edge_index` from the split's adjacency; `assert_supervision_not_in_graph` runs on every fit.
- Hybrid out-of-fold: fold k's pairs are excluded from that fold's GNN training (`run_gnn.run_oof`),
  message passing uses the train graph, and all 5 × 9,028 scores are finite.
- Prediction-year activity appears only in `splits.py` (choosing `neg_hard_active`,
  `ranking_hard_active`, `train_pool_hard_active`). It is never in nodes, edges or features, and
  pair-focused training negatives come from the 2023 window only.

**No selection on test:** every read of test data in `scripts/` and `src/` (grep, all instances) only
scores or reports. All of the following use validation:
- the validation-best heuristic per protocol
- Katz beta and the recency half-life
- the negative mix and both LightGBM searches
- early stopping (LightGBM, the GNNs and the out-of-fold models)
- node2vec p, q and operator, and the base LightGBM for node2vec features
- the GNN grid choice and the best GNN

LightGBM early stopping really uses average precision: AP is the first metric registered, and
`best_iteration_` equals the AP argmax (159; 199 for the pair-focused model), not the logloss minimum (8).

**Statistics:**
- ROC-AUC and AP match sklearn on 300 heavily tied random cases (max |diff| 2e-16 / 0).
- Expected Hits@K and MRR match a 20,000-draw random tie-break simulation within its noise.
- `query_auc` matches direct counting.
- Paired bootstrap resamples the same indices for both models (a monotone-shifted copy gives difference
  0 with a [0, 0] interval).
- "k/5 seeds" is computed as described.

**Reproducibility:**
- A fresh clone of `origin/main` with an environment built from `environment.yml` (Python 3.11.17,
  torch 2.4.1, PyG 2.6.1, LightGBM 4.5.0) passes the tests: 136 passed, 50 data-dependent skipped.
- All 17 imported `linkpred` modules and the cross-imported scripts are tracked; no `.py` file is ignored.
- Re-runs are bit-identical (max |diff| ≤ 1.1e-16): Adamic-Adar on test (98 rows incl. CIs), LightGBM
  seeds 0 and 1 (98 rows each), and node2vec cosine seed 0 including embedding retraining.
- Different seeds give different outputs (LightGBM by up to 0.185 per score; node2vec embeddings differ).
- Regenerating `docs/data_audit.md`, `docs/splits.md`, `results/baselines.md`,
  `results/feature_models.md` and `results/gnn.md` from the CSVs reproduces the committed files exactly.

**Consistency:**
- Split years, the 25-author cap, the excluded types and the UTS-author definition agree across
  config, code and docs.
- Protocol definitions in code match `decisions.md`.
- Grid sizes agree: LightGBM 24 (and 24 pair-focused), node2vec 9 (p, q) × 4 operators, GNN 48 (the
  Phase 6 "40" was already corrected).
- Spot-checked decision entries match the code: ranking head = smaller author ID, uniform hard sampling,
  specialist ratio, logistic regression C = 1, GAT head concat/average, gradient clipping 5, weight decay
  5e-4, node2vec lr 0.05 / 1 epoch, ~1.07M directed edges (536,306 × 2).

## The four suspected items
1. **Procrustes alignment:** direction and anchors are correct; it transfers the Hadamard classifier
   almost losslessly. Aligned < cosine is explained by the L1 operator choice (M4), not by a bug.
2. **General GNNs near chance on `hard_active`:** real, not a bug. They score 0.896–0.946 on random and
   0.748–0.773 on hard (`results/gnn.csv`), and their scores track activity and degree (Spearman 0.73 with recent works, 0.65
   with preferential attachment, for the structural model). Inside `hard_active` those signals are
   inverted by the newcomer composition: preferential attachment alone scores 0.367 and career age alone
   0.354 (M2).
3. **Seed 4's early-stopped out-of-fold folds:** no detectable bias. Hybrid − PF on `hard_active` by seed:
   +0.0112 / +0.0113 / +0.0135 / +0.0125 / **+0.0109**; test AUC 0.8248 / 0.8242 / 0.8255 / 0.8252 / **0.8245**.
4. **Pair-focused LightGBM below chance on random protocols:** the training objective (M1), not a sign
   or label flip (training positives average 0.61, negatives 0.13).

## Which headline numbers would change
| Headline (test) | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| Pair-focused LightGBM hard_active 0.813, +0.107 vs AA | no (0.8126 trained on d2 only) | no (matched 0.815 / 0.781) | possibly optimistic, unquantified | — | no (p ≤ 0.001 in every seed) |
| Hybrid +0.012 hard_active; rank_hard_active "not reliable" | — | not re-checked by stratum except established (+0.009, seed means) | as above | — | rank_hard_active becomes p = 0.04 (single test, uncorrected) |
| Pair-focused random 0.229 / rank_random 0.043 ("collapses") | **yes** (random ≈ 0.954) | — | — | — | — |
| Best GNN hard_active 0.687 ("n.s. vs AA") | likely affected (same training set) | **yes**: significantly worse than AA in both matched strata | — | — | — |
| Cold-start gaps in Phases 3–5 (e.g. AA hard 0.675 vs 0.773) | — | **yes**: gap about halves (0.706 vs 0.748) | — | — | — |
| node2vec random 0.675 / rank_random 0.158 | — | — | — | explained, not changed (Hadamard ≈ 0.90, but L1 was the legitimate validation choice) | — |
| node2vec / GNN hard_active below AA | — | GNN: stronger conclusion | — | no | — |

---

## Resolution (2026-10-15, remediation session)
| ID | Resolution |
|---|---|
| M1 | Pair-focused LightGBM retrained on distance-2 positives only (same search, 5 seeds) and made the main model; old model kept as `lightgbm_pair_focused_shortcut` (appendix). Hybrid rebuilt on it with out-of-fold GNN scores restricted to distance-2 training pairs. Pair-focused GNN shortcut quantified from saved scores (`results/gnn_shortcut_check.csv`); no Colab retraining. |
| M2 | Within-group (newcomer-matched) evaluation for every pair-level protocol, macro headline, CIs and paired tests on the macro metric; `prefer_newer` reference row in every table; old positives-only cold-start tables moved to appendices; resampling alternative rejected (decisions.md). |
| M3 | Limitation added (`docs/data_audit.md`); validation diagnostics reported first; CLAUDE.md rule 8 (validation first, test last). |
| M4 | Decision log corrected; L1 remains the reported node2vec; Hadamard reported as a labelled sensitivity analysis. |
| M5 | Primary tests = one paired bootstrap on seed-averaged scores per comparison, Holm across a defined 20-test family (`scripts/final_comparison.py`); k/5 kept as a secondary column; seed-mean CIs for headline values; per-seed CI averages labelled. |
| m1 | Stale numbers corrected inline with notes. |
| m2 | Training-log range corrected (0.065–0.177). |
| m3 | `report/gnn/run.log` imported (import script copies it); `report/node2vec/quality.csv` generated by `scripts/node2vec_quality.py`. |
| m4 | Labels fixed; seed-mean CIs used for headline values. |
| m5 | 3 keys wired, 5 removed; test prevents decorative keys. |
| m6 | CLAUDE.md rules 2 and 3 rewritten to match the code. |
| m7 | `LINKPRED_SNAPSHOT_PATH` override, fetcher fallback, README "Rebuilding the data" incl. Colab versions. |
| m8 | Cache checked by object identity; test added. |
| m9 | Two activity-matched gains explained (decisions.md, feature_models.md). |
| m10 | Recorded in decisions.md. |
