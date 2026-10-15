# Training log

An honest running record of training problems (overfitting, loss blow-ups, memory, slow epochs,
reproducibility) and how each was handled. Newest at the bottom. Numbers come from the logs or
results files named in each entry.

## Phase 5 — node2vec (2026-10-14)
- **Slow epochs.** ~100 s per epoch on the 75k–100k-node graphs (CPU, torch-cluster walks).
  5 epochs × 33 trainings would have taken > 5 h. **Fix:** 1 epoch of 10 walks per node (the
  original node2vec protocol).
- **Under-training.** First dry run: rank_random MRR ≈ 0.16, i.e. near-random embeddings. PyG
  initialises embeddings N(0, 1) and lr 0.01 barely moves them in one epoch. **Fix:** unsupervised
  check on the train graph (edge-vs-random cosine AUC 0.868 at lr 0.01 → 0.997 at lr 0.05);
  lr 0.05. `report/node2vec/quality.csv`.
- **Non-reproducible biased walks.** p/q ≠ 1 walks draw from C `rand()`, not torch's generator.
  **Fix:** `srand(seed)`; regression test.
- **Stale cache risk.** Embedding cache keys did not include lr. **Fix:** key on every
  hyperparameter, old cache deleted.

## Phase 6 — GNNs, local CPU preparation (2026-10-15)
- **GAT on CPU is slow, and slower with deterministic algorithms.** One GAT forward pass on the
  train graph (75k nodes, 778k directed edges, hidden 128, 4 heads): ~17 s on the first call
  (warm-up), then 2–3 s; with `torch.use_deterministic_algorithms(True)` ~15 s, because scatter
  falls back to a deterministic, slower path. GraphSAGE: ~0.5–1.1 s per training step. **Handling:**
  CPU is used only for the dry run (2 epochs, 1 config per cell); real training runs on a Colab GPU.
  The GPU cost of deterministic mode is unknown until the Colab speed check (notebook cell 7).
- **CPU determinism.** The same-seed re-run in the dry run reproduced scores exactly
  (max |Δscore| = 0). GPU variation is measured by the determinism stage of `run_gnn.py`
  (`results/gnn_determinism.csv`) and will be recorded here after the Colab run.
- **Dry run** (`run_gnn.py --dry`, CPU): every stage (grid, final, dot ablation, determinism,
  out-of-fold) completed in 229 s; resume skipped all 20 completed units in 2 s; import and local
  analysis ran end to end. Dry-run metrics (2 epochs) are meaningless and were not imported into
  `results/`.

## Phase 6 — Colab GPU run (2026-10-15)
Source: `report/gnn/run.log` (imported from the Colab results folder, audit m3), per-run curves in
`report/gnn/curves/`, `results/gnn_versions.json`.

- **Environment and wall time.** Tesla T4, Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, cuDNN 9.2.7,
  torch_geometric 2.6.1, numpy 2.1.3 (Colab's preinstalled torch, not the local 2.4.1). One session,
  22:13:27 → 23:42:47, 5,361 s (≈ 89 min) for 48 grid + 40 final + 10 ablation + 1 determinism +
  25 out-of-fold trainings — faster than the 2–2.5 h estimate.
- **Grid size (corrected).** The config specifies 8 SAGE + 4 GAT = 12 configs per objective ×
  node-feature setting, 48 in total, and 48 ran. The pre-run estimate and the decisions entry said
  10 / 40 — a counting error in the prose, now corrected; the code always ran what the config said.
- **Overfitting in general SAGE + text configs.** With text node features the general objective
  overfits: at lr 0.005 training loss falls to 0.065–0.177 (h64_L2 0.065, h128_L2 0.081, h128_L3 0.105,
  h64_L3 0.177; corrected per audit m2, `report/gnn/curves/`) while validation `hard` ROC-AUC plateaus
  at 0.75–0.765; the final `gnn_general_sage_text` runs reach a training loss
  of 0.051. Early stopping (patience 30) keeps the best validation epoch, so the reported models are
  the pre-overfitting checkpoints, but the gap shows that the 384-d text input lets the model memorise
  the ~9k training pairs. Not fixed further (dropout and weight decay were held fixed in the grid);
  a stronger regulariser is an option if GNNs are revisited.
- **Dot-decoder loss blow-ups (pair-focused ablation).** Epoch-0 loss was 36–71 for
  `gnn_pair_focused_sage_structural_dot` (seeds 0–4: 50.8, 42.0, 36.4, 71.1, 55.3) and never fell
  below 1.25 — worse than the 0.69 of a constant predictor. The raw inner product of unnormalised
  128-d SAGE embeddings gives logits in the tens, so the sigmoid saturates and BCE explodes; gradient
  clipping (norm 5) kept training finite but the model stayed poor (best epochs 6–30). The general dot
  runs started lower (0.92–0.97) but stopped at their best epoch 2–3 in 3 of 5 seeds. Not fixed: the
  dot decoder is an ablation and is reported as it behaved; the MLP decoder does not have the problem.
- **GAT slower and weaker than SAGE.** Median 0.28 s per epoch vs 0.10 s for SAGE (≈ 2.8×); the 16 GAT
  grid configs took 636 s vs 401 s for the 32 SAGE ones. Best validation score in the grid was lower
  for GAT in all four objective × feature cells (e.g. pair-focused structural `hard_active` AP 0.604
  vs 0.665; general text `hard` ROC-AUC 0.743 vs 0.765), so no final GAT model was the best GNN.
- **Early-stopped out-of-fold folds.** In seed 4 of the out-of-fold stage, folds 1 and 2 stopped
  around epoch 50–55 (validation AP peaked near epoch 25, dipped, and patience ran out) while the
  other 23 folds trained ~200–250 epochs; their out-of-fold scores are visibly flatter (sd 0.43 vs
  0.55–0.69). Every training pair still has an out-of-fold score (5 × 9,028, all finite). Kept as is
  and noted as a caveat for hybrid seed 4. The out-of-fold models do not save curves; the epochs are
  read from `run.log`.
- **Misleading log line (fixed).** `run_gnn.py` printed `oof … fold k` *after* fold k trained, so each
  fold's epoch lines appear above its label; the last label ("seed 4 fold 4", no epochs after it) looked
  like an untrained fold, and under "determinism re-run" the second trace was actually out-of-fold
  seed 0 fold 0. The message now says "fold k done (best epoch …, n epochs)".
- **GPU determinism.** With `use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, re-running `gnn_pair_focused_sage_structural` seed 0 reproduced
  the original final run exactly: all 231 epochs of training loss and validation AP identical, same best
  epoch (200), max |Δscore| = 0 on val and test (`results/gnn_determinism.csv`). `run.log` contains no
  "nondeterministic" warnings (warn_only would have printed one per op), including during GAT
  training. Caveat: only this SAGE config was re-run; GAT determinism is inferred from the absence of
  warnings, not measured.
- **Cosmetic warning (fixed).** `float(loss)` on a tensor requiring grad triggered a UserWarning;
  replaced with `loss.item()`.

## Audit remediation re-runs (2026-10-15)
- **Memory pressure stall.** During the feature-model re-run one seed took ~16 min instead of ~35 s: the
  8 GB laptop was swapping (6.2 of 7 GB swap used, process mostly paged out). No effect on results (all
  runs deterministic); later seeds ran at normal speed. Keep other heavy apps closed during full re-runs.
- **GNN scores re-evaluated, not retrained.** All saved Colab scores were re-evaluated locally with the
  current protocols; pooled rows match the Colab evaluation to 1.1e-16 (`results/gnn_reeval_check.json`).
