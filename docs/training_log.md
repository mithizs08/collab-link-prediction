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

## Phase 6 — Colab GPU run
_To be filled in from `run.log`, `versions.json` and the training curves after the Colab run._
