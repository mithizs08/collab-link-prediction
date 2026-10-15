# Who Collaborates Next?

Temporal link prediction on a university co-authorship network: heuristics vs. LightGBM vs.
node2vec vs. GraphSAGE/GAT under a leak-free temporal split with hard negatives.

## Setup
Exact environment (recommended):
```bash
conda env create -f environment.yml    # run from the repo root; installs llvm-openmp, torch-cluster, -e .
conda activate linkpred
make test                              # data-dependent tests skip until the data is built
```
Or manually:
```bash
conda create -n linkpred python=3.11 -y && conda activate linkpred
conda install -y llvm-openmp       # macOS: OpenMP runtime required by LightGBM
pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install -e .
```

## Rebuilding the data from a fresh clone
`data/` is not in git. The pipeline needs OpenAlex works for UTS (institution `I114017466`), 2015–2025:
1. **From an existing snapshot** (a JSON array of OpenAlex work objects, as used for the reported results):
   `export LINKPRED_SNAPSHOT_PATH=/path/to/uts_works_2015_2025.json` (overrides `data.snapshot_path` in
   `configs/default.yaml`).
2. **Or download it**: `export OPENALEX_MAILTO=you@example.org` (and `OPENALEX_API_KEY` if you have one),
   then `python -m linkpred.data.fetch_openalex` — cursor-paged, cached page by page in
   `data/raw/openalex_pages/`, resumable; `make data` uses those pages when no snapshot path exists.
   (The fetcher is unit-tested but the reported results were built from the snapshot; a fresh download
   will differ slightly because OpenAlex keeps indexing.)
3. `make data splits features train baselines` — then `python scripts/final_comparison.py` for the
   primary results. GNNs train on Colab (`make gnn-bundle`, `notebooks/06_gnn_colab.ipynb`); the reported
   GNN results were produced with torch 2.11 / CUDA 13.0 / numpy 2.1.3 on a T4
   (`results/gnn_versions.json`), not the pinned local versions.

See `CLAUDE.md` for project rules and `docs/decisions.md` for the decision log.
