# Who Collaborates Next?

Temporal link prediction on a university co-authorship network: heuristics vs. LightGBM vs.
node2vec vs. GraphSAGE/GAT under a leak-free temporal split with hard negatives.

## Setup
```bash
cd collab-link-prediction          # all commands run from the repo root
conda create -n linkpred python=3.11 -y && conda activate linkpred
conda install -y llvm-openmp       # macOS: OpenMP runtime required by LightGBM
pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install -e .
make test
```

See `CLAUDE.md` for project rules and `docs/decisions.md` for the decision log.
