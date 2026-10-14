"""Phase 5 diagnostic: unsupervised node2vec embedding-quality check on the TRAIN input graph only (no labels):
ROC-AUC of cosine similarity for 20k existing edges vs 20k random pairs, p = q = 1, seed 0.
Used to pick the learning rate (docs/decisions.md). -> report/node2vec/quality.csv"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
rows = []
from linkpred.data.splits import load_split
from linkpred.eval.metrics import roc_auc
from linkpred.models.node2vec import train_embeddings, cosine

sp = load_split("data/processed/splits", "train")
A = sp["A"].tocoo()
rng = np.random.default_rng(0)
m = A.row < A.col
eu, ev = A.row[m], A.col[m]
pick = rng.choice(len(eu), 20000, replace=False)
eu, ev = eu[pick], ev[pick]
n = sp["A"].shape[0]
ru, rv = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
base = dict(embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, batch_size=256)
for lr, ep in [(0.01, 1), (0.05, 1), (0.1, 1), (0.01, 3)]:
    t = time.time()
    E = train_embeddings(sp["A"], 1.0, 1.0, 0, {**base, "lr": lr, "epochs": ep})
    auc = roc_auc(cosine(E, eu, ev), cosine(E, ru, rv))
    rows.append({"lr": lr, "epochs": ep, "edge_vs_random_cosine_auc": round(auc, 4),
                 "mean_norm": round(float(np.linalg.norm(E, axis=1).mean()), 2), "seconds": round(time.time() - t)})
    print(rows[-1], flush=True)
out = Path("report/node2vec"); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "quality.csv", index=False)
