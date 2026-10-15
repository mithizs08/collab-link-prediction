"""GNN link predictors (PyTorch Geometric): GraphSAGE / GAT encoder + MLP or dot decoder.

Leakage design (docs/decisions.md, Phase 6):
- Message passing runs on a split's input graph only (edges from works before its cutoff).
- Node features come from data/processed/node_features (built by the Phase 4 FeatureContext code).
- Supervision pairs are the TRAIN split's positives and pool negatives. Positives are new links
  (not connected before 2023) and negatives are unconnected by construction, so no supervision
  pair is a message-passing edge; `assert_supervision_not_in_graph` enforces it at train time.
- SAGE / GAT are inductive: the trained weights are applied to the validation / test input
  graphs directly. Nothing is learned per node, so -- unlike node2vec -- there is no embedding
  space to align across splits.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from linkpred.eval.metrics import average_precision, roc_auc


# ---------------------------------------------------------------- determinism

def set_determinism(seed: int) -> None:
    """Seed everything and request deterministic kernels where PyTorch / PyG support them.

    warn_only=True: PyG's scatter-based aggregation (index_add / scatter_reduce on CUDA) has no
    deterministic implementation, so GPU runs with the same seed can differ slightly. The size of
    that variation is measured and reported (run_gnn.py, determinism check), not hidden.
    """
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- data

@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    n: int = field(init=False)

    def __post_init__(self):
        self.n = self.x.shape[0]

    def to(self, device):
        return GraphData(self.x.to(device), self.edge_index.to(device))


def edge_index_from_adjacency(A: sp.csr_matrix) -> torch.Tensor:
    B = A.tocoo()
    return torch.tensor(np.vstack([B.row, B.col]), dtype=torch.long)


def make_node_features(structural: np.ndarray, text: np.ndarray | None, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise the structural block with TRAIN-split statistics; append text vectors if given."""
    s = (structural - mean) / np.where(std > 0, std, 1.0)
    return np.hstack([s, text]).astype(np.float32) if text is not None else s.astype(np.float32)


def assert_supervision_not_in_graph(A: sp.csr_matrix, u: np.ndarray, v: np.ndarray) -> None:
    """No supervision pair may be a message-passing edge (it would leak its own label)."""
    hits = np.asarray(A[u, v]).ravel() != 0
    if hits.any():
        raise AssertionError(f"{int(hits.sum())} supervision pairs are edges of the input graph")


# ---------------------------------------------------------------- model

class Encoder(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int, layers: int, dropout: float, heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv, SAGEConv

        self.kind, self.dropout = kind, dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for i in range(layers):
            if kind == "sage":
                self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            elif kind == "gat":
                # Hidden layers concatenate heads (hidden // heads each); the last layer averages.
                last = i == layers - 1
                self.convs.append(GATConv(dims[i], dims[i + 1] if last else dims[i + 1] // heads,
                                          heads=heads, concat=not last, dropout=dropout))
            else:
                raise ValueError(kind)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x) if self.kind == "gat" else F.relu(x)
        return x


class Decoder(nn.Module):
    """'mlp': MLP over z_u * z_v (Hadamard, symmetric); 'dot': <z_u, z_v> (ablation)."""

    def __init__(self, kind: str, hidden: int, dropout: float):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        elif kind != "dot":
            raise ValueError(kind)

    def forward(self, zu, zv):
        if self.kind == "dot":
            return (zu * zv).sum(-1)
        return self.mlp(zu * zv).squeeze(-1)


class LinkPredictor(nn.Module):
    def __init__(self, encoder: str, decoder: str, in_dim: int, hidden: int, layers: int, dropout: float, heads: int = 4):
        super().__init__()
        self.encoder = Encoder(encoder, in_dim, hidden, layers, dropout, heads)
        self.decoder = Decoder(decoder, hidden, dropout)

    def forward(self, g: GraphData, u, v):
        z = self.encoder(g.x, g.edge_index)
        return self.decoder(z[u], z[v])

    @torch.no_grad()
    def score(self, g: GraphData, u: np.ndarray, v: np.ndarray, batch: int = 200_000) -> np.ndarray:
        self.eval()
        z = self.encoder(g.x, g.edge_index)
        dev = z.device
        out = [self.decoder(z[torch.as_tensor(u[s:s + batch], device=dev)],
                            z[torch.as_tensor(v[s:s + batch], device=dev)]).float().cpu()
               for s in range(0, len(u), batch)]
        return torch.cat(out).numpy().astype(np.float64)


# ---------------------------------------------------------------- training

@dataclass
class TrainResult:
    model: LinkPredictor
    best_epoch: int
    best_val: float
    curve: list[dict]
    seconds: float


def train(model: LinkPredictor, g_train: GraphData, u: np.ndarray, v: np.ndarray, y: np.ndarray,
          g_val: GraphData, uv_val: tuple[np.ndarray, np.ndarray], y_val: np.ndarray, *, metric: str,
          lr: float, weight_decay: float, epochs: int, patience: int, device: str, log=None) -> TrainResult:
    """Full-batch training on g_train; early stopping on a validation metric ('roc_auc' or 'average_precision')."""
    t0 = time.time()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ut, vt = torch.as_tensor(u, device=device), torch.as_tensor(v, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    fn = roc_auc if metric == "roc_auc" else average_precision
    best, best_state, best_epoch, bad, curve = -np.inf, None, -1, 0, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(g_train, ut, vt), yt)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {ep}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        s = model.score(g_val, *uv_val)
        val = fn(s[y_val == 1], s[y_val == 0])
        curve.append({"epoch": ep, "train_loss": loss.item(), f"val_{metric}": val, "seconds": time.time() - t0})
        if val > best + 1e-5:
            best, best_epoch, bad = val, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if log and ep % 25 == 0:
            log(f"  epoch {ep}: loss {loss.item():.4f} val {metric} {val:.4f}")
    model.load_state_dict(best_state)
    return TrainResult(model, best_epoch, float(best), curve, time.time() - t0)


def clone(model: LinkPredictor) -> LinkPredictor:
    return copy.deepcopy(model)
