"""Logistic regression and LightGBM on pair features.

Training data come only from the TRAIN split (graph < 2023, labels 2023). Negatives are
prefixes of the shuffled train pools, so a mix is a uniform sample of each type:
  n_neg = ratio * n_pos,  n_hard = round(hard_fraction * n_neg),  n_random = n_neg - n_hard.
The distance-2 specialist uses only distance-2 positives and hard negatives.

LightGBM early-stops on VALIDATION average precision under the `hard` protocol
(distance-2 positives vs hard negatives); test is never seen during fitting or selection.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from linkpred.features.build import features_for
from linkpred.features.pair_features import LOG1P


def training_pairs(train: dict, hard_fraction: float, ratio: int, specialist: bool = False,
                   active: bool = False, d2_only: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Training pairs (u_idx, v_idx) and labels for a negative mix (shared by all pair classifiers).

    active=True: pair-focused training -- all negatives are activity-matched hard negatives from the
    2023 window (train split), i.e. both authors published in 2023. That is a training LABEL choice
    on the train window, which precedes the validation cutoff; model inputs are unchanged.
    """
    pos = train["positives"]
    if specialist or d2_only:
        # Distance-2 positives only. Required whenever all negatives are at distance 2 (hard /
        # activity-matched): otherwise distance >= 3 alone identifies a positive (audit M1, Phase 4 rule).
        pos = pos[pos["distance"] == 2]
    if specialist:
        hard_fraction = 1.0
    n_neg = ratio * len(pos)
    if active:
        parts = [("train_pool_hard_active", n_neg)]
    else:
        n_hard = int(round(hard_fraction * n_neg))
        parts = [("train_pool_hard", n_hard), ("train_pool_random", n_neg - n_hard)]
    for table, k in parts:
        if k > len(train[table]):
            raise ValueError(f"training pool {table} too small for this mix")
    pairs = pd.concat([pos[["u_idx", "v_idx"]]] + [train[t].iloc[:k][["u_idx", "v_idx"]] for t, k in parts],
                      ignore_index=True)
    return pairs, np.r_[np.ones(len(pos)), np.zeros(n_neg)]


def training_set(train: dict, feat: pd.DataFrame, columns: list[str], hard_fraction: float, ratio: int,
                 specialist: bool = False, active: bool = False, d2_only: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    pairs, y = training_pairs(train, hard_fraction, ratio, specialist, active, d2_only)
    return features_for(feat, pairs["u_idx"].values, pairs["v_idx"].values, columns), y


def eval_features(split: dict, feat: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Features for every evaluation pair set of a split, aligned with the saved tables."""
    sets = {"positives": ("positives", "u_idx", "v_idx"), "neg_random": ("neg_random", "u_idx", "v_idx"),
            "neg_hard": ("neg_hard", "u_idx", "v_idx"), "ranking": ("ranking", "head_idx", "cand_idx"),
            "ranking_hard": ("ranking_hard", "head_idx", "cand_idx"),
            "neg_hard_active": ("neg_hard_active", "u_idx", "v_idx"),
            "ranking_hard_active": ("ranking_hard_active", "head_idx", "cand_idx")}
    return {k: features_for(feat, split[t][a].values, split[t][b].values, columns) for k, (t, a, b) in sets.items()}


def hard_eval_set(split: dict, X: dict[str, pd.DataFrame], negatives: str = "neg_hard"):
    """Distance-2 positives vs `negatives` (neg_hard for `hard`, neg_hard_active for `hard_active`)."""
    d2 = (split["positives"]["distance"] == 2).values
    Xe = pd.concat([X["positives"][d2], X[negatives]], ignore_index=True)
    return Xe, np.r_[np.ones(d2.sum()), np.zeros(len(X[negatives]))]


def make_logreg(columns: list[str], C: float = 1.0):
    log_cols = [i for i, c in enumerate(columns) if c in LOG1P]

    def transform(X):
        X = np.asarray(X, dtype=np.float64).copy()
        X[:, log_cols] = np.log1p(np.maximum(X[:, log_cols], 0))
        return X
    return make_pipeline(FunctionTransformer(transform), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


def fit_logreg(X, y, columns, seed: int, C: float = 1.0):  # C from config models.logreg.C
    # lbfgs is deterministic: the seed is recorded but cannot change the fit.
    m = make_logreg(columns, C)
    m.fit(X.values, y)
    return m


def fit_lightgbm(X, y, Xval, yval, params: dict, seed: int, max_rounds: int = 2000, patience: int | None = None):
    m = lgb.LGBMClassifier(
        n_estimators=max_rounds, learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"], subsample=params["subsample"], subsample_freq=1,
        colsample_bytree=params["colsample_bytree"], reg_lambda=params.get("reg_lambda", 0.0),
        random_state=seed, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
    patience = patience or params.get("early_stopping_rounds", 50)   # config models.lightgbm.early_stopping_rounds
    m.fit(X, y, eval_set=[(Xval, yval)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    return m


def predict(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, lgb.LGBMClassifier):
        return model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1]
    return model.predict_proba(X.values)[:, 1]


def score_sets(model, X: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {k: predict(model, x) for k, x in X.items()}
