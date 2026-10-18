"""LAC-style nonconformity scores for binary classification.

For a classifier with fraud probability p = P(Y=1|x), the score of label y is
    s(x, y) = 1 - P(Y=y | x),
i.e. s(x, 1) = 1 - p and s(x, 0) = p.
Column index of the returned matrix equals the label value.
"""
from __future__ import annotations

import numpy as np


def _check_proba(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("proba must be 1-D (probability of class 1)")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1] and contain no NaN")
    return p


def lac_score_matrix(proba_fraud: np.ndarray) -> np.ndarray:
    p = _check_proba(proba_fraud)
    return np.column_stack([p, 1.0 - p])


def lac_true_label_scores(proba_fraud: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must be binary 0/1")
    S = lac_score_matrix(proba_fraud)
    return S[np.arange(len(y)), y]
