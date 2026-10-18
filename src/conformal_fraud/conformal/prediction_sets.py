from __future__ import annotations

import numpy as np

from .scores import lac_score_matrix


def prediction_sets(proba_fraud: np.ndarray, qhat0: float, qhat1: float) -> np.ndarray:
    """Boolean (n, 2) membership matrix; column y is True iff label y is in the set.

    Label y is included iff s(x, y) <= qhat_y (non-strict inequality, deterministic,
    no randomisation; ties are therefore resolved conservatively).
    """
    S = lac_score_matrix(proba_fraud)
    return np.column_stack([S[:, 0] <= qhat0, S[:, 1] <= qhat1])


def set_type(sets: np.ndarray) -> np.ndarray:
    """'empty' | 'only0' | 'only1' | 'both' per row."""
    in0, in1 = sets[:, 0], sets[:, 1]
    out = np.full(len(sets), "empty", dtype=object)
    out[in0 & ~in1] = "only0"
    out[~in0 & in1] = "only1"
    out[in0 & in1] = "both"
    return out
