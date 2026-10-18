"""Split-conformal calibration (marginal and label-conditional)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prediction_sets import prediction_sets
from .scores import lac_true_label_scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    Returns the k-th smallest score with k = ceil((n+1)(1-alpha)); returns +inf when
    k > n (too few calibration points for the requested alpha), so that every
    label is included. Under exchangeability of calibration and test points,
    P(s_test <= q) >= 1 - alpha.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        raise ValueError("no calibration scores")
    if np.isnan(s).any():
        raise ValueError("NaN calibration scores")
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


@dataclass
class SplitConformalClassifier:
    """Binary LAC split conformal predictor.

    mode="marginal":          one threshold from all calibration scores.
    mode="class_conditional": one threshold per true label (Mondrian on Y), which
                              targets P(Y in C(X) | Y=y) >= 1-alpha for each y.
    Calibration data must be disjoint from the data used to fit the classifier.
    """
    alpha: float
    mode: str = "marginal"
    qhat: dict = field(default_factory=dict)       # label -> threshold
    n_cal: dict = field(default_factory=dict)      # label -> #calibration points used

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        y = np.asarray(y_cal).astype(int)
        s = lac_true_label_scores(proba_cal, y)
        if self.mode == "marginal":
            q = conformal_quantile(s, self.alpha)
            self.qhat = {0: q, 1: q}
            self.n_cal = {0: len(s), 1: len(s)}
        elif self.mode == "class_conditional":
            self.qhat, self.n_cal = {}, {}
            for lab in (0, 1):
                sl = s[y == lab]
                if sl.size == 0:
                    raise ValueError(f"no calibration points with label {lab}")
                self.qhat[lab] = conformal_quantile(sl, self.alpha)
                self.n_cal[lab] = int(sl.size)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        return self

    def predict_sets(self, proba: np.ndarray) -> np.ndarray:
        if not self.qhat:
            raise RuntimeError("call fit() first")
        return prediction_sets(proba, self.qhat[0], self.qhat[1])

    def describe(self) -> dict:
        return {"alpha": self.alpha, "mode": self.mode,
                "qhat": {int(k): float(v) for k, v in self.qhat.items()},
                "n_cal": {int(k): int(v) for k, v in self.n_cal.items()},
                "qhat_infinite": any(np.isinf(v) for v in self.qhat.values())}
