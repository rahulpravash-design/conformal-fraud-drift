from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_val: np.ndarray, score_val: np.ndarray, fpr: float) -> float:
    """Threshold t such that predicting positive iff score > t gives FPR <= fpr on the
    VALIDATION negatives. Computed from validation data only (never test labels)."""
    y = np.asarray(y_val).astype(int)
    neg = np.asarray(score_val, float)[y == 0]
    if neg.size == 0:
        raise ValueError("no negatives in validation set")
    return float(np.quantile(neg, 1.0 - fpr, method="higher"))


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    if y.min() == y.max():
        raise ValueError("only one class present; metrics undefined")
    pred = score > threshold
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (prec + rec) > 0 else float("nan")
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),  # sklearn AP (step-wise, not trapezoidal PR-AUC)
        "threshold": float(threshold),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
