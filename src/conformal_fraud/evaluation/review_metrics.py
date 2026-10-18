"""Hypothetical review policy (experimental, NOT a validated bank policy).

    set {0}      -> approve
    set {1}      -> escalate (fraud action / investigation)
    set {0,1}    -> human review (ambiguous)
    set {}       -> human review (out-of-policy / unsupported)
"""
from __future__ import annotations

import numpy as np

from ..statistics.confidence_intervals import wilson_interval
from ..conformal.prediction_sets import set_type


def review_summary(sets: np.ndarray, y: np.ndarray, conf: float = 0.95) -> dict:
    y = np.asarray(y).astype(int)
    t = set_type(np.asarray(sets, bool))
    n = len(y)
    approve, escalate = t == "only0", t == "only1"
    review = (t == "both") | (t == "empty")
    pos = y == 1

    def rate(prefix, mask):
        k = int(mask.sum())
        lo, hi = wilson_interval(k, n, conf)
        return {prefix: k / n if n else float("nan"), f"{prefix}_lo": lo, f"{prefix}_hi": hi}

    out = {"n": n}
    out.update(rate("review_rate", review))
    out.update(rate("escalate_rate", escalate))
    out.update(rate("approve_rate", approve))
    out["fraud_total"] = int(pos.sum())
    out["fraud_approved"] = int((approve & pos).sum())     # fraud that the policy waves through
    out["fraud_reviewed"] = int((review & pos).sum())
    out["fraud_escalated"] = int((escalate & pos).sum())
    nf = max(int(pos.sum()), 1)
    out["fraud_approved_rate"] = out["fraud_approved"] / nf if pos.any() else float("nan")
    out["fraud_flagged_recall"] = (out["fraud_reviewed"] + out["fraud_escalated"]) / nf if pos.any() else float("nan")
    out["fraud_share_of_reviewed"] = (
        out["fraud_reviewed"] / int(review.sum()) if review.any() else float("nan"))  # fraud share among reviewed items
    return out
