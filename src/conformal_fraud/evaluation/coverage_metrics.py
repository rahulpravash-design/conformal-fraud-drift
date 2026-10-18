from __future__ import annotations

import numpy as np

from ..statistics.confidence_intervals import wilson_interval


def coverage_summary(sets: np.ndarray, y: np.ndarray, conf: float = 0.95) -> dict:
    """Empirical coverage and set-size statistics.

    covered_i = [y_i in C(x_i)]. coverage = mean(covered). Class-conditional coverage is
    the same restricted to y=1 (fraud) or y=0.
    """
    y = np.asarray(y).astype(int)
    sets = np.asarray(sets, bool)
    if sets.shape != (len(y), 2):
        raise ValueError("sets must have shape (n, 2)")
    n = len(y)
    covered = sets[np.arange(n), y]
    size = sets.sum(axis=1)
    out = {"n": n, "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}

    def prop(prefix, mask):
        m = int(mask.sum())
        k = int(covered[mask].sum()) if m else 0
        lo, hi = wilson_interval(k, m, conf)
        out[f"{prefix}_k"] = k
        out[prefix] = k / m if m else float("nan")
        out[f"{prefix}_lo"], out[f"{prefix}_hi"] = lo, hi

    prop("coverage", np.ones(n, bool))
    prop("coverage_pos", y == 1)
    prop("coverage_neg", y == 0)
    out["mean_set_size"] = float(size.mean()) if n else float("nan")
    out["frac_empty"] = float((size == 0).mean()) if n else float("nan")
    out["frac_singleton"] = float((size == 1).mean()) if n else float("nan")
    out["frac_both"] = float((size == 2).mean()) if n else float("nan")
    return out
