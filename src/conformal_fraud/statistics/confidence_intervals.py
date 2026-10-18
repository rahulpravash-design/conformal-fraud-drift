"""Interval estimates for proportions and bootstrap metrics.

All intervals treat records as independent. BAF records are synthetic and described as
independent, but ordering/sampling structure means these are descriptive, not a proof
of distribution-free validity under drift.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff_interval(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Newcombe (1998) hybrid-score interval for p1 - p2 (independent samples).

    Returns (diff, lower, upper).
    """
    if n1 <= 0 or n2 <= 0:
        return (float("nan"),) * 3
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, conf)
    l2, u2 = wilson_interval(k2, n2, conf)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (float(d), float(lo), float(hi))


def bootstrap_ci(metric_fn, *arrays, n_boot: int = 200, conf: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of metric_fn(*arrays) resampling rows jointly.

    Returns (point, lower, upper). Resamples that make metric_fn raise ValueError
    (e.g. no positives) are skipped; the number skipped is not hidden: if fewer than
    90% succeed, a RuntimeError is raised.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(metric_fn(*[a[idx] for a in arrays])))
        except ValueError:
            continue
    if len(vals) < 0.9 * n_boot:
        raise RuntimeError(f"bootstrap: only {len(vals)}/{n_boot} resamples valid")
    a = (1 - conf) / 2
    return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
