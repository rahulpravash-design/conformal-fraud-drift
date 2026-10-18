from __future__ import annotations

import operator

import numpy as np
import pandas as pd

from ..statistics.confidence_intervals import newcombe_diff_interval
from .coverage_metrics import coverage_summary
from .review_metrics import review_summary

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


def assign_group(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Binary group label: 'high' where column <op> threshold, else 'low'. Missing -> NaN (excluded and counted)."""
    col = df[spec["column"]]
    g = pd.Series(np.where(_OPS[spec["op"]](col, spec["threshold"]), "high", "low"), index=df.index, dtype=object)
    g[col.isna()] = np.nan
    return g


def group_table(sets: np.ndarray, y: np.ndarray, group: pd.Series, months: pd.Series,
                min_pos: int, conf: float = 0.95) -> pd.DataFrame:
    """Per (month, group) coverage/review table. `month` includes an 'all' pooled row.

    `interpretable_pos` flags n_pos >= min_pos (screening threshold for class-conditional
    inference on fraud), `interpretable_n` flags n_neg>=min_pos. These are warnings, not guarantees.
    """
    rows = []
    sets = np.asarray(sets, bool); y = np.asarray(y).astype(int)
    grp = group.to_numpy(object); mon = months.to_numpy()
    for m in [*sorted(np.unique(mon)), "all"]:
        mm = np.ones(len(y), bool) if m == "all" else mon == m
        n_missing = int((mm & pd.isna(grp)).sum())
        for g in ("low", "high"):
            mask = mm & (grp == g)
            if not mask.any():
                continue
            r = {"month": m, "group": g, "n_missing_group_excluded": n_missing}
            r.update(coverage_summary(sets[mask], y[mask], conf))
            rv = review_summary(sets[mask], y[mask], conf)
            r.update({k: v for k, v in rv.items() if k != "n"})
            r["interpretable_pos"] = bool(r["n_pos"] >= min_pos)
            r["interpretable_neg"] = bool(r["n_neg"] >= min_pos)
            rows.append(r)
    return pd.DataFrame(rows)


def group_gaps(table: pd.DataFrame, conf: float = 0.95) -> pd.DataFrame:
    """high-minus-low coverage gaps with Newcombe CIs, for marginal and fraud-class coverage."""
    rows = []
    for m, sub in table.groupby("month", sort=False):
        lo, hi = sub[sub.group == "low"], sub[sub.group == "high"]
        if lo.empty or hi.empty:
            continue
        lo, hi = lo.iloc[0], hi.iloc[0]
        r = {"month": m}
        for name, kcol, ncol in (("coverage", "coverage_k", "n"), ("coverage_pos", "coverage_pos_k", "n_pos"),
                                 ("coverage_neg", "coverage_neg_k", "n_neg")):
            d, a, b = newcombe_diff_interval(int(hi[kcol]), int(hi[ncol]), int(lo[kcol]), int(lo[ncol]), conf)
            r[f"gap_{name}"], r[f"gap_{name}_lo"], r[f"gap_{name}_hi"] = d, a, b
        r["gap_interpretable"] = bool(hi.interpretable_pos and lo.interpretable_pos)
        rows.append(r)
    return pd.DataFrame(rows)
