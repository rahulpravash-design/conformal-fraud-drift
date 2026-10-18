"""Descriptive dataset audit. Purely descriptive: no design decision may be conditioned on test-month labels
(see docs/protocol.md section 'Audit vs confirmatory')."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..evaluation.group_metrics import assign_group
from ..statistics.confidence_intervals import wilson_interval
from .load_data import ROW_ID, feature_columns
from .temporal_split import SplitSpec


def audit_variant(df: pd.DataFrame, dcfg: dict, ecfg: dict) -> dict:
    label, tcol, cats = dcfg["label"], dcfg["time_column"], dcfg["categorical_features"]
    feats = feature_columns(df, label, tcol, dcfg.get("exclude_features", []))
    num = [c for c in feats if c not in cats]
    out: dict = {}

    out["overview"] = {
        "n_rows": int(len(df)), "n_columns_excluding_row_id": int(df.shape[1] - 1),
        "columns": [c for c in df.columns if c != ROW_ID],
        "n_fraud": int(df[label].sum()), "fraud_prevalence": float(df[label].mean()),
        "months": sorted(int(m) for m in df[tcol].unique()),
        "full_row_duplicates": int(df.drop(columns=[ROW_ID]).duplicated().sum()),
        "feature_vector_duplicates": int(df[feats].duplicated().sum()),
        "constant_columns": [c for c in feats if df[c].nunique(dropna=False) <= 1],
    }
    dtypes = df.drop(columns=[ROW_ID]).dtypes.astype(str)
    miss = pd.DataFrame({"dtype": dtypes, "n_nan": df.drop(columns=[ROW_ID]).isna().sum()})
    # BAF documents -1 (and negative for intended_balcon_amount) as missing-value sentinels.
    miss["n_minus1"] = [int((df[c] == -1).sum()) if c in df and pd.api.types.is_numeric_dtype(df[c]) else 0 for c in miss.index]
    miss["n_negative"] = [int((df[c] < 0).sum()) if c in df and pd.api.types.is_numeric_dtype(df[c]) else 0 for c in miss.index]
    out["missingness"] = miss.reset_index(names="column")

    g = df.groupby(tcol)[label].agg(n="size", n_fraud="sum").reset_index()
    g["prevalence"] = g.n_fraud / g.n
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(g.n_fraud, g.n)]
    g["prev_lo"], g["prev_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    out["monthly"] = g

    rows = []
    for key, spec in dcfg["groups"].items():
        grp = assign_group(df, spec)
        tmp = pd.DataFrame({tcol: df[tcol], label: df[label], "group": grp})
        for (m, gg), s in tmp.dropna(subset=["group"]).groupby([tcol, "group"]):
            rows.append({"group_def": spec["name"], "month": int(m), "group": gg, "n": len(s),
                         "n_fraud": int(s[label].sum()), "n_nonfraud": int((1 - s[label]).sum()),
                         "prevalence": float(s[label].mean()),
                         "share_of_month": len(s) / int((df[tcol] == m).sum())})
    out["group_by_month"] = pd.DataFrame(rows)
    if "customer_age" in df.columns:
        out["customer_age_values"] = {int(k): int(v) for k, v in df["customer_age"].value_counts().sort_index().items()}

    # Shift descriptors relative to the pooled training months. These describe covariate (P(X)) and
    # label (P(Y)) shift only; they do NOT establish concept drift P(Y|X).
    spec = SplitSpec.from_config(ecfg)
    ref = df[df[tcol].isin(spec.train)]
    drift = []
    for m in sorted(int(x) for x in df[tcol].unique()):
        if m in spec.train:
            continue
        cur = df[df[tcol] == m]
        for c in num:
            if ref[c].nunique() <= 1:
                continue
            drift.append({"month": m, "feature": c, "kind": "numeric_KS", "value": float(ks_2samp(ref[c], cur[c]).statistic)})
        for c in cats:
            p = ref[c].astype(str).value_counts(normalize=True); q = cur[c].astype(str).value_counts(normalize=True)
            idx = p.index.union(q.index)
            drift.append({"month": m, "feature": c, "kind": "categorical_TV",
                          "value": float(0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())})
    out["drift_vs_train"] = pd.DataFrame(drift)

    # Feasibility of the proposed split for conformal calibration.
    part = []
    for name, ms in spec.roles().items():
        s = df[df[tcol].isin(ms)]
        part.append({"partition": name, "months": list(ms), "n": int(len(s)), "n_fraud": int(s[label].sum()),
                     "prevalence": float(s[label].mean()) if len(s) else float("nan")})
    out["split_feasibility"] = pd.DataFrame(part)
    return out
