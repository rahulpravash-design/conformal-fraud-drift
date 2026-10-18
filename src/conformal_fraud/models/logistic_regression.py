from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _pipeline(num_cols, cat_cols, C, class_weight, max_iter, seed):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                                                               random_state=seed))])


def fit_logreg(Xtr, ytr, Xva, yva, methods_cfg: dict, categorical: list[str], seed: int):
    """Grid over C and class_weight; select on validation average precision. Scaler/encoder fit inside
    the pipeline on training rows only. Missing sentinels (-1) are left as-is (documented simplification)."""
    cat_cols = [c for c in categorical if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    Xtr = Xtr.copy(); Xva = Xva.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype(object); Xva[c] = Xva[c].astype(object)
    cfg = methods_cfg
    rows, best = [], None
    for C in cfg["C_grid"]:
        for cw in cfg["class_weight_grid"]:
            t0 = time.time()
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                p = _pipeline(num_cols, cat_cols, C, cw, cfg["max_iter"], seed).fit(Xtr, ytr)
            ap = float(average_precision_score(yva, p.predict_proba(Xva)[:, 1]))
            rows.append({"C": C, "class_weight": cw, "val_average_precision": ap,
                         "n_iter": int(np.max(p.named_steps["clf"].n_iter_)), "fit_seconds": time.time() - t0,
                         "warnings": "; ".join(sorted({str(w.message) for w in wlist}))[:300]})
            if best is None or ap > best[0]:
                best = (ap, p, {"C": C, "class_weight": cw})
    return best[1], pd.DataFrame(rows), best[2]
