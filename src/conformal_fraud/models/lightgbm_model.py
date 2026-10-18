from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score


def fit_lightgbm(Xtr, ytr, Xva, yva, methods_cfg: dict, seed: int, n_jobs: int = 4):
    """Fit every prespecified config on TRAIN, early-stop and select on VALIDATION average
    precision. Returns (best_model, results_table, best_config). Test/cal data never enter."""
    fixed = dict(methods_cfg["fixed"])
    rows, best = [], None
    for i, cfg in enumerate(methods_cfg["grid"]):
        params = {**fixed, **cfg, "random_state": seed, "n_jobs": n_jobs}
        t0 = time.time()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            m = LGBMClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[early_stopping(methods_cfg["early_stopping_rounds"], first_metric_only=True, verbose=False),
                             log_evaluation(0)])
        pva = m.predict_proba(Xva)[:, 1]
        ap = float(average_precision_score(yva, pva))
        rows.append({"config_id": i, **cfg, "best_iteration": int(m.best_iteration_ or params["n_estimators"]),
                     "val_average_precision": ap, "fit_seconds": time.time() - t0,
                     "warnings": "; ".join(sorted({str(w.message) for w in wlist}))[:300]})
        if best is None or ap > best[0]:
            best = (ap, m, {**cfg, "config_id": i})
    return best[1], pd.DataFrame(rows), best[2]
