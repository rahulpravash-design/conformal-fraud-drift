"""Two-stage experiment: (1) train + select on train/val, (2) conformal calibration on cal, evaluation on test.

Stage 1 never touches cal/test labels for fitting or selection. Stage 2 uses ONLY the calibration
partition to compute conformal thresholds. Test labels are used solely to compute reported metrics.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .conformal.calibration import SplitConformalClassifier
from .conformal.prediction_sets import set_type
from .data.load_data import ROW_ID, feature_columns, validate_schema
from .data.temporal_split import SplitSpec, assert_no_leakage, leakage_checks, make_split
from .evaluation.classification_metrics import classification_metrics, threshold_at_fpr
from .evaluation.coverage_metrics import coverage_summary
from .evaluation.group_metrics import assign_group, group_gaps, group_table
from .evaluation.review_metrics import review_summary
from .models.lightgbm_model import fit_lightgbm
from .models.logistic_regression import fit_logreg
from .models.preprocessing import FraudPreprocessor
from .utils.provenance import environment_info, git_commit, sha256_file, utc_now, write_json


def run_id(variant: str, model: str, seed: int) -> str:
    return f"{variant}_{model}_seed{seed}"


def _proba(model_name: str, model, X: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    if model_name == "logreg":
        X = X.copy()
        for c in categorical:
            if c in X.columns:
                X[c] = X[c].astype(object)
    return model.predict_proba(X)[:, 1]


def train_stage(df: pd.DataFrame, cfgs: dict, variant: str, model_name: str, out_root: str | Path,
                data_path: str | Path | None = None, methods_override: dict | None = None) -> Path:
    t_start = time.time()
    dcfg, ecfg, mcfg = cfgs["data"], cfgs["experiment"], cfgs["methods"]
    label, tcol, cats = dcfg["label"], dcfg["time_column"], dcfg["categorical_features"]
    group_cols = sorted({g["column"] for g in dcfg["groups"].values()})
    validate_schema(df, label, tcol, cats, group_cols)
    feats = feature_columns(df, label, tcol, dcfg.get("exclude_features", []))
    spec = SplitSpec.from_config(ecfg)
    parts = make_split(df, spec, tcol)
    checks = leakage_checks(parts, spec, feats, label, tcol)
    assert_no_leakage(checks)

    seed = int(ecfg["seed"])
    rid = run_id(variant, model_name, seed)
    run_dir = Path(out_root) / "metrics" / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    pre = FraudPreprocessor(feats, cats).fit(parts["train"])
    X = {k: pre.transform(v) for k, v in parts.items()}
    y = {k: v[label].astype(int).to_numpy() for k, v in parts.items()}

    mm = methods_override or mcfg
    if model_name == "lightgbm":
        model, sel, best = fit_lightgbm(X["train"], y["train"], X["val"], y["val"], mm["lightgbm"], seed, ecfg.get("n_jobs", 4))
    elif model_name == "logreg":
        model, sel, best = fit_logreg(X["train"], y["train"], X["val"], y["val"], mm["logreg"], cats, seed)
    else:
        raise ValueError(model_name)

    proba = {k: _proba(model_name, model, X[k], cats) for k in X}
    thr = threshold_at_fpr(y["val"], proba["val"], ecfg["fpr_operating_point"])   # validation-only threshold

    rows = []
    for part in ("train", "val", "cal", "test"):
        rows.append({"partition": part, "month": "all", **classification_metrics(y[part], proba[part], thr)})
    for m in sorted(parts["test"][tcol].unique()):
        mask = (parts["test"][tcol] == m).to_numpy()
        rows.append({"partition": "test", "month": int(m), **classification_metrics(y["test"][mask], proba["test"][mask], thr)})
    pd.DataFrame(rows).to_csv(run_dir / "baseline_metrics.csv", index=False)
    sel.to_csv(run_dir / "model_selection.csv", index=False)

    keep = [ROW_ID, tcol, *group_cols]
    pred = pd.concat([parts[k][keep].assign(partition=k, y=y[k], proba=proba[k]) for k in ("val", "cal", "test")], ignore_index=True)
    pred = pred.rename(columns={tcol: "month"})
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    joblib.dump({"model": model, "preprocessor": pre}, run_dir / "model.joblib")

    meta = {
        "run_id": rid, "stage": "train", "variant": variant, "model": model_name, "timestamp_utc": utc_now(),
        "seed": seed, "split": spec.as_dict(), "features": feats, "n_features": len(feats),
        "selected_config": best, "operating_threshold_from_validation": thr,
        "fpr_operating_point": ecfg["fpr_operating_point"],
        "partition_sizes": {k: int(len(v)) for k, v in parts.items()},
        "partition_fraud": {k: int(y[k].sum()) for k, v in parts.items()},
        "leakage_checks": checks, "runtime_seconds": time.time() - t_start,
        "data_file": str(data_path) if data_path else None,
        "data_sha256": sha256_file(data_path) if data_path and Path(data_path).exists() else None,
        "git": git_commit(), "environment": environment_info(), "configs": cfgs,
    }
    write_json(meta, run_dir / "run_metadata_train.json")
    write_json(meta, Path(out_root) / "run_metadata" / f"{rid}_train.json")
    return run_dir


def conformal_stage(run_dir: str | Path, cfgs: dict) -> dict:
    run_dir = Path(run_dir)
    ecfg, dcfg, mcfg = cfgs["experiment"], cfgs["data"], cfgs["methods"]
    conf = ecfg["confidence_level"]
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    cal = pred[pred.partition == "cal"]
    test = pred[pred.partition == "test"].reset_index(drop=True)
    assert set(cal.row_id).isdisjoint(set(test.row_id)), "calibration/test overlap"
    assert cal.month.max() < test.month.min(), "calibration must precede test"

    groups = {k: assign_group(test, g) for k, g in dcfg["groups"].items()}
    gnames = {k: g["name"] for k, g in dcfg["groups"].items()}
    ytest, ptest = test.y.to_numpy(), test.proba.to_numpy()

    summ, gtabs, gaps, calib, set_cols = [], [], [], [], {}
    for method in mcfg["conformal"]["methods"]:
        for alpha in ecfg["alphas"]:
            cp = SplitConformalClassifier(alpha=alpha, mode=method).fit(cal.proba.to_numpy(), cal.y.to_numpy())
            sets = cp.predict_sets(ptest)
            d = cp.describe(); d["n_cal_total"] = int(len(cal)); d["n_cal_fraud"] = int(cal.y.sum())
            calib.append(d)
            set_cols[f"set_{method}_a{alpha}"] = set_type(sets)
            for m in [*sorted(test.month.unique()), "all"]:
                mask = np.ones(len(test), bool) if m == "all" else (test.month == m).to_numpy()
                row = {"method": method, "alpha": alpha, "nominal": 1 - alpha, "month": m}
                row.update(coverage_summary(sets[mask], ytest[mask], conf))
                row.update({k: v for k, v in review_summary(sets[mask], ytest[mask], conf).items() if k != "n"})
                row["coverage_deviation"] = row["coverage"] - (1 - alpha)
                summ.append(row)
            for gk, g in groups.items():
                t = group_table(sets, ytest, g, test.month, ecfg["min_positives_for_group_inference"], conf)
                t.insert(0, "group_def", gnames[gk]); t.insert(0, "alpha", alpha); t.insert(0, "method", method)
                gtabs.append(t)
                gp = group_gaps(t, conf)
                gp.insert(0, "group_def", gnames[gk]); gp.insert(0, "alpha", alpha); gp.insert(0, "method", method)
                gaps.append(gp)

    pd.DataFrame(summ).to_csv(run_dir / "conformal_summary.csv", index=False)
    pd.concat(gtabs, ignore_index=True).to_csv(run_dir / "group_metrics.csv", index=False)
    pd.concat(gaps, ignore_index=True).to_csv(run_dir / "group_gaps.csv", index=False)
    write_json(calib, run_dir / "calibration.json")
    out = test[[ROW_ID, "month", "y", "proba"]].copy()
    for k, v in set_cols.items():
        out[k] = v
    out.to_parquet(run_dir / "test_prediction_sets.parquet", index=False)
    meta = {"stage": "conformal", "timestamp_utc": utc_now(), "git": git_commit(), "calibration": calib,
            "alphas": ecfg["alphas"], "methods": mcfg["conformal"]["methods"], "score": mcfg["conformal"]["score"]}
    write_json(meta, run_dir / "run_metadata_conformal.json")
    return meta
