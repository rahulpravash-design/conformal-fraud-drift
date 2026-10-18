"""Mechanics-only smoke test on SYNTHETIC data. Not a research result."""
import json

import numpy as np
import pandas as pd


def test_end_to_end_synthetic(synth_df, cfgs, tiny_methods, tmp_path):
    from conformal_fraud.pipeline import conformal_stage, train_stage
    cfgs["experiment"]["alphas"] = [0.2, 0.1, 0.05]
    rd = train_stage(synth_df, cfgs, "SYN", "lightgbm", tmp_path, methods_override=tiny_methods)
    conformal_stage(rd, cfgs)
    for f in ["predictions.parquet", "baseline_metrics.csv", "model_selection.csv", "run_metadata_train.json",
              "conformal_summary.csv", "group_metrics.csv", "group_gaps.csv", "calibration.json", "test_prediction_sets.parquet"]:
        assert (rd / f).exists(), f
    meta = json.load(open(rd / "run_metadata_train.json"))
    assert all(c["passed"] for c in meta["leakage_checks"])
    assert meta["partition_sizes"]["test"] == 12000 and "git" in meta and "environment" in meta
    pred = pd.read_parquet(rd / "predictions.parquet")
    assert set(pred[pred.partition == "test"].month) == {6, 7} and set(pred[pred.partition == "cal"].month) == {5}
    s = pd.read_csv(rd / "conformal_summary.csv")
    assert s.coverage.between(0, 1).all() and s.mean_set_size.between(0, 2).all()
    # class-conditional method should cover fraud near nominal on this (mild-drift) synthetic data; marginal LAC on a rare class typically does not
    allrows = s[(s.month == "all") & (s.alpha == 0.1)].set_index("method")
    assert allrows.loc["class_conditional", "coverage_pos"] > allrows.loc["marginal", "coverage_pos"] - 1e-9
    base = pd.read_csv(rd / "baseline_metrics.csv")
    assert set(base.partition) == {"train", "val", "cal", "test"} and base.roc_auc.between(.5, 1).all()


def test_logreg_runs(synth_df, cfgs, tmp_path):
    from conformal_fraud.pipeline import train_stage
    ml = {"logreg": {"C_grid": [1.0], "class_weight_grid": [None, "balanced"], "max_iter": 500}}
    rd = train_stage(synth_df, cfgs, "SYN", "logreg", tmp_path, methods_override=ml)
    assert (rd / "baseline_metrics.csv").exists()
