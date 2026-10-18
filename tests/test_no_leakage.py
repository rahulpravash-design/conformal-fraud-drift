import numpy as np
import pandas as pd
import pytest

from conformal_fraud.data.load_data import feature_columns
from conformal_fraud.data.temporal_split import SplitSpec, assert_no_leakage, leakage_checks, make_split
from conformal_fraud.models.preprocessing import FraudPreprocessor
from conformal_fraud.pipeline import conformal_stage, train_stage

SPEC = SplitSpec((0, 1, 2, 3), (4,), (5,), (6, 7))
CATS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


def _parts(df):
    feats = feature_columns(df, "fraud_bool", "month", [])
    return make_split(df, SPEC, "month"), feats


def test_clean_split_passes(synth_df):
    parts, feats = _parts(synth_df)
    checks = leakage_checks(parts, SPEC, feats, "fraud_bool", "month")
    assert_no_leakage(checks)
    assert len(checks) >= 8


def test_detects_row_overlap(synth_df):
    parts, feats = _parts(synth_df)
    parts["test"] = pd.concat([parts["test"], parts["cal"].head(5)])
    with pytest.raises(AssertionError, match="row_ids_disjoint|months_match"):
        assert_no_leakage(leakage_checks(parts, SPEC, feats, "fraud_bool", "month"))


def test_detects_label_or_time_in_features(synth_df):
    parts, feats = _parts(synth_df)
    with pytest.raises(AssertionError, match="label_not_in_features"):
        assert_no_leakage(leakage_checks(parts, SPEC, feats + ["fraud_bool"], "fraud_bool", "month"))
    with pytest.raises(AssertionError, match="time_column_not_in_features"):
        assert_no_leakage(leakage_checks(parts, SPEC, feats + ["month"], "fraud_bool", "month"))


def test_detects_future_rows_in_train(synth_df):
    parts, feats = _parts(synth_df)
    parts["train"] = pd.concat([parts["train"], synth_df[synth_df.month == 7].head(10)])
    with pytest.raises(AssertionError):
        assert_no_leakage(leakage_checks(parts, SPEC, feats, "fraud_bool", "month"))


def test_detects_cross_partition_duplicates(synth_df):
    parts, feats = _parts(synth_df)
    dup = parts["train"].head(3).copy(); dup["row_id"] = -np.arange(1, 4); dup["month"] = 6
    parts["test"] = pd.concat([parts["test"], dup])
    with pytest.raises(AssertionError, match="duplicate"):
        assert_no_leakage(leakage_checks(parts, SPEC, feats, "fraud_bool", "month"))


def test_preprocessor_fit_on_train_only(synth_df):
    parts, feats = _parts(synth_df)
    pre = FraudPreprocessor(feats, CATS).fit(parts["train"])
    before = {k: list(v) for k, v in pre.categories_.items()}
    # an unseen category in later data must not extend the vocabulary; it becomes NaN
    te = parts["test"].copy(); te.loc[te.index[:5], "payment_type"] = "ZZ_UNSEEN"
    X = pre.transform(te)
    assert {k: list(v) for k, v in pre.categories_.items()} == before
    assert X["payment_type"].isna().iloc[:5].all() and "ZZ_UNSEEN" not in X["payment_type"].cat.categories
    assert pre.n_fit_rows_ == len(parts["train"])


def test_test_labels_cannot_influence_model_or_calibration(synth_df, cfgs, tiny_methods, tmp_path):
    """Run the pipeline twice: once as-is, once with all test-month labels scrambled.
    Model, selection, operating threshold and conformal thresholds must be identical."""
    import json
    cfgs["methods"]["conformal"]["methods"] = ["marginal", "class_conditional"]
    d1 = train_stage(synth_df, cfgs, "SYN", "lightgbm", tmp_path / "a", methods_override=tiny_methods)
    scr = synth_df.copy()
    rng = np.random.default_rng(5)
    m = scr.month.isin([6, 7])
    scr.loc[m, "fraud_bool"] = rng.permutation(scr.loc[m, "fraud_bool"].to_numpy())
    d2 = train_stage(scr, cfgs, "SYN", "lightgbm", tmp_path / "b", methods_override=tiny_methods)
    conformal_stage(d1, cfgs); conformal_stage(d2, cfgs)
    m1 = json.load(open(d1 / "run_metadata_train.json")); m2 = json.load(open(d2 / "run_metadata_train.json"))
    assert m1["selected_config"] == m2["selected_config"]
    assert m1["operating_threshold_from_validation"] == m2["operating_threshold_from_validation"]
    p1 = pd.read_parquet(d1 / "predictions.parquet"); p2 = pd.read_parquet(d2 / "predictions.parquet")
    np.testing.assert_array_equal(p1.proba.to_numpy(), p2.proba.to_numpy())          # same model => same scores
    c1 = json.load(open(d1 / "calibration.json")); c2 = json.load(open(d2 / "calibration.json"))
    assert c1 == c2
    # ...while the reported test metrics DO change (they use test labels), proving the labels were scrambled
    s1 = pd.read_csv(d1 / "conformal_summary.csv"); s2 = pd.read_csv(d2 / "conformal_summary.csv")
    assert not np.allclose(s1.coverage_pos.fillna(-1), s2.coverage_pos.fillna(-1))
