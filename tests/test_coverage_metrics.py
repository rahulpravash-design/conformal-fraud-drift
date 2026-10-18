import numpy as np
import pandas as pd
import pytest

from conformal_fraud.evaluation.classification_metrics import classification_metrics, threshold_at_fpr
from conformal_fraud.evaluation.coverage_metrics import coverage_summary
from conformal_fraud.evaluation.group_metrics import assign_group, group_gaps, group_table
from conformal_fraud.evaluation.review_metrics import review_summary
from conformal_fraud.statistics.confidence_intervals import bootstrap_ci, newcombe_diff_interval, wilson_interval


def test_coverage_hand_example():
    sets = np.array([[1, 0], [1, 1], [0, 0], [0, 1], [1, 0]], bool)
    y = np.array([0, 1, 1, 1, 1])
    r = coverage_summary(sets, y)
    # covered: yes, yes, no, yes, no -> 3/5
    assert r["coverage"] == pytest.approx(0.6)
    assert r["coverage_pos"] == pytest.approx(2 / 4) and r["coverage_neg"] == 1.0
    assert r["mean_set_size"] == pytest.approx(1.0)
    assert (r["frac_empty"], r["frac_singleton"], r["frac_both"]) == (0.2, 0.6, 0.2)
    assert r["n_pos"] == 4 and r["n_neg"] == 1


def test_wilson_known():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    assert np.isnan(wilson_interval(0, 0)[0])


def test_newcombe_contains_diff_and_symmetry():
    d, lo, hi = newcombe_diff_interval(90, 100, 80, 100)
    assert lo < d < hi and d == pytest.approx(0.1)
    d2, lo2, hi2 = newcombe_diff_interval(80, 100, 90, 100)
    assert (d2, lo2, hi2) == pytest.approx((-d, -hi, -lo))


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    pt, lo, hi = bootstrap_ci(lambda a: a.mean(), x, n_boot=100, seed=1)
    assert lo <= pt <= hi


def test_review_summary():
    sets = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], bool)   # approve, escalate, both, empty, approve
    y = np.array([0, 1, 1, 0, 1])
    r = review_summary(sets, y)
    assert r["review_rate"] == pytest.approx(2 / 5) and r["escalate_rate"] == pytest.approx(1 / 5) and r["approve_rate"] == pytest.approx(2 / 5)
    assert r["fraud_total"] == 3 and r["fraud_approved"] == 1 and r["fraud_reviewed"] == 1 and r["fraud_escalated"] == 1
    assert r["fraud_approved_rate"] == pytest.approx(1 / 3)


def test_threshold_at_fpr_uses_only_given_data_and_bounds_fpr():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < .05).astype(int)
    s = rng.random(5000) + .5 * y
    t = threshold_at_fpr(y, s, 0.05)
    assert (s[y == 0] > t).mean() <= 0.05
    m = classification_metrics(y, s, t)
    assert m["fpr"] <= 0.05 and 0 <= m["roc_auc"] <= 1 and m["tp"] + m["fn"] == y.sum()
    with pytest.raises(ValueError):
        classification_metrics(np.zeros(5, int), np.arange(5.0), 0.5)


def test_group_assignment_and_tables():
    df = pd.DataFrame({"customer_age": [20, 40, 50, 60, np.nan]})
    ge = assign_group(df, {"column": "customer_age", "op": ">=", "threshold": 50})
    gt = assign_group(df, {"column": "customer_age", "op": ">", "threshold": 50})
    assert list(ge[:4]) == ["low", "low", "high", "high"] and list(gt[:4]) == ["low", "low", "low", "high"] and ge.isna().iloc[4]
    rng = np.random.default_rng(0)
    n = 400
    g = pd.Series(rng.choice(["low", "high"], n)); mon = pd.Series(rng.choice([6, 7], n)); y = (rng.random(n) < .2).astype(int)
    sets = np.column_stack([rng.random(n) < .9, rng.random(n) < .9])
    t = group_table(sets, y, g, mon, min_pos=30)
    assert set(t.month) == {6, 7, "all"} and (t[t.month == "all"].n.sum() == n)
    gaps = group_gaps(t)
    assert {"gap_coverage", "gap_coverage_lo", "gap_coverage_hi", "gap_interpretable"} <= set(gaps.columns)
