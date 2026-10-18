import numpy as np
import pytest

from conformal_fraud.conformal.calibration import SplitConformalClassifier, conformal_quantile


def test_quantile_known_values():
    s = np.arange(1, 10, dtype=float)             # n=9
    assert conformal_quantile(s, 0.1) == 9.0      # k = ceil(10*0.9) = 9
    s = np.arange(1, 20, dtype=float)             # n=19, k = ceil(20*0.9)=18
    assert conformal_quantile(s, 0.1) == 18.0
    s = np.arange(1, 101, dtype=float)            # n=100, k=ceil(101*0.9)=91
    assert conformal_quantile(s, 0.1) == 91.0


def test_quantile_infinite_when_too_few_points():
    assert np.isinf(conformal_quantile(np.arange(5.0), 0.1))   # k=ceil(6*.9)=6 > 5
    assert not np.isinf(conformal_quantile(np.arange(9.0), 0.1))


def test_quantile_order_invariant_and_validation():
    rng = np.random.default_rng(0)
    s = rng.random(50)
    assert conformal_quantile(s, 0.2) == conformal_quantile(rng.permutation(s), 0.2)
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal_quantile(s, bad)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([0.1, np.nan]), 0.1)


def test_quantile_not_plain_percentile():
    # np.quantile(s, 0.9) would give 8.2 for 1..9 (interpolated); conformal must give an actual score.
    s = np.arange(1, 10, dtype=float)
    assert conformal_quantile(s, 0.1) in s


def test_marginal_vs_class_conditional_thresholds():
    p = np.array([.1, .2, .3, .4, .6, .7, .8, .9, .95, .99] * 3)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 3)
    m = SplitConformalClassifier(0.2, "marginal").fit(p, y)
    c = SplitConformalClassifier(0.2, "class_conditional").fit(p, y)
    assert m.qhat[0] == m.qhat[1]
    assert set(c.qhat) == {0, 1} and c.n_cal == {0: 12, 1: 18}
    assert c.describe()["qhat_infinite"] is False


def test_class_conditional_needs_both_classes():
    with pytest.raises(ValueError):
        SplitConformalClassifier(0.1, "class_conditional").fit(np.array([.1, .2]), np.array([0, 0]))


def test_predict_before_fit():
    with pytest.raises(RuntimeError):
        SplitConformalClassifier(0.1).predict_sets(np.array([.5]))


def test_empirical_coverage_under_exchangeability():
    """Marginal and label-conditional coverage should be >= 1-alpha on average when cal/test are exchangeable."""
    rng = np.random.default_rng(1)
    alpha, reps = 0.1, 300
    cov_m, cov_c1 = [], []
    for _ in range(reps):
        def draw(n):
            y = (rng.random(n) < 0.2).astype(int)
            p = np.clip(np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n)), 1e-6, 1 - 1e-6)
            return p, y
        pc, yc = draw(400); pt, yt = draw(2000)
        sm = SplitConformalClassifier(alpha, "marginal").fit(pc, yc).predict_sets(pt)
        sc = SplitConformalClassifier(alpha, "class_conditional").fit(pc, yc).predict_sets(pt)
        cov_m.append(sm[np.arange(len(yt)), yt].mean())
        cov_c1.append(sc[yt == 1, 1].mean())
    # finite-sample guarantee is in expectation; allow Monte Carlo slack
    assert np.mean(cov_m) >= 1 - alpha - 0.005
    assert np.mean(cov_c1) >= 1 - alpha - 0.005
    assert np.mean(cov_m) <= 1 - alpha + 1 / 400 + 0.01   # and not wildly conservative
