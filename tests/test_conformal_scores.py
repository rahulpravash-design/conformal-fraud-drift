import numpy as np
import pytest

from conformal_fraud.conformal.prediction_sets import prediction_sets, set_type
from conformal_fraud.conformal.scores import lac_score_matrix, lac_true_label_scores


def test_scores():
    p = np.array([0.1, 0.8])
    S = lac_score_matrix(p)
    np.testing.assert_allclose(S, [[0.1, 0.9], [0.8, 0.2]])      # col y = 1 - P(y|x)
    np.testing.assert_allclose(lac_true_label_scores(p, np.array([0, 1])), [0.1, 0.2])


def test_score_validation():
    for bad in ([1.2], [-0.1], [np.nan]):
        with pytest.raises(ValueError):
            lac_score_matrix(np.array(bad))
    with pytest.raises(ValueError):
        lac_true_label_scores(np.array([.5]), np.array([2]))


def test_set_construction_hand_worked():
    # Dyadic probabilities keep 1-p exact in floating point, so the boundary tie is well defined.
    # q=0.25: include y iff 1-P(y|x) <= 0.25 i.e. P(y|x) >= 0.75
    p = np.array([0.0625, 0.5, 0.9375, 0.75, 0.25])
    sets = prediction_sets(p, 0.25, 0.25)
    expected = np.array([[1, 0], [0, 0], [0, 1], [0, 1], [1, 0]], bool)
    # p=.75: s1 = .25 <= .25 -> included (ties inclusive); p=.25: s0 = .25 -> {0}; p=.5: both scores .5 -> empty
    np.testing.assert_array_equal(sets, expected)
    assert list(set_type(sets)) == ["only0", "empty", "only1", "only1", "only0"]


def test_calibration_and_test_scores_share_float_expression():
    # The calibration threshold is an actual calibration score computed as 1-p; a test point with the same p
    # must therefore be included at the boundary, regardless of float rounding of 1-p.
    from conformal_fraud.conformal.calibration import conformal_quantile
    from conformal_fraud.conformal.scores import lac_true_label_scores
    p_cal = np.array([0.7] * 9 + [0.9])
    s = lac_true_label_scores(p_cal, np.ones(10, int))
    q = conformal_quantile(s, 0.1)
    assert prediction_sets(np.array([0.7]), q, q)[0, 1]


def test_large_threshold_gives_both_and_inf_gives_both():
    p = np.array([0.2, 0.9])
    assert prediction_sets(p, 0.9, 0.9).all()
    assert prediction_sets(p, np.inf, np.inf).all()
