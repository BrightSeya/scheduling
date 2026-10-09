"""Calibration, the trainable model and readable factors. Data is synthetic."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from app.evaluation import (LABEL, evaluate, expected_calibration_error,
                            load_bookings)
from app.explain import factors
from app.features import build_features, features_for_booking
from app.model import PlattCalibrator, calibration_report, train


@pytest.fixture(scope="module")
def trained():
    return train()


def test_ece_is_zero_when_predictions_match_observed_rates():
    assert expected_calibration_error([0, 0, 0, 1], [0.25] * 4) == pytest.approx(0.0)


def test_ece_is_large_when_predictions_are_overconfident():
    assert expected_calibration_error([0, 0, 0, 0], [0.9] * 4) == pytest.approx(0.9)


def test_evaluate_reports_brier_and_ece():
    out = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert out["brier"] == pytest.approx(0.025)
    assert "ece" in out


def test_platt_calibrator_keeps_the_ranking():
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    out = PlattCalibrator().fit(p, y).transform(p)
    assert np.all(np.diff(out) > 0)
    assert roc_auc_score(y, out) == pytest.approx(roc_auc_score(y, p))


def test_model_is_better_calibrated_than_the_baseline():
    rows = {r["name"]: r for r in calibration_report()}
    assert rows["logistic, calibrated"]["brier"] < rows["baseline (constant)"]["brier"]
    assert rows["logistic, calibrated"]["ece"] < 0.1


def test_probabilities_are_valid(trained):
    df = load_bookings().head(200)
    prob = trained.probability(build_features(df))
    assert ((prob >= 0) & (prob <= 1)).all()


def test_higher_history_of_no_shows_scores_higher(trained):
    df = load_bookings()
    risky = df.loc[df["prior_no_shows"].idxmax(), "booking_id"]
    safe = df[(df["prior_bookings"] > 0) & (df["prior_no_shows"] == 0)].iloc[0]["booking_id"]
    p_risky = trained.probability(features_for_booking(risky))[0]
    p_safe = trained.probability(features_for_booking(safe))[0]
    assert p_risky > p_safe


def test_factors_are_readable_and_tagged(trained):
    df = load_bookings()
    risky = df.loc[df["prior_no_shows"].idxmax(), "booking_id"]
    out = factors(trained.pipeline, features_for_booking(risky))
    assert len(out) == 3
    assert all(f.endswith("risk)") for f in out)
    assert any("prior no-show" in f for f in out)
