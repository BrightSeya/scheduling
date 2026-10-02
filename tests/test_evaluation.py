"""Baseline and evaluation protocol. Data is synthetic."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.evaluation import (FORBIDDEN_FEATURES, LABEL, assert_no_leakage,
                            baseline_rate, calibration_table, evaluate,
                            load_bookings, run_baseline, time_split)
from app.main import app


def tiny_frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "created_at": pd.date_range("2024-01-01", periods=n, freq="h")[::-1],
        LABEL: [i % 4 == 0 for i in range(n)],
    }).astype({LABEL: int})


def test_split_is_chronological_and_75_25():
    train, test = time_split(tiny_frame(100))
    assert (len(train), len(test)) == (75, 25)
    assert train["created_at"].max() <= test["created_at"].min()


def test_baseline_uses_training_rate_only():
    df = tiny_frame(100)
    train, test = time_split(df)
    assert baseline_rate(train) == train[LABEL].mean()


def test_label_matches_the_no_show_definition():
    df = load_bookings()
    assert (df[LABEL] == (df["booking_status"] != "COMPLETED").astype(int)).all()


def test_dataset_is_the_expected_synthetic_set():
    df = load_bookings()
    assert len(df) == 21333
    assert df[LABEL].sum() == 4644
    assert df["is_synthetic"].all()


def test_baseline_roc_auc_is_chance():
    result = run_baseline()
    assert result["metrics"]["roc_auc"] == pytest.approx(0.5)
    assert result["train_rows"] + result["test_rows"] == 21333


@pytest.mark.parametrize("column", FORBIDDEN_FEATURES)
def test_leakage_guard_rejects_post_appointment_columns(column):
    with pytest.raises(ValueError):
        assert_no_leakage(["lead_time_hours", column])


def test_leakage_guard_accepts_pre_appointment_columns():
    assert_no_leakage(["lead_time_hours", "prior_no_shows", "rating_average"])


def test_calibration_table_accounts_for_every_row():
    table = calibration_table([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert sum(r["n"] for r in table) == 4
    assert all(0.0 <= r["observed"] <= 1.0 for r in table)


def test_evaluate_reports_all_metrics():
    out = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert out["roc_auc"] == 1.0
    assert out["precision"] == 1.0 and out["recall"] == 1.0


def test_endpoint_returns_the_baseline_probability():
    res = TestClient(app).post("/noshow/predict", json={"bookingId": "B0000000"})
    body = res.json()
    assert res.status_code == 200
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["factors"], list)
