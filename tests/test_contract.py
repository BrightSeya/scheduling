"""Contract test: the shape other teams rely on. If this breaks, you broke someone else."""
from fastapi.testclient import TestClient

from app.evaluation import load_bookings
from app.main import app

client = TestClient(app)
URL = "/noshow/predict"
KNOWN = "B0008746"


def predict(booking_id=KNOWN):
    return client.post(URL, json={"bookingId": booking_id})


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_valid_booking_returns_the_contract_shape():
    res = predict()
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body) == {"probability", "factors", "isSynthetic", "method"}
    assert isinstance(body["probability"], float)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["factors"] and all(isinstance(f, str) for f in body["factors"])
    assert body["isSynthetic"] is True


def test_unknown_booking_is_404():
    res = predict("nope")
    assert res.status_code == 404
    assert "nope" in res.json()["detail"]


def test_booking_created_through_the_slot_api_is_404():
    assert predict("bk_0123456789").status_code == 404


def test_missing_booking_id_is_422():
    assert client.post(URL, json={}).status_code == 422


def test_non_string_booking_id_is_422():
    assert client.post(URL, json={"bookingId": 123}).status_code == 422


def test_same_booking_gives_the_same_probability():
    assert predict().json()["probability"] == predict().json()["probability"]


def test_prediction_depends_on_the_booking():
    df = load_bookings()
    risky = df.loc[df["prior_no_shows"].idxmax(), "booking_id"]
    first = df[df["is_first_booking"] == 1].iloc[0]["booking_id"]
    assert predict(risky).json()["probability"] > predict(first).json()["probability"]


def test_service_starts_and_trains_the_model():
    with TestClient(app) as started:
        res = started.post(URL, json={"bookingId": KNOWN})
    assert res.status_code == 200
