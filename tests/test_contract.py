"""Contract test: the shape other teams rely on. If this breaks, you broke someone else."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_endpoint_responds():
    res = client.post("/noshow/predict", json={"bookingId": "B0008746"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["factors"], list)
