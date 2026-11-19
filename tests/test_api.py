import pytest
from fastapi.testclient import TestClient

import app.api as api

VALID_TRIP = {
    "pulocationid": 161,
    "dolocationid": 236,
    "tpep_pickup_datetime": "2025-01-15T08:30:00",
}


class DummyModel:
    """Stands in for the MLflow model so tests don't need a running server."""

    def predict(self, X):
        return [13.6] * len(X)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: (DummyModel(), "test"))
    with TestClient(api.app) as c:  # `with` runs the lifespan (model loading)
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK", "model_version": "test"}


def test_predict_returns_rounded_minutes(client):
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    assert response.json() == {"duration": "14 minutes"}


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("tpep_pickup_datetime", "not-a-date"),
        ("pulocationid", "abc"),
    ],
)
def test_predict_rejects_invalid_input(client, field, bad_value):
    response = client.post("/predict", json={**VALID_TRIP, field: bad_value})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    trip = {k: v for k, v in VALID_TRIP.items() if k != "dolocationid"}
    response = client.post("/predict", json=trip)
    assert response.status_code == 422
