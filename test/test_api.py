from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import api.main as api_main


class FakeModel:
    feature_names_in_ = np.array(["tenure", "MonthlyCharges"])
    classes_ = np.array([0, 1])

    def predict(self, frame):
        assert list(frame.columns) == ["tenure", "MonthlyCharges"]
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.25, 0.75]])


def test_predict_returns_prediction_and_probability(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: FakeModel())
    client = TestClient(api_main.app)

    response = client.post(
        "/predict",
        json={"features": {"tenure": 12, "MonthlyCharges": 64.5}},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "probability": 0.75}


def test_predict_validates_missing_features(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: FakeModel())
    client = TestClient(api_main.app)

    response = client.post("/predict", json={"features": {"tenure": 12}})

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["MonthlyCharges"]
