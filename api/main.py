from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"


class InferenceRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Raw feature values for one customer record.",
    )


class InferenceResponse(BaseModel):
    prediction: int | str
    probability: float | None


app = FastAPI(
    title="Automated Model Validator Inference API",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def get_expected_features(model) -> list[str]:
    features = getattr(model, "feature_names_in_", None)
    if features is None:
        return []
    return [str(feature) for feature in features]


def make_input_frame(features: dict[str, Any], model) -> pd.DataFrame:
    expected_features = get_expected_features(model)
    if not expected_features:
        return pd.DataFrame([features])

    missing = [feature for feature in expected_features if feature not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required feature values.", "missing_features": missing},
        )

    return pd.DataFrame([{feature: features[feature] for feature in expected_features}])


def positive_class_probability(model, frame: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(frame)
    classes = list(getattr(model, "classes_", []))
    if not classes:
        return float(probabilities[0][-1])

    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return float(probabilities[0][positive_index])


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = DEFAULT_MODEL_PATH
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_available": model_path.exists(),
    }


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest) -> InferenceResponse:
    try:
        model = load_model()
        frame = make_input_frame(request.features, model)
        prediction = model.predict(frame)[0]
        probability = positive_class_probability(model, frame)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return InferenceResponse(prediction=prediction, probability=probability)
