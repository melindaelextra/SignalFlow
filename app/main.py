from __future__ import annotations

from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import RiskRequest, RiskResponse
from src.config import BASE_DIR
from src.predict import load_model_bundle, predict_risk

app = FastAPI(title="SignalFlow Risk Prediction API", version="0.1.0")
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
_model_bundle = None


@app.on_event("startup")
def load_model_on_startup() -> None:
    global _model_bundle
    if MODEL_PATH.exists():
        _model_bundle = load_model_bundle(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/predict-risk", response_model=RiskResponse)
def predict_single(request: RiskRequest) -> RiskResponse:
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not found. Run python scripts/run_pipeline.py first.")

    latest = request.readings[-1].model_dump()
    row = {
        "cycle": latest["cycle"],
        "cycle_norm": 1.0,
        "sensor_1": latest.get("sensor_1", 0.0),
        "sensor_2": latest.get("sensor_2", 0.0),
        "sensor_3": latest.get("sensor_3", 0.0),
    }
    features = pd.DataFrame([row])
    result = predict_risk(_model_bundle, features)
    return RiskResponse(**result, top_features=_model_bundle.get("feature_cols", [])[:5])


@app.post("/batch-predict")
def batch_predict(requests: list[RiskRequest]) -> dict:
    return {"predictions": [predict_single(req).model_dump() for req in requests]}
