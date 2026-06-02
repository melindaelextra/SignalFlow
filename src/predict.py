from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def load_model_bundle(model_path: Path) -> dict:
    return joblib.load(model_path)


def predict_risk(model_bundle: dict, features: pd.DataFrame) -> dict:
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    aligned = features.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[feature_cols]

    score = float(model.predict_proba(aligned)[:, 1][0])
    label = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"

    return {
        "risk_score": round(score, 4),
        "risk_label": label,
        "model_version": model_bundle.get("model_name", "unknown"),
    }
