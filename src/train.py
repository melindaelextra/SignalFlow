from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, MODELS_DIR, EXPERIMENTS_DIR


@dataclass
class TrainResult:
    best_model_name: str
    best_model_path: Path
    results: pd.DataFrame


def split_by_unit(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unit_id to avoid leakage across train/validation/test."""
    units = pd.Series(df["unit_id"].unique()).sample(frac=1, random_state=RANDOM_STATE).to_list()
    n_train = int(len(units) * train_frac)
    n_val = int(len(units) * val_frac)

    train_units = set(units[:n_train])
    val_units = set(units[n_train:n_train + n_val])
    test_units = set(units[n_train + n_val:])

    return (
        df[df["unit_id"].isin(train_units)].copy(),
        df[df["unit_id"].isin(val_units)].copy(),
        df[df["unit_id"].isin(test_units)].copy(),
    )


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target_col: str = "risk_label") -> TrainResult:
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        start_pred = time.perf_counter()
        _ = model.predict_proba(X_val.head(200))
        latency_ms = ((time.perf_counter() - start_pred) / max(1, len(X_val.head(200)))) * 1000

        metrics = {
            "model": name,
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, proba),
            "avg_latency_ms": latency_ms,
            "train_seconds": train_seconds,
        }
        rows.append(metrics)

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_name = name
            best_model = model

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"model": best_model, "feature_cols": feature_cols, "model_name": best_name}, best_model_path)
    results.to_csv(EXPERIMENTS_DIR / "results.csv", index=False)

    return TrainResult(best_name, best_model_path, results)
