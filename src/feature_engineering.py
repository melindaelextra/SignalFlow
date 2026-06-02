from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import SENSOR_COLUMNS


def create_time_series_features(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    windows: tuple[int, ...] = (5, 10, 20),
    include_base_sensors: bool = True,
) -> pd.DataFrame:
    """Create rolling, lag, difference, trend, and normalized cycle features per unit.

    The trend feature uses a fast approximation:
    current value minus value from N cycles ago, divided by N.
    """
    sensor_cols = sensor_cols or SENSOR_COLUMNS
    base = df.copy().sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    new_features: dict[str, pd.Series] = {}

    max_cycle = base.groupby("unit_id")["cycle"].transform("max")
    new_features["cycle_norm"] = base["cycle"] / max_cycle

    for col in sensor_cols:
        grouped = base.groupby("unit_id")[col]
        new_features[f"{col}_lag_1"] = grouped.shift(1)
        new_features[f"{col}_diff_1"] = grouped.diff(1)

        for window in windows:
            rolling = grouped.rolling(window=window, min_periods=1)
            new_features[f"{col}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            new_features[f"{col}_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
            new_features[f"{col}_roll_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
            new_features[f"{col}_roll_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
            new_features[f"{col}_trend_{window}"] = (base[col] - grouped.shift(window)) / window

    features = pd.DataFrame(new_features, index=base.index)
    out = pd.concat([base, features], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in out.columns if c != "risk_label"]
    out[feature_cols] = out[feature_cols].fillna(0)

    if not include_base_sensors:
        out = out.drop(columns=sensor_cols)
    return out


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"unit_id", "risk_label", "rul"}
    return [c for c in df.columns if c not in excluded]
