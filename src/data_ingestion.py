from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import SENSOR_COLUMNS, SETTING_COLUMNS, ID_COLUMNS, RANDOM_STATE


def load_cmapss_train(path: Path) -> pd.DataFrame:
    """Load a CMAPSS-style training txt file.

    CMAPSS files are whitespace-separated and usually contain:
    unit_id, cycle, 3 operational settings, and 21 sensor columns.
    """
    columns = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(columns)]
    df.columns = columns
    return df


def generate_synthetic_sensor_data(n_units: int = 20, min_cycles: int = 50, max_cycles: int = 90) -> pd.DataFrame:
    """Generate a small synthetic dataset so the project can run without CMAPSS."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for unit in range(1, n_units + 1):
        total_cycles = int(rng.integers(min_cycles, max_cycles + 1))
        base_health = rng.normal(0, 0.3)

        for cycle in range(1, total_cycles + 1):
            degradation = cycle / total_cycles
            row = {
                "unit_id": unit,
                "cycle": cycle,
                "setting_1": rng.normal(0, 1),
                "setting_2": rng.normal(0, 1),
                "setting_3": rng.normal(0, 1),
            }
            for i in range(1, 22):
                noise = rng.normal(0, 0.08)
                row[f"sensor_{i}"] = base_health + (0.02 * i * degradation) + noise
            rows.append(row)

    return pd.DataFrame(rows)


def load_raw_data(raw_path: Path | None = None) -> pd.DataFrame:
    """Load CMAPSS if available; otherwise return synthetic data."""
    if raw_path and raw_path.exists():
        return load_cmapss_train(raw_path)
    return generate_synthetic_sensor_data()


def validate_raw_data(df: pd.DataFrame) -> None:
    required = set(ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[ID_COLUMNS].isna().any().any():
        raise ValueError("unit_id and cycle cannot contain missing values")
    duplicated = df.duplicated(subset=["unit_id", "cycle"]).sum()
    if duplicated:
        raise ValueError(f"Found duplicate unit_id/cycle rows: {duplicated}")
