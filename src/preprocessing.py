from __future__ import annotations

import pandas as pd
from src.config import SENSOR_COLUMNS, SETTING_COLUMNS


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows, coerce numeric fields, and fill missing sensor values per unit."""
    clean = df.copy()
    clean = clean.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    numeric_cols = ["cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean[SENSOR_COLUMNS + SETTING_COLUMNS] = (
        clean.groupby("unit_id", group_keys=False)[SENSOR_COLUMNS + SETTING_COLUMNS]
        .apply(lambda g: g.ffill().bfill())
    )
    clean[SENSOR_COLUMNS + SETTING_COLUMNS] = clean[SENSOR_COLUMNS + SETTING_COLUMNS].fillna(
        clean[SENSOR_COLUMNS + SETTING_COLUMNS].median(numeric_only=True)
    )
    return clean
