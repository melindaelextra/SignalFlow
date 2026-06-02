from __future__ import annotations

import pandas as pd
from src.config import RISK_RUL_THRESHOLD


def add_rul_and_risk_label(df: pd.DataFrame, threshold: int = RISK_RUL_THRESHOLD) -> pd.DataFrame:
    """Calculate remaining useful life and binary risk label per unit."""
    labeled = df.copy()
    max_cycle = labeled.groupby("unit_id")["cycle"].transform("max")
    labeled["rul"] = max_cycle - labeled["cycle"]
    labeled["risk_label"] = (labeled["rul"] <= threshold).astype(int)
    return labeled
