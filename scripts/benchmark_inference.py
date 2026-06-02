from __future__ import annotations

from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.monitoring import benchmark_model


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "best_model.pkl")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X = test_df[bundle["feature_cols"]].head(500)
    print(benchmark_model(bundle["model"], X))


if __name__ == "__main__":
    main()
