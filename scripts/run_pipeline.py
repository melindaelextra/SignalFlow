from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_ingestion import load_raw_data, validate_raw_data
from src.preprocessing import clean_sensor_data
from src.labeling import add_rul_and_risk_label
from src.feature_engineering import create_time_series_features, get_model_feature_columns
from src.config import SENSOR_COLUMNS
from src.train import split_by_unit, train_models
from src.evaluate import evaluate_model


def main() -> None:
    raw_path = RAW_DATA_DIR / "train_FD001.txt"
    df = load_raw_data(raw_path if raw_path.exists() else None)
    validate_raw_data(df)

    clean = clean_sensor_data(df)
    labeled = add_rul_and_risk_label(clean)
    featured = create_time_series_features(labeled, sensor_cols=SENSOR_COLUMNS[:8], windows=(5, 10))

    train_df, val_df, test_df = split_by_unit(featured)
    feature_cols = get_model_feature_columns(featured)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    result = train_models(train_df, val_df, feature_cols)
    metrics = evaluate_model(result.best_model_path, test_df)

    print("Pipeline complete.")
    print(f"Best model: {result.best_model_name}")
    print(result.results.to_string(index=False))
    print(f"Report: {metrics['report_path']}")


if __name__ == "__main__":
    main()
