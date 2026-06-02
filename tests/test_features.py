from src.data_ingestion import generate_synthetic_sensor_data
from src.preprocessing import clean_sensor_data
from src.labeling import add_rul_and_risk_label
from src.feature_engineering import create_time_series_features


def test_feature_engineering_creates_expected_columns():
    df = generate_synthetic_sensor_data(n_units=3, min_cycles=10, max_cycles=12)
    df = clean_sensor_data(df)
    df = add_rul_and_risk_label(df)
    features = create_time_series_features(df, sensor_cols=["sensor_1"], windows=(5,))

    assert "sensor_1_roll_mean_5" in features.columns
    assert "sensor_1_lag_1" in features.columns
    assert "sensor_1_trend_5" in features.columns
    assert "cycle_norm" in features.columns
