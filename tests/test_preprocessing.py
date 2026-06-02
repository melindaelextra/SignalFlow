from src.data_ingestion import generate_synthetic_sensor_data
from src.preprocessing import clean_sensor_data


def test_preprocessing_removes_nulls():
    df = generate_synthetic_sensor_data(n_units=2, min_cycles=8, max_cycles=8)
    df.loc[0, "sensor_1"] = None
    clean = clean_sensor_data(df)
    assert clean.isna().sum().sum() == 0
