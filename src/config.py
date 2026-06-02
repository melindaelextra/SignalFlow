from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

RISK_RUL_THRESHOLD = 30
RANDOM_STATE = 42

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
ID_COLUMNS = ["unit_id", "cycle"]

for directory in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
