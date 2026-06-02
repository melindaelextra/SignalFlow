# SignalFlow: End-to-End ML Pipeline for Time-Series Risk Prediction

SignalFlow is an end-to-end machine learning project for predicting risk events from sensor time-series data. It includes data ingestion, validation, rolling-window feature engineering, model training, evaluation reports, FastAPI inference, Docker packaging, and basic tests.

## Why this project exists

Given historical sensor readings from a device, engine, or IoT system, SignalFlow predicts whether the system is likely to enter a risky/failure state soon.

Default task for NASA CMAPSS-style data:

> Predict `risk_label = 1` if remaining useful life is less than or equal to 30 cycles, otherwise `0`.

This project is designed to show practical AI engineering skills: data cleaning, time-series feature engineering, model evaluation, API serving, documentation, and monitoring-style reporting.

## Architecture

```text
Raw sensor data
   ↓
Data validation
   ↓
RUL + risk label creation
   ↓
Rolling-window feature engineering
   ↓
Train/validation/test split by unit
   ↓
Model training and evaluation
   ↓
FastAPI inference service
   ↓
Reports + latency benchmark
```

## Repo structure

```text
signalflow/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
├── src/
├── app/
├── models/
├── reports/
│   └── figures/
├── tests/
├── scripts/
└── experiments/
```

## Dataset

Recommended dataset: NASA CMAPSS turbofan engine degradation dataset.

Place the raw training file here:

```text
data/raw/train_FD001.txt
```

Expected columns are CMAPSS-style: unit id, cycle, operational settings, and sensor readings. If no raw dataset is found, the starter pipeline automatically generates a small synthetic sensor dataset so you can test the full project immediately.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python scripts/run_pipeline.py
```

This will:

1. Load CMAPSS data or generate synthetic sample data.
2. Create RUL and risk labels.
3. Generate rolling-window features.
4. Split by unit to avoid leakage.
5. Train Logistic Regression and Random Forest.
6. Save the best model to `models/best_model.pkl`.
7. Generate `reports/model_report.md` and figures.

## Run the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "engine_001",
    "readings": [
      {"cycle": 120, "sensor_1": 0.1, "sensor_2": 0.2, "sensor_3": 0.3}
    ]
  }'
```

## Run tests

```bash
pytest
```

## Current models

The starter version trains:

- Logistic Regression baseline
- Random Forest classifier

Optional next upgrade: add XGBoost or LightGBM after the basic pipeline works.

## Metrics generated

- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Feature importance chart
- Inference latency benchmark

## Limitations

- The synthetic dataset is only for testing the code path. Use CMAPSS or another real dataset for final GitHub results.
- The API currently accepts a small number of sensor fields for a simple demo. Expand the schema once you finalize the model feature set.
- This starter does not include cloud deployment yet. A good next step is Cloud Run deployment.
