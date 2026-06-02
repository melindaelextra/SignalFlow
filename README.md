# 🚦 SignalFlow: End-to-End ML Pipeline for Time-Series Risk Prediction

SignalFlow is an end-to-end machine learning project for predicting risk events from sensor time-series data. It uses the NASA CMAPSS turbofan engine degradation dataset to predict whether an engine is approaching a high-risk failure state based on recent sensor readings.

The goal of this project is to demonstrate a complete AI engineering workflow: data ingestion, preprocessing, label creation, rolling-window feature engineering, model training, evaluation, inference API development, testing, and reporting.

## 🎯 Problem Statement

Given historical sensor readings from an engine, SignalFlow predicts whether the engine is entering a high-risk state.

For this version, the target label is defined as:

```text
risk_label = 1 if Remaining Useful Life <= 30 cycles
risk_label = 0 otherwise
```

This turns the original Remaining Useful Life problem into a binary risk classification task.

## 🌍 Why This Project Matters

Many real-world systems generate continuous sensor data, including industrial machines, IoT devices, smart buildings, environmental monitoring systems, and aquaculture infrastructure. Detecting risk early can help teams take action before a system reaches a failure state.

Although this project uses aircraft engine simulation data, the pipeline design is domain-general and can be adapted to other sensor-based prediction problems.

## 📊 Dataset

This project uses the NASA CMAPSS turbofan engine degradation dataset, specifically the FD001 training subset.

The dataset contains multivariate time-series data from multiple simulated engines. Each engine has a sequence of operating cycles and sensor readings. The objective is to model degradation patterns and estimate risk as the engine approaches failure.

Expected raw file location:

```text
data/raw/train_FD001.txt
```

## 🏗️ Project Architecture

```text
Raw sensor data
   ↓
Data ingestion
   ↓
Preprocessing and validation
   ↓
Remaining Useful Life label creation
   ↓
Rolling-window feature engineering
   ↓
Train/validation/test split by engine unit
   ↓
Model training and comparison
   ↓
Model evaluation and reporting
   ↓
FastAPI inference service
   ↓
Testing and inference benchmarking
```

## ✅ Features Implemented

* Load and clean raw CMAPSS sensor time-series data
* Create Remaining Useful Life labels
* Convert RUL prediction into binary risk classification
* Generate rolling-window sensor features
* Generate lag and difference features
* Split data by engine unit to reduce leakage
* Train and compare machine learning models
* Evaluate precision, recall, F1 score, ROC-AUC, and inference latency
* Save the best model for inference
* Serve predictions through a FastAPI API
* Generate model reports and figures
* Include automated tests for preprocessing, feature engineering, and API behavior

## 🧪 Model Results

The current results are based on the NASA CMAPSS FD001 training subset using a binary risk label where `RUL <= 30` cycles is treated as high risk.

| Model               | Precision | Recall | F1 Score | ROC-AUC | Avg Latency |
| ------------------- | --------: | -----: | -------: | ------: | ----------: |
| Logistic Regression |    0.8938 | 0.9957 |   0.9420 |  0.9993 |   0.0087 ms |
| Random Forest       |    0.9061 | 0.8925 |   0.8992 |  0.9952 |   0.1617 ms |

The best model in this run was **Logistic Regression**. Although Random Forest achieved slightly higher precision, Logistic Regression produced stronger recall, F1 score, ROC-AUC, and lower inference latency in this experiment.

## 🚀 API Endpoints

The project includes a FastAPI service for model inference.

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Single Prediction

```text
POST /predict-risk
```

Example request:

```json
{
  "unit_id": "engine_001",
  "readings": [
    {
      "cycle": 120,
      "sensor_1": 518.67,
      "sensor_2": 642.3,
      "sensor_3": 1581.2,
      "sensor_4": 1400.0,
      "sensor_5": 14.62
    }
  ]
}
```

Example response:

```json
{
  "risk_score": 0.87,
  "risk_label": "high",
  "model_version": "best_model"
}
```

### Batch Prediction

```text
POST /batch-predict
```

Used for generating predictions for multiple engines or multiple sensor windows.

## 📁 Repository Structure

```text
signalflow_starter/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── app/
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── experiments/
│   └── results.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── README.md
├── reports/
│   ├── model_report.md
│   └── figures/
├── scripts/
│   ├── run_pipeline.py
│   └── benchmark_inference.py
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── labeling.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── monitoring.py
└── tests/
    ├── test_api.py
    ├── test_features.py
    └── test_preprocessing.py
```

## ⚙️ How to Run Locally

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Add the dataset

Download the NASA CMAPSS dataset and place the FD001 training file here:

```text
data/raw/train_FD001.txt
```

### 4. Run the full pipeline

```powershell
python scripts/run_pipeline.py
```

This will train models, evaluate results, save the best model, and generate the report.

### 5. Run tests

```powershell
pytest
```

Expected result:

```text
3 passed
```

### 6. Start the API

```powershell
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## 📈 Reports

The pipeline generates a model report at:

```text
reports/model_report.md
```

Figures are saved under:

```text
reports/figures/
```

These reports summarize model performance, feature importance, prediction behavior, and evaluation results.

## 🛠️ Skills Demonstrated

This project demonstrates:

* Python machine learning development
* Time-series data preprocessing
* Feature engineering with rolling windows, lag features, and trend features
* Classification model training and evaluation
* Leakage-aware train/test splitting by engine unit
* Model serving with FastAPI
* API testing with pytest
* Inference latency measurement
* Reproducible project structure
* Technical documentation and reporting

## 🔮 Future Improvements

Planned improvements include:

* Add XGBoost or LightGBM model comparison
* Add LSTM sequence modeling
* Add experiment tracking for multiple training runs
* Add data validation checks for missing values, duplicate rows, and schema mismatches
* Add monitoring simulation for incoming sensor data
* Add Docker deployment instructions
* Deploy the API to Google Cloud Run
* Add a simple dashboard for prediction monitoring

## 📌 Project Status

Current version: working end-to-end local ML pipeline with FastAPI inference and passing tests.

The project can currently:

* Load CMAPSS data
* Generate RUL-based risk labels
* Build time-series features
* Train baseline models
* Save the best model
* Generate evaluation reports
* Serve predictions through an API
* Pass automated tests
