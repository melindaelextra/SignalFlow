from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score

from src.config import REPORTS_DIR, FIGURES_DIR


def _extract_feature_importance(model_bundle: dict) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False)


def evaluate_model(model_path: Path, test_df: pd.DataFrame, target_col: str = "risk_label") -> dict:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion Matrix")
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    feature_importance = _extract_feature_importance(bundle).head(20)
    fi_path = FIGURES_DIR / "feature_importance.png"
    if not feature_importance.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance["feature"][::-1], feature_importance["importance"][::-1])
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(fi_path, bbox_inches="tight")
        plt.close()

    pred_path = FIGURES_DIR / "prediction_distribution.png"
    plt.figure(figsize=(7, 4))
    plt.hist(proba, bins=30)
    plt.title("Prediction Score Distribution")
    plt.xlabel("Risk score")
    plt.ylabel("Count")
    plt.savefig(pred_path, bbox_inches="tight")
    plt.close()

    markdown = f"""# SignalFlow Model Report

## Model

Best model: `{bundle.get('model_name', 'unknown')}`

## Test metrics

- ROC-AUC: `{roc_auc:.4f}`
- PR-AUC: `{pr_auc:.4f}`

## Classification report

```text
{report_text}
```

## Generated figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`
- `reports/figures/prediction_distribution.png`
"""
    report_path = REPORTS_DIR / "model_report.md"
    report_path.write_text(markdown, encoding="utf-8")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "report_path": report_path}
