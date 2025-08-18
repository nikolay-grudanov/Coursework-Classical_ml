#!/usr/bin/env python3
"""src/summary_recommendations.py

Generate residuals analysis, feature importances, plots and a short recommendations
report. Reads processed data produced by the pipeline (data/processed_data.csv) and
saves outputs to figures/ and reports/.
"""

from pathlib import Path
import os
import yaml
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split

# Paths
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model_config.yaml"
FIG_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"
ANALYSIS_DIR = ROOT / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Load config if available
config = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

random_state = int(config.get("random_state", 42))
test_size = float(config.get("test_size", 0.2))

# Load processed data
processed_path = Path(config.get("processed_data_path", "data/processed_data.csv"))
if not processed_path.exists():
    processed_path = ROOT / "data" / "processed_data.csv"

if not processed_path.exists():
    raise FileNotFoundError(f"Processed data not found at {processed_path}")

print(f"Loading processed data from {processed_path}")
df = pd.read_csv(processed_path)

# Define targets
TARGET_REG = "IC50, mM"
TARGET_CLASS = "SI_above_8"

# Prepare feature columns: numeric, exclude known targets
exclude_cols = [
    TARGET_REG,
    "CC50, mM",
    "SI",
    "IC50_above_median",
    "CC50_above_median",
    "SI_above_median",
    TARGET_CLASS,
]
feature_cols = [
    c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
]
X = df[feature_cols].copy()
y_reg = df[TARGET_REG].copy()
y_clf = df[TARGET_CLASS].copy() if TARGET_CLASS in df.columns else None

# Train/test split for regression
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=test_size, random_state=random_state
)

# Helper to fill na for tree models
X_train_f = X_train.fillna(0)
X_test_f = X_test.fillna(0)

# Train RandomForestRegressor for analysis
rfr = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
print(f"Training RandomForestRegressor for {TARGET_REG}")
rfr.fit(X_train_f, y_train_reg)

preds = rfr.predict(X_test_f)
residuals = y_test_reg - preds

# Regression metrics
mse = mean_squared_error(y_test_reg, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, preds)
r2 = r2_score(y_test_reg, preds)
print(f"Regression metrics for {TARGET_REG}: rmse={rmse:.3f}, mae={mae:.3f}, r2={r2:.3f}")

# Plot residuals histogram
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Residual (y_true - y_pred)")
plt.title(f"Residuals distribution for {TARGET_REG}")
res_hist_path = FIG_DIR / "residuals_ic50_hist.png"
plt.tight_layout()
plt.savefig(res_hist_path)
plt.close()

# Predicted vs Actual scatter
plt.figure(figsize=(6, 6))
sns.scatterplot(x=preds, y=y_test_reg, alpha=0.6)
minv = min(y_test_reg.min(), preds.min())
maxv = max(y_test_reg.max(), preds.max())
plt.plot([minv, maxv], [minv, maxv], "r--")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Predicted vs Actual for {TARGET_REG}")
pp_path = FIG_DIR / "predicted_vs_actual_ic50.png"
plt.tight_layout()
plt.savefig(pp_path)
plt.close()

# Save residuals
residuals_df = pd.DataFrame({"actual": y_test_reg.values, "predicted": preds, "residual": residuals.values})
residuals_df.to_csv(ANALYSIS_DIR / "residuals_ic50.csv", index=False)

# Feature importances
importances = pd.Series(rfr.feature_importances_, index=feature_cols).sort_values(ascending=False)
important_top = importances.head(30)

plt.figure(figsize=(8, max(4, len(important_top) * 0.25)))
sns.barplot(x=important_top.values, y=important_top.index)
plt.xlabel("Importance")
plt.title("Top feature importances (RandomForestRegressor)")
fi_path = FIG_DIR / "feature_importances_ic50.png"
plt.tight_layout()
plt.savefig(fi_path)
plt.close()
important_top.to_csv(ANALYSIS_DIR / "feature_importances_ic50.csv", header=["importance"] )

# Classification analysis for SI_above_8 (if available)
if y_clf is not None:
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    rfc = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    print(f"Training RandomForestClassifier for {TARGET_CLASS}")
    rfc.fit(X_train_c.fillna(0), y_train_c)
    preds_c = rfc.predict(X_test_c.fillna(0))
    probs_c = rfc.predict_proba(X_test_c.fillna(0))[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test_c, preds_c)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix for SI > 8 classifier")
    cm_path = FIG_DIR / "confusion_matrix_si8.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test_c, probs_c)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve for SI > 8")
    plt.legend()
    pr_path = FIG_DIR / "pr_curve_si8.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    # Feature importances for classifier
    importances_c = pd.Series(rfc.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_c = importances_c.head(30)
    plt.figure(figsize=(8, max(4, len(top_c) * 0.25)))
    sns.barplot(x=top_c.values, y=top_c.index)
    plt.xlabel("Importance")
    plt.title("Top feature importances (RandomForestClassifier)")
    fi_c_path = FIG_DIR / "feature_importances_si8.png"
    plt.tight_layout()
    plt.savefig(fi_c_path)
    plt.close()
    top_c.to_csv(ANALYSIS_DIR / "feature_importances_si8.csv", header=["importance"]) 

# Write a short recommendations report
report_lines = []
report_lines.append("# Summary Recommendations")
report_lines.append("")
report_lines.append(f"Processed data: {processed_path}")
report_lines.append("")
report_lines.append("## Regression (IC50) analysis")
report_lines.append(f"- RMSE: {rmse:.3f}")
report_lines.append(f"- MAE: {mae:.3f}")
report_lines.append(f"- R2: {r2:.3f}")
report_lines.append("")
report_lines.append("Observed issues and recommendations:")
report_lines.append("- Residuals distribution saved to figures/residuals_ic50_hist.png. If residuals are skewed consider log-transforming target or using robust regression.")
report_lines.append("- Predicted vs Actual plot: figures/predicted_vs_actual_ic50.png. Large scatter indicates potential missing nonlinearities or noisy target.")
report_lines.append("- Top features are saved to analysis/feature_importances_ic50.csv and figure figures/feature_importances_ic50.png. Consider using these features for simpler models or interaction features.")
report_lines.append("")
report_lines.append("## Classification (SI > 8) analysis")
report_lines.append("- If present, confusion matrix and PR curve saved to figures/.")
report_lines.append("- If PR AUC is low, consider class rebalancing (SMOTE), different thresholds, or alternative models.")
report_lines.append("")
report_lines.append("## General recommendations")
report_lines.append("- Perform feature selection (use model-based importances or recursive feature elimination) to reduce noise and overfitting.")
report_lines.append("- Expand hyperparameter search for tree-based methods and consider stacking for improved performance.")
report_lines.append("- Validate models with repeated cross-validation and report mean±std for robust metrics.")
report_lines.append("- Add model interpretability (SHAP) for the final selected models to support domain conclusions.")

report_path = REPORTS_DIR / "summary_recommendations.md"
with open(report_path, "w") as fh:
    fh.write("\n".join(report_lines))

print(f"Analysis complete. Figures saved to {FIG_DIR}, report saved to {report_path}")
