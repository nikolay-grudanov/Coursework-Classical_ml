#!/usr/bin/env python3
"""Compute subgroup RMSE/MAE and append results to reports/summary_recommendations.md

This script expects the following files to exist:
- data/processed_data.csv
- analysis/residuals_ic50.csv
- analysis/feature_importances_ic50.csv
- reports/summary_recommendations.md

It will compute RMSE/MAE by IC50 quartiles and by top feature quartiles and
append formatted markdown tables and short interpretations to the report.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC_PATH = ROOT / "data" / "processed_data.csv"
RES_PATH = ROOT / "analysis" / "residuals_ic50.csv"
FI_PATH = ROOT / "analysis" / "feature_importances_ic50.csv"
REPORT_PATH = ROOT / "reports" / "summary_recommendations.md"

if not PROC_PATH.exists():
    raise FileNotFoundError(PROC_PATH)
if not RES_PATH.exists():
    raise FileNotFoundError(RES_PATH)
if not FI_PATH.exists():
    raise FileNotFoundError(FI_PATH)
if not REPORT_PATH.exists():
    raise FileNotFoundError(REPORT_PATH)

def rmse_mae(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse)), float(mean_absolute_error(y_true, y_pred))

# Load files
proc = pd.read_csv(PROC_PATH)
res = pd.read_csv(RES_PATH)
fi = pd.read_csv(FI_PATH, index_col=0)

# Reconstruct test split to align indices: use same random_state and test_size as config
# Default values matching pipeline
random_state = 42
test_size = 0.2

feature_cols = [c for c in proc.columns if c not in ["IC50, mM", "CC50, mM", "SI",
                                                       "IC50_above_median", "CC50_above_median",
                                                       "SI_above_median", "SI_above_8"]]
X = proc[feature_cols]
Y = proc["IC50, mM"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Ensure order alignment: reset indices
X_test = X_test.reset_index(drop=True)
res = res.reset_index(drop=True)
# If lengths mismatch, we will trim/resample to min length
n = min(len(res), len(X_test))
if len(res) != len(X_test):
    print(f"Warning: residuals ({len(res)}) and X_test ({len(X_test)}) lengths differ, aligning first {n} rows")
res = res.iloc[:n].reset_index(drop=True)
X_test = X_test.iloc[:n].reset_index(drop=True)

# 1) RMSE/MAE by IC50 quartiles (based on actual values in residuals)
res_actual = res['actual'].astype(float)
quartiles = pd.qcut(res_actual, 4, labels=['Q1','Q2','Q3','Q4'])
by_q = []
for name, grp in res.groupby(quartiles):
    r, m = rmse_mae(grp['actual'], grp['predicted'])
    by_q.append((name, r, m, len(grp)))

# 2) RMSE/MAE by top feature quartiles
top_feature = fi.index[0]
if top_feature in X_test.columns:
    feat_vals = X_test[top_feature].astype(float)
    feat_q = pd.qcut(feat_vals, 4, labels=['FQ1','FQ2','FQ3','FQ4'])
    by_f = []
    tmp = res.copy()
    tmp['feat'] = feat_vals.values
    for name, grp in tmp.groupby(feat_q):
        r, m = rmse_mae(grp['actual'], grp['predicted'])
        by_f.append((name, r, m, len(grp)))
else:
    by_f = []

# Append to report
with open(REPORT_PATH, 'a') as fh:
    fh.write('\n\n')
    fh.write('## Subgroup error analysis (automatically generated)\n')
    fh.write('\n### RMSE/MAE by IC50 quartile (test set)\n')
    fh.write('\n|Quartile|RMSE|MAE|N|\n')
    fh.write('|:---|---:|---:|---:|\n')
    for name, r, m, cnt in by_q:
        fh.write(f'|name}|r:.3f}|m:.3f}|cnt}|\n')

    if by_f:
        fh.write(f'\n### RMSE/MAE by top feature quartile ({top_feature})\n')
        fh.write('\n|FeatQuartile|RMSE|MAE|N|\n')
        fh.write('|:---|---:|---:|---:|\n')
        for name, r, m, cnt in by_f:
            fh.write(f'|name}|r:.3f}|m:.3f}|cnt}|\n')

    # Add brief interpretations
    fh.write('\n### Brief interpretation\n')
    fh.write('\n- The table above shows the RMSE and MAE computed on the pipeline test set, split by quartiles of the actual IC50.\n')
    fh.write('- If RMSE/MAE substantially larger in certain quartiles, this indicates the model performs worse on that range and further stratified modeling or transformations may help.\n')
    if by_f:
        fh.write(f'- The top feature used for grouping is **top_feature}** (from analysis/feature_importances_ic50.csv). Compare error patterns across its quartiles to identify heterogeneity.\n')
    fh.write('\n- Recommended actions: log-transform the target, perform targeted feature engineering for quartiles with high errors, and consider separate models or calibration in those subranges.\n')

print('Subgroup metrics appended to', REPORT_PATH)
