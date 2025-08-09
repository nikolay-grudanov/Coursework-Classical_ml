"""python src/ic50_stabilize_T3.py

Perform IC50 stabilization with gentle cleaning, winsorization at 99.0/99.5/99.9, log1p transform,
RobustScaler, and CV for LinearRegression, Ridge, ElasticNet, RandomForest.

Writes:
- reports/ic50_data_issues_T3.md
- reports/ic50_stability_T3.md
- artifacts/ic50_{model}_cappedpctl}_robust.joblib
- appends lines to artifacts/baseline_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import joblib

ROOT = Path('.')
ART = ROOT / 'artifacts'
REPORTS = ROOT / 'reports'
ART.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

DATA_PATH = '/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx'

def find_col(cols, keyword):
    keyword = keyword.lower()
    for c in cols:
        if keyword in str(c).lower():
            return c
    return None

# load
df = pd.read_excel(DATA_PATH, sheet_name=0)
ic50_col = find_col(df.columns, 'ic50')
if ic50_col is None:
    raise RuntimeError('IC50 column not found')

# detect NaN/Inf in IC50 and features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# list problematic columns with NaN or infinite
cols_with_nan = [c for c in numeric_cols if df[c].isna().any()]
cols_with_inf = []
for c in numeric_cols:
    s = df[c]
    if np.isinf(s.select_dtypes(include=[np.number]).to_numpy()).any():
        cols_with_inf.append(c)

# rows with any NaN or Inf in numeric features or target
def row_has_problem(row):
    for c in numeric_cols:
        v = row[c]
        if pd.isna(v):
            return True
        try:
            if np.isinf(v):
                return True
        except Exception:
            pass
    return False

problem_rows = []
for idx, row in df[numeric_cols].iterrows():
    if row_has_problem(row):
        problem_rows.append(idx)

# Write report of issues
issues_md = REPORTS / 'ic50_data_issues_T3.md'
with open(issues_md,'w') as f:
    f.write('# IC50 Data Issues T3\n\n')
    f.write(f'Detected IC50 column: {ic50_col}\n\n')
    f.write('Columns with NaN (numeric):\n')
    for c in cols_with_nan:
        f.write(f'- {c}: {df[c].isna().sum()} NaNs\n')
    f.write('\nColumns with infinite values:\n')
    for c in cols_with_inf:
        f.write(f'- {c}\n')
    f.write('\nRows with any NaN/Inf in numeric columns (showing up to 50 indices):\n')
    for idx in problem_rows[:50]:
        f.write(f'- row index: {idx}\n')
    f.write(f'\nTotal rows with numeric problems: {len(problem_rows)} / {len(df)}\n')

# For this task, prefer to delete rows with NaN in target (IC50). We'll drop rows where IC50 is NaN.
before_count = len(df)
df_clean = df.dropna(subset=[ic50_col]).reset_index(drop=True)
after_count = len(df_clean)

dropped_target_rows = before_count - after_count

# Winsorization levels
pcts = [0.99, 0.995, 0.999]
models = [('LinearRegression', LinearRegression()), ('Ridge', Ridge(alpha=1.0)), ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)), ('RandomForest', RandomForestRegressor(n_estimators=200, random_state=42))]

results_summary = []
X_full = df_clean.select_dtypes(include=[np.number]).copy()
# ensure we exclude target columns from features
exclude = {ic50_col}
features = [c for c in X_full.columns if c not in exclude]
X_full = X_full[features]
y_full = df_clean[ic50_col].astype(float)

for p in pcts:
    cap = float(y_full.quantile(p))
    y_capped = y_full.clip(upper=cap)
    y_log = np.log1p(y_capped)
    # CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models:
        rmses=[]; maes=[]; r2s=[]
        for train_idx, test_idx in kf.split(X_full):
            X_train = X_full.iloc[train_idx]
            X_test = X_full.iloc[test_idx]
            y_train = y_log.iloc[train_idx]
            y_test_orig = y_full.iloc[test_idx]
            # impute and scale
            imp = SimpleImputer(strategy='median')
            X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
            X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
            scaler = RobustScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
            X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
            # fit on log target
            model.fit(X_train_s, y_train)
            y_pred_log = model.predict(X_test_s)
            y_pred = np.expm1(y_pred_log)
            # reverse capping? we used capped target in training only — predicted can exceed cap; keep as is
            # compute metrics on original y_test_orig
            rmse = mean_squared_error(y_test_orig, y_pred, squared=False)
            mae = mean_absolute_error(y_test_orig, y_pred)
            r2 = r2_score(y_test_orig, y_pred)
            rmses.append(float(rmse)); maes.append(float(mae)); r2s.append(float(r2))
        res = {'pctl': p, 'model': name, 'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses, ddof=1)), 'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes, ddof=1)), 'r2_mean': float(np.mean(r2s)), 'r2_std': float(np.std(r2s, ddof=1))}
        results_summary.append(res)
        # fit final model on full data and save artifact
        imp = SimpleImputer(strategy='median')
        scaler = RobustScaler()
        X_imp_full = pd.DataFrame(imp.fit_transform(X_full), columns=X_full.columns)
        X_s_full = pd.DataFrame(scaler.fit_transform(X_imp_full), columns=X_full.columns)
        y_log_full = np.log1p(y_capped)
        model.fit(X_s_full, y_log_full)
        p_int = int(p * 1000)
        art_name = f"ic50_{name}_capped_{p_int}_robust.joblib"
        joblib.dump({'imputer': imp, 'scaler': scaler, 'model': model, 'cap': cap, 'pctl': p}, ART / art_name)

# append to artifacts/baseline_summary.csv
csv_path = ART / 'baseline_summary.csv'
rows = []
try:
    prev = pd.read_csv(csv_path)
except Exception:
    prev = pd.DataFrame()
for r in results_summary:
    p_int = int(r['pctl'] * 1000)
    rows.append({'model': f"{r['model']}_capped_{p_int}", 'rmse_mean': r['rmse_mean'], 'mae_mean': r['mae_mean'], 'r2_mean': r['r2_mean'], 'notes': f"capped_{r['pctl']}"})
newdf = pd.DataFrame(rows)
if prev.empty:
    newdf.to_csv(csv_path, index=False)
else:
    pd.concat([prev, newdf], ignore_index=True).to_csv(csv_path, index=False)

# write results report
report_md = REPORTS / 'ic50_stability_T3.md'
with open(report_md,'w') as f:
    f.write('# IC50 Stability T3\n\n')
    f.write(f'Dropped {dropped_target_rows} rows with NaN in target IC50 (from {before_count} to {after_count}).\n\n')
    f.write('Winsorization percentiles tested: 99.0, 99.5, 99.9\n\n')
    f.write('Results (RMSE / MAE / R2):\n')
    for r in results_summary:
        f.write(f"- pctl={r['pctl']}: {r['model']}: RMSE={r['rmse_mean']:.4f} (std {r['rmse_std']:.4f}), MAE={r['mae_mean']:.4f}, R2={r['r2_mean']:.4f}\n")

print('Done. Wrote reports and artifacts.')

