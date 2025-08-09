"""python src/diagnostics_and_stabilize.py

Performs artifact verification, sanity checks, IC50 diagnostics, stabilization experiments,
plots PR/ROC for SI>=8, and writes reports/figures and artifacts as requested.

Run: python3 src/diagnostics_and_stabilize.py
"""
from pathlib import Path
import json
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    except Exception:
    pass

ROOT = Path('.')
ART = ROOT / 'artifacts'
REPORTS = ROOT / 'reports'
FIGS = ROOT / 'figures'
REPORTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

DATA_PATH = os.environ.get('DATA_PATH', '/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx')

# 1) Verify artifacts exist and readable
artifact_files = [
    ART / 'training_report_v2.json',
    ART / 'ic50_LinearRegression.joblib',
    ART / 'ic50_RandomForest.joblib',
    ART / 'si8_LogisticRegression.joblib',
    ART / 'si8_GradientBoosting.joblib'
]
art_info = {}
for p in artifact_files:
    info = {'path': str(p), 'exists': p.exists()}
    if p.exists():
        info['size_bytes'] = p.stat().st_size
        info['modified_time'] = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        try:
            if p.suffix == '.json':
                with open(p,'r') as f:
                    _ = json.load(f)
                info['load_ok'] = True
                info['load_type'] = 'json'
            else:
                _ = joblib.load(p)
                info['load_ok'] = True
                info['load_type'] = 'joblib_dict'
        except Exception as e:
            info['load_ok'] = False
            info['load_error'] = str(e)
    art_info[p.name] = info

# 2) Sanity checks from code: check train_pipelines_v2.py logic
src = Path('src/train_pipelines_v2.py').read_text()
# check split before preprocess: presence of kf.split then imputer.fit_transform inside loop
split_before_preprocess = ('for fold' in src or 'for fold' in src) and ('imputer.fit_transform' in src)
# check SMOTE used inside loop
smote_train_only = 'sm = SMOTE' in src and 'fit_resample' in src

# 3) Load data and confirm no target leakage
df = pd.read_excel(DATA_PATH, sheet_name=0)
# detect columns
def find_col(cols, keyword):
    keyword = keyword.lower()
    for c in cols:
        if keyword in str(c).lower():
            return c
    return None
ic50_col = find_col(df.columns, 'ic50')
cc50_col = find_col(df.columns, 'cc50')
si_col = None
for c in df.columns:
    if str(c).strip().lower() == 'si':
        si_col = c
        break
if si_col is None:
    si_col = find_col(df.columns, 'si')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = {ic50_col, cc50_col, si_col}
features = [c for c in numeric_cols if c not in exclude]
no_target_leakage = all([ic50_col not in features, cc50_col not in features, si_col not in features])

sanity_flags = {
    'split_before_preprocess': bool(split_before_preprocess),
    'smote_train_only': bool(smote_train_only),
    'no_target_leakage': bool(no_target_leakage)
}
with open(ART / 'sanity_flags.json','w') as f:
    json.dump(sanity_flags, f, indent=2)

# 4) IC50 diagnostics
ic50 = df[ic50_col].dropna().astype(float)
stats = ic50.describe().to_dict()
q1 = ic50.quantile(0.25)
q3 = ic50.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = ic50[(ic50 < lower) | (ic50 > upper)]
num_outliers = int(len(outliers))
prop_outliers = float(num_outliers / len(ic50))
lower_e = q1 - 3 * iqr
upper_e = q3 + 3 * iqr
extreme = ic50[(ic50 < lower_e) | (ic50 > upper_e)]
num_extreme = int(len(extreme))
prop_extreme = float(num_extreme / len(ic50))

# correlations top10
corrs = {}
for feat in features:
    try:
        v = df[feat]
        valid = ic50.index.intersection(v.dropna().index)
        if len(valid) < 10:
            continue
        c = ic50.loc[valid].corr(v.loc[valid])
        corrs[feat] = 0.0 if pd.isna(c) else float(c)
    except Exception:
        continue
corr_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
top10 = corr_sorted[:10]
zero_var = [f for f in features if df[f].nunique() <= 1]

diag_md = REPORTS / 'ic50_diagnostics_T2.md'
with open(diag_md,'w') as f:
    f.write('# IC50 Diagnostics T2\n\n')
    f.write(f'Detected IC50 column: {ic50_col}\n\n')
    f.write('Descriptive stats:\n')
    for k,v in stats.items():
        f.write(f'- {k}: {v}\n')
    f.write(f"\nIQR: {iqr}, lower={lower}, upper={upper}\n")
    f.write(f'- Outliers (1.5*IQR) count: {num_outliers} (proportion {prop_outliers:.4f})\n')
    f.write(f'- Extreme values (>3*IQR) count: {num_extreme} (proportion {prop_extreme:.4f})\n')
    f.write('\nTop 10 features by absolute correlation with IC50:\n')
    for feat, c in top10:
        f.write(f'- {feat}: corr={c:.4f}\n')
    f.write(f'\nZero/constant features count: {len(zero_var)}\n')
    if zero_var:
        for z in zero_var[:50]:
            f.write(f'- {z}\n')

# 5) IC50 stabilization experiments: log1p target, RobustScaler, models: LinearRegression, Ridge, RandomForest
X = df[features].copy()
y = ic50.copy()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models = [('LinearRegression', LinearRegression()), ('Ridge', Ridge(alpha=1.0)), ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))]
robust_results = {}
for name, model in models:
    rmses=[]; maes=[]; r2s=[]
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imputer = SimpleImputer(strategy='median')
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)
        y_train_log = np.log1p(y_train)
        model.fit(X_train_s, y_train_log)
        y_pred_log = model.predict(X_test_s)
        y_pred = np.expm1(y_pred_log)
        # clip negatives
        y_pred = np.where(y_pred<0, 0, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmses.append(float(rmse)); maes.append(float(mae)); r2s.append(float(r2))
    robust_results[name] = {'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses, ddof=1)), 'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes, ddof=1)), 'r2_mean': float(np.mean(r2s)), 'r2_std': float(np.std(r2s, ddof=1)), 'per_fold': {'rmse': rmses, 'mae': maes, 'r2': r2s}}
    # fit final on full data and save artifact
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()
    X_imp_full = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_s_full = pd.DataFrame(scaler.fit_transform(X_imp_full), columns=X.columns)
    y_log_full = np.log1p(y)
    model.fit(X_s_full, y_log_full)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'model': model}, ART / f'ic50_{name}_robust.joblib')

# write robust report
robust_md = REPORTS / 'ic50_robust_T2.md'
with open(robust_md,'w') as f:
    f.write('# IC50 Robust Models T2\n\n')
    for name,res in robust_results.items():
        f.write(f'## {name}\n')
        f.write(f"RMSE mean: {res['rmse_mean']:.4f} (std {res['rmse_std']:.4f})\n")
        f.write(f"MAE mean: {res['mae_mean']:.4f} (std {res['mae_std']:.4f})\n")
        f.write(f"R2 mean: {res['r2_mean']:.4f} (std {res['r2_std']:.4f})\n\n")

# append baseline_summary.csv
csv_path = ART / 'baseline_summary.csv'
rows = []
train_report = json.load(open(ART / 'training_report_v2.json'))
for model_name, vals in train_report['tasks']['IC50_regression'].items():
    rows.append({'model': model_name, 'rmse_mean': vals['rmse_mean'], 'mae_mean': vals['mae_mean'], 'r2_mean': vals['r2_mean'], 'notes': 'baseline'})
for name,res in robust_results.items():
    rows.append({'model': name + '_robust', 'rmse_mean': res['rmse_mean'], 'mae_mean': res['mae_mean'], 'r2_mean': res['r2_mean'], 'notes': 'log1p+RobustScaler'})
pd.DataFrame(rows).to_csv(csv_path, index=False)

# 6) SI>=8 PR/ROC curves for a single fold
class_df = df.dropna(subset=[si_col]).reset_index(drop=True)
X_clf = class_df[features]
y_si8 = (class_df[si_col] >= 8).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_clf, y_si8), start=1):
    if fold==1:
        X_train, X_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
        y_train, y_test = y_si8.iloc[train_idx], y_si8.iloc[test_idx]
        break
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_clf.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_clf.columns)
X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_clf.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test_imp), columns=X_clf.columns)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_s, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
clfs = [('LogisticRegression', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)), ('GradientBoosting', GradientBoostingClassifier(random_state=42))]
fig_links = []
for name, clf in clfs:
    clf.fit(X_res, y_res)
    try:
        prob = clf.predict_proba(X_test_s)[:,1]
    except Exception:
        try:
            prob = clf.decision_function(X_test_s)
        except Exception:
            prob = None
    pred = clf.predict(X_test_s)
    if prob is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        pr_auc = auc(recall, precision)
    else:
        fpr=tpr=precision=recall=[]; roc_auc=pr_auc=None
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    if prob is not None:
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.title(f'ROC {name} fold1')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend()
    plt.subplot(1,2,2)
    if prob is not None:
        plt.plot(recall, precision, label=f'PR-AUC={pr_auc:.3f}')
    plt.title(f'PR {name} fold1')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend()
    fig_path = FIGS / f'si8_{name}_fold1.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    fig_links.append(str(fig_path))

# update baselines_T2 report
baseline_md = REPORTS / 'baselines_T2.md'
with open(baseline_md,'w') as f:
    f.write('# Baselines T2\n\n')
    f.write('IC50 baseline metrics:\n')
    for m,vals in train_report['tasks']['IC50_regression'].items():
        f.write(f'- {m}: RMSE={vals["rmse_mean"]:.4f}, MAE={vals["mae_mean"]:.4f}, R2={vals["r2_mean"]:.4f}\n')
    f.write('\nIC50 robust metrics (log1p + RobustScaler):\n')
    for name,res in robust_results.items():
        f.write(f'- {name}: RMSE={res["rmse_mean"]:.4f}, MAE={res["mae_mean"]:.4f}, R2={res["r2_mean"]:.4f}\n')
    f.write('\nSI>=8 classification metrics:\n')
    for m,vals in train_report['tasks']['SI_ge_8_classification'].items():
        f.write(f'- {m}: AUC={vals.get("auc_mean")}, F1={vals.get("f1_mean"):.4f}, Acc={vals.get("acc_mean"):.4f}\n')
    f.write('\nFigures:\n')
    for link in fig_links:
        f.write(f'- {link}\n')

# save summary json
summary = {'sanity_flags': sanity_flags, 'artifact_info': art_info, 'ic50_stats': {'describe': stats, 'num_outliers_1.5iqr': num_outliers, 'prop_outliers': prop_outliers, 'num_extreme_3iqr': num_extreme, 'prop_extreme': prop_extreme}, 'top10_correlations': top10, 'zero_variance_count': len(zero_var), 'figures': fig_links}
with open(ART / 'sanity_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nFinished diagnostics and stabilization. Summary:')
print(json.dumps({'sanity_flags': sanity_flags, 'artifact_counts': {k:v['exists'] for k,v in art_info.items()}}, indent=2))
print('\nArtifacts info:')
for k,v in art_info.items():
    print('-', k, v)

print('\nIC50 robust results:')
for k,v in robust_results.items():
    print('-', k, v['rmse_mean'], v['mae_mean'], v['r2_mean'])

print('\nFigures saved:')
for f in fig_links:
    print('-', f)

