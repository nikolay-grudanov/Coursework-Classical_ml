"""python src/train_pipelines_v2.py

Improved training script with robust column name detection.
"""
from pathlib import Path
import json
import platform
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Optional libraries
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


def get_versions():
    import sklearn
    versions = {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
    }
    if HAS_LGB:
        versions['lightgbm'] = lgb.__version__
    if HAS_SMOTE:
        import imblearn
        versions['imblearn'] = imblearn.__version__
    return versions


def find_column(df_columns, keywords):
    # keywords: list of substrings to match (case-insensitive). Return first matching column or None.
    for col in df_columns:
        name = str(col).lower()
        for kw in keywords:
            if kw.lower() in name:
                return col
    return None


def run_regression_cv(X, y, model, n_splits=5, random_state=42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = {'rmse': [], 'mae': [], 'r2': []}
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics['rmse'].append(float(rmse))
        metrics['mae'].append(float(mae))
        metrics['r2'].append(float(r2))
        print(f"[REG] Fold {fold} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    summary = {k + ('_mean' if k!='per_fold' else ''): (float(np.mean(v)) if isinstance(v, list) else v) for k,v in metrics.items()}
    summary = {
        'rmse_mean': float(np.mean(metrics['rmse'])),
        'rmse_std': float(np.std(metrics['rmse'], ddof=1)),
        'mae_mean': float(np.mean(metrics['mae'])),
        'mae_std': float(np.std(metrics['mae'], ddof=1)),
        'r2_mean': float(np.mean(metrics['r2'])),
        'r2_std': float(np.std(metrics['r2'], ddof=1)),
        'per_fold': metrics,
    }
    return summary


def run_classification_cv(X, y, model, use_smote=False, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imputer = SimpleImputer(strategy='median')
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
        if use_smote and HAS_SMOTE:
            sm = SMOTE(random_state=random_state)
            X_res_arr, y_res_arr = sm.fit_resample(X_train_imp, y_train)
            X_res = pd.DataFrame(X_res_arr, columns=X.columns)
            y_res = pd.Series(y_res_arr)
        else:
            X_res, y_res = X_train_imp, y_train
        scaler = StandardScaler()
        X_res_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)
        model.fit(X_res_scaled, y_res)
        preds = model.predict(X_test_scaled)
        prob = None
        try:
            prob = model.predict_proba(X_test_scaled)[:,1]
        except Exception:
            try:
                prob = model.decision_function(X_test_scaled)
            except Exception:
                prob = None
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = float(roc_auc_score(y_test, prob)) if prob is not None else None
        metrics['acc'].append(float(acc))
        metrics['prec'].append(float(prec))
        metrics['rec'].append(float(rec))
        metrics['f1'].append(float(f1))
        metrics['auc'].append(float(auc) if auc is not None else None)
        print(f"[CLF] Fold {fold} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc}")
    aucs = [a for a in metrics['auc'] if a is not None]
    summary = {
        'acc_mean': float(np.mean(metrics['acc'])),
        'acc_std': float(np.std(metrics['acc'], ddof=1)),
        'prec_mean': float(np.mean(metrics['prec'])),
        'prec_std': float(np.std(metrics['prec'], ddof=1)),
        'rec_mean': float(np.mean(metrics['rec'])),
        'rec_std': float(np.std(metrics['rec'], ddof=1)),
        'f1_mean': float(np.mean(metrics['f1'])),
        'f1_std': float(np.std(metrics['f1'], ddof=1)),
        'auc_mean': float(np.mean(aucs)) if len(aucs)>0 else None,
        'auc_std': float(np.std(aucs, ddof=1)) if len(aucs)>1 else None,
        'per_fold': metrics,
    }
    return summary


def save_artifact(imputer, scaler, model, path: Path):
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'model': model}, path)


def main(data_path: str, artifact_dir: str):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print('Versions:')
    versions = get_versions()
    print(json.dumps(versions, indent=2))

    print('Loading data from', data_path)
    df = pd.read_excel(data_path, sheet_name=0)
    print('Data shape:', df.shape)

    # find target columns
    ic50_col = find_column(df.columns, ['ic50'])
    cc50_col = find_column(df.columns, ['cc50'])
    # SI is short, so match exact 'si' token or column containing 'si'
    si_col = None
    for col in df.columns:
        if str(col).strip().lower() == 'si':
            si_col = col
            break
    if si_col is None:
        si_col = find_column(df.columns, ['si'])

    missing = [n for n,v in [('IC50', ic50_col), ('CC50', cc50_col), ('SI', si_col)] if v is None]
    if missing:
        raise ValueError(f'Missing expected columns in data: {missing}')

    print(f'Detected columns -> IC50: {ic50_col}, CC50: {cc50_col}, SI: {si_col}')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {ic50_col, cc50_col, si_col}
    FEATURES = [c for c in numeric_cols if c not in exclude_cols]
    print(f'Using {len(FEATURES)} numeric features (excluding IC50/CC50/SI)')

    report = {'versions': versions, 'tasks': {}}

    # IC50 regression
    reg_df = df.dropna(subset=[ic50_col]).reset_index(drop=True)
    X_reg = reg_df[FEATURES]
    y_ic50 = reg_df[ic50_col]

    reg_models = [
        ('LinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    if HAS_LGB:
        reg_models.append(('LightGBM', lgb.LGBMRegressor(random_state=42)))

    report['tasks']['IC50_regression'] = {}
    for name, model in reg_models:
        print('\n=== IC50 regression:', name)
        cv_res = run_regression_cv(X_reg, y_ic50, model, n_splits=5)
        report['tasks']['IC50_regression'][name] = cv_res
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_imp = pd.DataFrame(imputer.fit_transform(X_reg), columns=X_reg.columns)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_reg.columns)
        model.fit(X_scaled, y_ic50)
        artifact_path = artifact_dir / f'ic50_{name}.joblib'
        save_artifact(imputer, scaler, model, artifact_path)
        report['tasks']['IC50_regression'][name]['artifact'] = str(artifact_path)
        print('Saved artifact to', artifact_path)

    # SI >= 8 classification
    class_df = df.dropna(subset=[si_col]).reset_index(drop=True)
    X_clf = class_df[FEATURES]
    y_si8 = (class_df[si_col] >= 8).astype(int)

    clf_models = [
        ('LogisticRegression', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
    ]
    if HAS_LGB:
        clf_models.append(('LightGBM', lgb.LGBMClassifier(random_state=42)))

    report['tasks']['SI_ge_8_classification'] = {}
    for name, model in clf_models:
        print('\n=== SI>=8 classification:', name)
        cv_res = run_classification_cv(X_clf, y_si8, model, use_smote=True, n_splits=5)
        report['tasks']['SI_ge_8_classification'][name] = cv_res
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_imp = pd.DataFrame(imputer.fit_transform(X_clf), columns=X_clf.columns)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_clf.columns)
        if HAS_SMOTE:
            sm = SMOTE(random_state=42)
            X_res_arr, y_res_arr = sm.fit_resample(X_scaled, y_si8)
            X_res = pd.DataFrame(X_res_arr, columns=X_clf.columns)
            model.fit(X_res, y_res_arr)
        else:
            model.fit(X_scaled, y_si8)
        artifact_path = artifact_dir / f'si8_{name}.joblib'
        save_artifact(imputer, scaler, model, artifact_path)
        report['tasks']['SI_ge_8_classification'][name]['artifact'] = str(artifact_path)
        print('Saved artifact to', artifact_path)

    report_path = artifact_dir / 'training_report_v2.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print('\nWrote report to', report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx')
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()
    main(args.data, args.artifacts)
