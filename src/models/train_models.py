"""Train regression and classification models used by src/main.py

Provides:
- train_regression_models
- train_classification_models
- save_model_results

This is a lightweight, deterministic implementation intended as a template
that other pipeline code in the repo will call.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
)

try:
    # root_mean_squared_error was added in newer scikit-learn versions.
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def _safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception:
        # fallback for models that return array-like
        return np.array(model.predict(X))


def train_regression_models(
    X_train, y_train, X_test, y_test, target_name: str
) -> Dict[str, Dict[str, float]]:
    """Train a small set of regression models and return metrics.

    Returns a dict keyed by model name, each value a dict with keys 'mse', 'rmse', 'r2'.
    """
    # Models to evaluate. We'll include Ridge and ElasticNet (ElasticNet tuned via GridSearchCV)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=50, random_state=42
        ),
    }

    # ElasticNet will be trained via internal CV grid search and appended to results
    # Narrowed alpha grid to improve convergence and reduce solver warnings
    enet_param_grid = {
        # use a slightly coarser but larger-scale alpha grid to improve
        # numerical stability and reduce solver iterations / warnings
        "alpha": [1e-2, 1e-1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    }

    results: Dict[str, Dict[str, float]] = {}

    # Preprocessing: impute and scale fitted on training data only to avoid leakage
    imputer = SimpleImputer(strategy="median")
    # Use StandardScaler for regression models: linear solvers converge better
    # with zero-mean unit-variance features.
    from sklearn.preprocessing import StandardScaler as _StdScaler

    scaler = _StdScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Target handling: for IC50 apply winsorization (on train only) and log1p transform
    y_train_proc = y_train.copy()

    def inverse_transform(preds):
        return preds

    if isinstance(target_name, str) and target_name.lower().startswith("ic50"):
        # winsorize using 99.5 percentile computed on training target only
        cap = float(np.nanpercentile(y_train_proc, 99.5))
        y_train_capped = np.minimum(y_train_proc, cap)
        # log-transform to stabilize variance; store inverse as expm1
        y_train_log = np.log1p(y_train_capped)
        y_train_proc = y_train_log

        def inverse_transform(preds):
            return np.expm1(preds)

    # Fit simple models (single loop) and record metrics
    for name, m in models.items():
        logger.info(f"Training {name} for {target_name}")
        m.fit(X_train_scaled, y_train_proc)
        preds = _safe_predict(m, X_test_scaled)
        # inverse transform preds if needed
        preds_inv = inverse_transform(preds)
        mae = float(mean_absolute_error(y_test, preds_inv))
        mse = float(mean_squared_error(y_test, preds_inv))
        if root_mean_squared_error is not None:
            rmse = float(root_mean_squared_error(y_test, preds_inv))
        else:
            rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds_inv))
        results[name] = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
        logger.info(f"{name} {target_name} -> rmse={rmse:.4f}, r2={r2:.4f}")

    # ElasticNet: perform GridSearchCV on training data only (with preprocessing already applied)
    try:
        # increase max_iter and tighten tolerance to help convergence
        # for coordinate descent on ElasticNet (still deterministic)
        enet = ElasticNet(max_iter=50000, tol=1e-4, random_state=42)
        gs = GridSearchCV(
            enet,
            enet_param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        gs.fit(X_train_scaled, y_train_proc)
        best = gs.best_estimator_
        logger.info(f"ElasticNet best params: {gs.best_params_}")
        preds_enet = best.predict(X_test_scaled)
        preds_enet_inv = inverse_transform(preds_enet)
        mae_en = float(mean_absolute_error(y_test, preds_enet_inv))
        mse_en = float(mean_squared_error(y_test, preds_enet_inv))
        if root_mean_squared_error is not None:
            rmse_en = float(root_mean_squared_error(y_test, preds_enet_inv))
        else:
            rmse_en = float(np.sqrt(mse_en))
        r2_en = float(r2_score(y_test, preds_enet_inv))
        results["ElasticNet"] = {
            "mse": mse_en,
            "mae": mae_en,
            "rmse": rmse_en,
            "r2": r2_en,
            "best_params": gs.best_params_,
        }
    except Exception as e:
        logger.warning(f"ElasticNet training failed: {e}")
        results["ElasticNet"] = {"error": str(e)}

    return results


def train_classification_models(
    X_train, y_train, X_test, y_test, target_name: str
) -> Dict[str, Dict[str, float]]:
    """Train a small set of classification models and return metrics.

    Returns a dict keyed by model name, each value a dict with key 'accuracy'.
    """
    models = {
        # increase max_iter for logistic regression to avoid convergence warnings
        "LogisticRegression": LogisticRegression(max_iter=20000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=50, random_state=42
        ),
    }

    results: Dict[str, Dict[str, float]] = {}

    # Preprocessing for classification: impute and scale fitted on train only
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # For OOF average precision (train set) use StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, m in models.items():
        logger.info(f"Training {name} for {target_name}")
        # Compute OOF probabilities on training set
        oof_probs = np.zeros(len(X_train_scaled))
        from sklearn.base import clone

        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train.iloc[train_idx]
            m_clone = clone(m)
            m_clone.fit(X_tr, y_tr)
            prob = None
            try:
                prob = m_clone.predict_proba(X_val)[:, 1]
            except Exception:
                try:
                    prob = m_clone.decision_function(X_val)
                except Exception:
                    prob = m_clone.predict(X_val)
            oof_probs[val_idx] = prob
        # OOF average precision on train
        try:
            oof_ap = float(average_precision_score(y_train, oof_probs))
        except Exception:
            oof_ap = None

        # Fit final model on full training set and evaluate on test
        m.fit(X_train_scaled, y_train)
        preds = _safe_predict(m, X_test_scaled)
        acc = float(accuracy_score(y_test, preds))
        test_prob = None
        try:
            test_prob = m.predict_proba(X_test_scaled)[:, 1]
        except Exception:
            try:
                test_prob = m.decision_function(X_test_scaled)
            except Exception:
                test_prob = None
        test_ap = (
            float(average_precision_score(y_test, test_prob))
            if test_prob is not None
            else None
        )

        results[name] = {"accuracy": acc, "oof_pr_auc": oof_ap, "test_pr_auc": test_ap}
        logger.info(
            f"{name} {target_name} -> accuracy={acc:.4f}, test_pr_auc={test_ap}"
        )

    return results


def save_model_results(results: Dict[str, Any], path: str) -> None:
    """Save results dict as JSON.

    Creates parent directories if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Saved model results to {path}")
