# Baselines T3 — Summary Metrics

This file summarizes current baseline and stabilized model metrics (validation / OOF where applicable).

## IC50 Regression (RMSE / MAE / R2)

| Model | RMSE (mean) | MAE (mean) | R2 (mean) | Notes | Artifact |
|---|---:|---:|---:|---|---|
| RandomForest (baseline) | 331.683 | 199.274 | 0.2370 | Baseline random forest (CV) | artifacts/ic50_RandomForest.joblib |
| LinearRegression (stabilized) | 8.055585018046065e+12 | 5.696159126743899e+11 | -3.5043738868963313e+21 | "Stabilized" run (log1p + RobustScaler + winsorization). Results indicate numerical instability on LR even after stabilization — very large errors. | artifacts/ic50_LinearRegression_robust.joblib |
| Ridge (stabilized) | 1.6069737810847756e+21 | 1.1363020577940315e+20 | -1.3945545081007197e+38 | Ridge (log1p + RobustScaler) — unstable / divergent results observed. | artifacts/ic50_Ridge_robust.joblib |
| ElasticNet (stabilized) | N/A | N/A | N/A | ElasticNet stabilized result not available (no artifact / metrics found). Recommend running IC50 stabilization script to evaluate ElasticNet or re-run with tighter winsorization and model regularization. | N/A |


## SI ≥ 8 Classification (OOF metrics)
Metrics reported as mean across CV folds (out-of-fold / CV estimates).

| Model | ROC-AUC (mean) | PR-AUC (mean) | F1-macro (mean) | Accuracy (mean) | Notes | Artifact |
|---|---:|---:|---:|---:|---|---|
| LogisticRegression | 0.7077 | n/a | 0.5423 | 0.6543 | OOF CV metrics (5-fold). PR-AUC not stored in summary JSON. | artifacts/si8_LogisticRegression.joblib |
| GradientBoosting | 0.7441 | n/a | 0.5670 | 0.7023 | OOF CV metrics (5-fold). PR-AUC not stored in summary JSON. | artifacts/si8_GradientBoosting.joblib |


## Notes and Recommendations
- The IC50 stabilized LinearRegression and Ridge runs currently show extreme numerical instability (very large RMSE/MAE/R2). These likely stem from incomplete winsorization, remaining extreme outliers, or from applying transformations incorrectly. Investigate per-fold target distributions and ensure winsorization is applied on training folds only.
- ElasticNet stabilized results are missing. Run src/ic50_stabilize_T3.py (or equivalent) to obtain ElasticNet results with a hyperparameter sweep (alpha, l1_ratio) and return the best validation configuration.
- PR-AUC (precision-recall AUC) was computed in plotting/debug scripts but not saved into training_report_v2.json. If PR-AUC is required for final comparison, re-compute PR-AUC OOF values (use sklearn.metrics.average_precision_score or auc(recall, precision)) and append to the report.
- Artifacts referenced above are saved in the `artifacts/` folder. Use these to inspect models and regenerate metrics if needed.

Generated from artifacts/training_report_v2.json and artifacts/baseline_summary.csv.
