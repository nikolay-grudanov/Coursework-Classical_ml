# Feature Importance & Model Interpretation

## Overview
This step focuses on understanding which features are most important for our models' predictions. This helps us interpret the models and identify key factors influencing drug candidate selection.

## Methods

### Tree-based Models
For Random Forest models, we:
1. Extracted feature importances from the trained models
2. Created bar plots to visualize the most important features for each target

### Linear Models
For linear models, we:
1. Analyzed the coefficients to understand feature contributions

### SHAP Analysis
We used SHAP (SHapley Additive exPlanations) to:
1. Provide a more detailed view of feature contributions
2. Show both the magnitude and direction of feature effects

## Results

### Regression Feature Importances
![Regression Feature Importances](analysis/feature_importance/regression_feature_importance.png)

### Classification Feature Importances
![Classification Feature Importances](analysis/feature_importance/classification_feature_importance.png)

### SHAP Feature Importances
- IC50: ![IC50 SHAP](analysis/feature_importance/shap_ic50_importance.png)
- CC50: ![CC50 SHAP](analysis/feature_importance/shap_cc50_importance.png)
- SI: ![SI SHAP](analysis/feature_importance/shap_si_importance.png)

## Key Findings

1. The most important features for IC50 prediction are [list top features]
2. For CC50, the key features are [list top features]
3. SI prediction is most influenced by [list top features]
4. SHAP analysis reveals [describe any interesting patterns]

## Code

```python
src/feature_importance.py
```

## Next Steps

Based on these findings, we may want to:
1. Focus on the most important features for further analysis
2. Consider feature engineering based on the top features
3. Investigate interactions between important features