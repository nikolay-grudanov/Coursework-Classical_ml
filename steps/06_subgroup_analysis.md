# Data Subgroup Analysis

## Overview
This step investigates how model performance varies across different subgroups of the data. This helps identify if there are specific ranges or categories where our models perform particularly well or poorly.

## Methods

1. Divided the data into subgroups based on target variable quartiles
2. Calculated residuals for each subgroup
3. Plotted actual vs. predicted values for each subgroup
4. Created residual distributions for each subgroup
5. Calculated error metrics (MAE, MSE, RMSE) for each subgroup

## Results

### IC50 Subgroup Analysis
![IC50 Subgroup Analysis](analysis/subgroup/ic50_subgroup_analysis.png)

### IC50 Residuals by Subgroup
![IC50 Residuals by Subgroup](analysis/subgroup/ic50_residuals_by_subgroup.png)

### Error Metrics by Subgroup

| Subgroup | MAE       | MSE        | RMSE      |
|----------|-----------|------------|-----------|
| Low      | [value]   | [value]    | [value]   |
| Mid-Low  | [value]   | [value]    | [value]   |
| Mid-High | [value]   | [value]    | [value]   |
| High     | [value]   | [value]    | [value]   |

## Key Findings

1. Model performance is [better/worse] for [specific subgroup]
2. Residuals are [larger/smaller] for [specific subgroup]
3. The model tends to [overpredict/underpredict] for [specific subgroup]

## Code

```python
src/data_subgroup_analysis.py
```

## Next Steps

Based on these findings, we may want to:
1. Investigate why the model performs differently for certain subgroups
2. Consider separate models for different subgroups if performance varies significantly
3. Focus on improving predictions for subgroups with poor performance