# Residual Analysis & Error Diagnostics

## Overview
This step focuses on analyzing the residuals (errors) from our regression models and the performance of our classification models. The goal is to understand how well our models are performing and identify any systematic patterns in the errors.

## Methods

### Regression Models
For each regression model (IC50, CC50, and SI), we:
1. Calculated residuals as the difference between actual and predicted values
2. Created scatter plots of actual vs. residuals to visualize error patterns
3. Plotted residual distributions using histograms with KDE
4. Created boxplots to show the spread of residuals
5. Generated QQ-plots to check if residuals follow a normal distribution

### Classification Models
For each classification model, we:
1. Generated confusion matrices to visualize performance
2. Analyzed the distribution of misclassified samples

## Results

### Residual Scatter Plots
![Residual Scatter Plots](analysis/residuals/residuals_scatter.png)

### Residual Distributions
![Residual Distributions](analysis/residuals/residuals_distribution.png)

### Residual Boxplots
![Residual Boxplots](analysis/residuals/residuals_boxplot.png)

### QQ-Plots
![QQ-Plots](analysis/residuals/residuals_qqplot.png)

### Confusion Matrices
![Confusion Matrices](analysis/residuals/confusion_matrices.png)

## Key Findings

1. The residual plots show [describe any patterns or trends observed]
2. The distribution of residuals appears to be [describe the distribution]
3. The QQ-plots indicate [describe normality or deviations]
4. The confusion matrices show [describe classification performance]

## Code

```python
src/residual_analysis.py
```

## Next Steps

Based on these findings, we may want to:
1. Investigate features that contribute to large residuals
2. Consider transforming target variables if residuals are not normally distributed
3. Focus on improving classification performance for specific classes