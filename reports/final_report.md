# Final Report

## Project Overview
This project involves building machine learning models to predict the effectiveness of compounds based on data provided by chemists. The goal is to build forecasts that allow for the selection of the most effective combination of parameters for creating medicines.

## Data Overview
- The dataset contains 1001 entries and 214 columns.
- Key target variables include `IC50, mM`, `CC50, mM`, and `SI` (Selectivity Index).
- The data was cleaned by dropping redundant columns and handling missing values.

## Methodology
### Regression Models
- Linear Regression was used to predict `IC50, mM`, `CC50, mM`, and `SI`.
- The models were evaluated using Mean Squared Error (MSE) and R-squared (R2).

### Classification Models
- Logistic Regression and Random Forest were used to predict whether `IC50`, `CC50`, and `SI` exceed their respective medians, and whether `SI` exceeds 8.
- SMOTE was used to handle class imbalance.
- The models were evaluated using accuracy, precision, recall, and F1-score.

## Results
### Regression Models
- **IC50 Regression**: MSE: 248730.11, R2: 0.2543
- **CC50 Regression**: MSE: 323245.14, R2: 0.3765
- **SI Regression**: MSE: 1860608.73, R2: 0.0737

### Classification Models
- **IC50 Classification**: Accuracy: 0.7612
- **CC50 Classification**: Accuracy: 0.8060
- **SI Classification**: Accuracy: 0.7065
- **SI > 8 Classification**: Accuracy: 0.7114

## Conclusion
- The regression models provide a baseline for predicting `IC50`, `CC50`, and `SI`, with the `CC50` model being the most effective.
- The classification models, particularly the `CC50` model, offer a robust solution for predicting whether the values exceed their medians.
- The use of SMOTE and Random Forest significantly improved the classification performance by addressing class imbalance.

## Future Work
- Further refine the models by exploring additional features and algorithms.
- Conduct more extensive hyperparameter tuning to improve model performance.
- Validate the models on additional datasets to ensure robustness.

## References
- Scikit-learn documentation for model building and evaluation.
- Pandas documentation for data manipulation.
- Matplotlib and Seaborn documentation for visualization.