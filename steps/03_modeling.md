# Modeling

## Description
This step involves building and evaluating regression and classification models to predict `IC50, mM`, `CC50, mM`, and `SI`.

## Tools Used
- Python for model building and evaluation.
- Scikit-learn for regression and classification models.
- SMOTE for handling class imbalance.
- Random Forest for classification.

## Scripts
- `src/regression_models.py`: Builds and evaluates regression models for `IC50, mM`, `CC50, mM`, and `SI`.
- `src/classification_models.py`: Builds and evaluates classification models for predicting whether `IC50`, `CC50`, and `SI` exceed their respective medians, and whether `SI` exceeds 8.

## Summary of Decisions and Results
- Regression models were built for `IC50, mM`, `CC50, mM`, and `SI` using Linear Regression.
- Classification models were built to predict whether `IC50`, `CC50`, and `SI` exceed their respective medians, and whether `SI` exceeds 8.
- SMOTE was used to handle class imbalance, and Random Forest was used for classification.
- The models were evaluated using accuracy, precision, recall, and F1-score.

## Next Steps
- Document the evaluation and analysis process in `steps`.
- Create a final report summarizing the findings and conclusions.