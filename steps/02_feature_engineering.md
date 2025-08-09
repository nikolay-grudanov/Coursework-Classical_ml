# Feature Engineering

## Description
This step involves handling missing values, calculating the Selectivity Index (SI), and preparing the data for modeling.

## Tools Used
- Python for data manipulation.
- Pandas for handling missing values and feature engineering.

## Scripts
- `src/preprocess_data.py`: Handles missing values and saves the preprocessed data.
- `src/feature_engineering.py`: Calculates the Selectivity Index (SI) and saves the data with new features.

## Summary of Decisions and Results
- Missing values were filled with the median of their respective columns.
- The `SI` was calculated as the ratio of `CC50, mM` to `IC50, mM`.
- The data is now clean and ready for further analysis and modeling.

## Next Steps
- Proceed with building regression and classification models.