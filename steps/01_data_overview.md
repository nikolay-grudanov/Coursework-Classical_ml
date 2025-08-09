# Data Overview

## Description
This step involves loading the data from the Excel file and performing an initial exploration to understand its structure and contents.

## Tools Used
- Python for data loading and exploration.
- Pandas for data manipulation.
- Matplotlib and Seaborn for visualization.

## Scripts
- `src/read_data.py`: Loads the data and saves it as a cleaned CSV file.
- `src/explore_data.py`: Performs exploratory data analysis on the cleaned data.

## Summary of Decisions and Results
- The dataset contains 1001 entries and 214 columns.
- The columns include `IC50, mM`, `CC50, mM`, and various numerical characteristics of chemical compounds.
- The `Unnamed: 0` column is redundant and was dropped.
- The `SI` (Selectivity Index) was calculated as the ratio of `CC50, mM` to `IC50, mM`.
- Missing values were handled by filling them with the median of their respective columns.

## Next Steps
- Proceed with further exploratory data analysis (EDA) to explore correlations and relationships between features and targets.