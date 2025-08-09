# Data Cleaning

## Description
This step involves handling missing values, removing unnecessary columns, checking for duplicates, and performing anomaly detection.

## Tools Used
- Python for data cleaning.
- Pandas for data manipulation.
- Isolation Forest for anomaly detection.

## Scripts
- `src/clean_data.py`: Handles missing values, removes unnecessary columns, checks for duplicates, and performs anomaly detection.

## Key Findings
- Missing values were handled by filling them with the median of their respective columns.
- The `Unnamed: 0` column was removed as it was redundant.
- No duplicates or inconsistencies were found.
- Anomaly detection using Isolation Forest identified potential issues.

## Next Steps
- Proceed with outlier detection and handling.