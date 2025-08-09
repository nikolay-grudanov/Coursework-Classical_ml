import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the cleaned data
data = pd.read_csv('data/cleaned_data.csv')

# Handle missing values by filling them with the median of their respective columns
data.fillna(data.median(), inplace=True)

# Check for duplicates
duplicates = data.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')

# Perform anomaly detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(data)
data['anomaly'] = anomalies

# Save the cleaned data to a new CSV file
data.to_csv('data/cleaned_data_with_anomalies.csv', index=False)

print("Data cleaning completed.")