import pandas as pd

# Load the preprocessed data
data = pd.read_csv('data/preprocessed_data.csv')

# Calculate the Selectivity Index (SI) if not present
if 'SI' not in data.columns:
    data['SI'] = data['CC50, mM'] / data['IC50, mM']

# Save the data with new features to a new CSV file
data.to_csv('data/feature_engineered_data.csv', index=False)

print("Feature engineering completed.")