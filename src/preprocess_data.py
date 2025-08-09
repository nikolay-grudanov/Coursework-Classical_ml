import pandas as pd

# Load the cleaned data
data = pd.read_csv('data/cleaned_data.csv')

# Handle missing values by filling them with the median of their respective columns
data.fillna(data.median(), inplace=True)

# Save the preprocessed data to a new CSV file
data.to_csv('data/preprocessed_data.csv', index=False)

print("Data preprocessing completed.")