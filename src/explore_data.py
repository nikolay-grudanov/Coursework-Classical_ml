import pandas as pd

# Load the cleaned data
data = pd.read_csv('data/cleaned_data.csv')

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Display basic information about the dataframe
print(data.info())

# Display summary statistics of the dataframe
print(data.describe())

# Check for missing values
print(data.isnull().sum().sum())

print("Exploratory data analysis completed.")