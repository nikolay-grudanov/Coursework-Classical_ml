import pandas as pd
import os

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "data/data.xlsx",
)

# Load the data from the Excel file
print(f"Loading data from {DATA_PATH}")
data = pd.read_excel(DATA_PATH)

# Drop the 'Unnamed: 0' column if present
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)

# Save the cleaned data to a new CSV file
os.makedirs("data", exist_ok=True)
cleaned_path = "data/cleaned_data.csv"
print(f"Saving cleaned data to {cleaned_path}")
data.to_csv(cleaned_path, index=False)

print("Data loaded and cleaned successfully.")
