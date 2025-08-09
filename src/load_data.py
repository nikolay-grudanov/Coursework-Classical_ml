import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If .env exists, try to read data path overrides
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; fall back to environment variables if set
    pass

import os

DATA_PATH = os.environ.get("DATA_PATH", "data/data.xlsx")
# Load the data from the Excel file
print(f"Loading data from {DATA_PATH}")
data = pd.read_excel(DATA_PATH)

# Drop the 'Unnamed: 0' column if present
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Display basic information about the dataframe
print(data.info())

# Display summary statistics of the dataframe
print(data.describe())

# Check for missing values
print("Total missing values:", data.isnull().sum().sum())

# Visualize the distribution of IC50, CC50, and SI if columns exist
plt.figure(figsize=(15, 5))

if "IC50" in data.columns or "IC50, mM" in data.columns:
    ic50_col = "IC50" if "IC50" in data.columns else "IC50, mM"
    plt.subplot(1, 3, 1)
    sns.histplot(data[ic50_col].dropna(), kde=True)
    plt.title("IC50 Distribution")
else:
    plt.subplot(1, 3, 1)
    plt.text(0.5, 0.5, "IC50 column not found", ha="center")

if "CC50" in data.columns or "CC50, mM" in data.columns:
    cc50_col = "CC50" if "CC50" in data.columns else "CC50, mM"
    plt.subplot(1, 3, 2)
    sns.histplot(data[cc50_col].dropna(), kde=True)
    plt.title("CC50 Distribution")
else:
    plt.subplot(1, 3, 2)
    plt.text(0.5, 0.5, "CC50 column not found", ha="center")

if "SI" in data.columns:
    plt.subplot(1, 3, 3)
    sns.histplot(data["SI"].dropna(), kde=True)
    plt.title("SI Distribution")
else:
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, "SI column not found", ha="center")

plt.tight_layout()
plt.show()

print("Data loading and initial inspection completed.")
