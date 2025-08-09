import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
data = pd.read_excel('/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx')

# Drop the 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Display basic information about the dataframe
print(data.info())

# Display summary statistics of the dataframe
print(data.describe())

# Check for missing values
print(data.isnull().sum().sum())

# Visualize the distribution of IC50, CC50, and SI
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(data['IC50, mM'], kde=True)
plt.title('IC50 Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['CC50, mM'], kde=True)
plt.title('CC50 Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data['SI'], kde=True)
plt.title('SI Distribution')

plt.tight_layout()
plt.show()

print("Data loading and initial inspection completed.")