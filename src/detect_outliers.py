import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the cleaned data
data = pd.read_csv('data/cleaned_data_with_anomalies.csv')

# Visualize outliers for each target using boxplots and violin plots
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.boxplot(x=data['IC50, mM'])
plt.title('IC50 Boxplot')

plt.subplot(2, 3, 2)
sns.boxplot(x=data['CC50, mM'])
plt.title('CC50 Boxplot')

plt.subplot(2, 3, 3)
sns.boxplot(x=data['SI'])
plt.title('SI Boxplot')

plt.subplot(2, 3, 4)
sns.violinplot(x=data['IC50, mM'])
plt.title('IC50 Violin Plot')

plt.subplot(2, 3, 5)
sns.violinplot(x=data['CC50, mM'])
plt.title('CC50 Violin Plot')

plt.subplot(2, 3, 6)
sns.violinplot(x=data['SI'])
plt.title('SI Violin Plot')

plt.tight_layout()
plt.show()

# Use IQR and z-score methods to detect and remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column):
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    return df[filtered_entries]

# Create separate datasets with and without outliers
data_no_outliers_iqr = remove_outliers(data, 'IC50, mM')
data_no_outliers_zscore = remove_outliers_zscore(data, 'IC50, mM')

# Save the datasets to new CSV files
data_no_outliers_iqr.to_csv('data/no_outliers_iqr.csv', index=False)
data_no_outliers_zscore.to_csv('data/no_outliers_zscore.csv', index=False)

print("Outlier detection and handling completed.")