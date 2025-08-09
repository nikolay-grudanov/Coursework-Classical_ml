import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data without outliers (IQR method)
data = pd.read_csv('data/no_outliers_iqr.csv')

# Compute and visualize correlation heatmaps for all features vs. targets
correlation_matrix = data.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Remove features with NaN correlations or high multicollinearity
threshold = 0.7
high_corr_var = np.where(correlation_matrix > threshold)
high_corr_var = [(correlation_matrix.index[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

# Drop highly correlated features
data_reduced = data.drop(columns=[col for col, _ in high_corr_var])

# Save the reduced dataset to a new CSV file
data_reduced.to_csv('data/reduced_data.csv', index=False)

print("Correlation and feature analysis completed.")