import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the feature-engineered data
data = pd.read_csv('data/feature_engineered_data.csv')

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

# Calculate and visualize the correlation matrix for the first 20 features and the target variables
correlation_matrix = data[['IC50, mM', 'CC50, mM', 'SI'] + data.columns[:20].tolist()].corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("Data visualization completed.")