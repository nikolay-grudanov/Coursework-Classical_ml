import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats

# Load the reduced dataset
data = pd.read_csv('data/reduced_data.csv')

# Create advanced visualizations such as pair plots, histograms with KDE, and scatter matrices for targets
sns.pairplot(data[['IC50, mM', 'CC50, mM', 'SI']])
plt.title('Pair Plot for Targets')
plt.show()

# Analyze class imbalances for classification tasks
data['IC50_above_median'] = (data['IC50, mM'] > data['IC50, mM'].median()).astype(int)
data['CC50_above_median'] = (data['CC50, mM'] > data['CC50, mM'].median()).astype(int)
data['SI_above_median'] = (data['SI'] > data['SI'].median()).astype(int)
data['SI_above_8'] = (data['SI'] > 8).astype(int)

# Perform PCA for dimensionality reduction visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data.drop(columns=['IC50_above_median', 'CC50_above_median', 'SI_above_median', 'SI_above_8']))

plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['SI_above_median'], cmap='viridis')
plt.title('PCA for Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='SI Above Median')
plt.show()

# Conduct statistical tests (e.g., t-tests) to check significance
t_stat, p_val = stats.ttest_ind(data[data['SI_above_median'] == 0]['SI'], data[data['SI_above_median'] == 1]['SI'])
print(f'T-test for SI above median: t-statistic = {t_stat}, p-value = {p_val}')

print("Advanced visualizations and insights completed.")