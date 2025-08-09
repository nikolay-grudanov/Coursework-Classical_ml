import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.stats as stats
import os

# Create analysis directory if it doesn't exist
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/residuals', exist_ok=True)

# Load the prepared datasets
test_features = pd.read_csv('data/test_features.csv')
test_ic50 = pd.read_csv('data/test_ic50.csv')
test_cc50 = pd.read_csv('data/test_cc50.csv')
test_si = pd.read_csv('data/test_si.csv')
test_ic50_class = pd.read_csv('data/test_ic50_class.csv')
test_cc50_class = pd.read_csv('data/test_cc50_class.csv')
test_si_class = pd.read_csv('data/test_si_class.csv')
test_si_8_class = pd.read_csv('data/test_si_8_class.csv')

# Load predictions
pred_ic50 = pd.read_csv('data/pred_ic50.csv')
pred_cc50 = pd.read_csv('data/pred_cc50.csv')
pred_si = pd.read_csv('data/pred_si.csv')
pred_ic50_class = pd.read_csv('data/pred_ic50_class.csv')
pred_cc50_class = pd.read_csv('data/pred_cc50_class.csv')
pred_si_class = pd.read_csv('data/pred_si_class.csv')
pred_si_8_class = pd.read_csv('data/pred_si_8_class.csv')

# Calculate residuals
residuals_ic50 = test_ic50['IC50, mM'] - pred_ic50['pred_ic50']
residuals_cc50 = test_cc50['CC50, mM'] - pred_cc50['pred_cc50']
residuals_si = test_si['SI'] - pred_si['pred_si']

# Plot residuals for each regression model
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(test_ic50['IC50, mM'], residuals_ic50, alpha=0.5)
plt.title('IC50 Residuals')
plt.xlabel('Actual IC50')
plt.ylabel('Residuals')

plt.subplot(1, 3, 2)
plt.scatter(test_cc50['CC50, mM'], residuals_cc50, alpha=0.5)
plt.title('CC50 Residuals')
plt.xlabel('Actual CC50')
plt.ylabel('Residuals')

plt.subplot(1, 3, 3)
plt.scatter(test_si['SI'], residuals_si, alpha=0.5)
plt.title('SI Residuals')
plt.xlabel('Actual SI')
plt.ylabel('Residuals')

plt.tight_layout()
plt.savefig('analysis/residuals/residuals_scatter.png')
plt.close()

# Analyze the distributions of residuals
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(residuals_ic50, kde=True)
plt.title('IC50 Residuals Distribution')

plt.subplot(1, 3, 2)
sns.histplot(residuals_cc50, kde=True)
plt.title('CC50 Residuals Distribution')

plt.subplot(1, 3, 3)
sns.histplot(residuals_si, kde=True)
plt.title('SI Residuals Distribution')

plt.tight_layout()
plt.savefig('analysis/residuals/residuals_distribution.png')
plt.close()

# Boxplots for error distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=residuals_ic50)
plt.title('IC50 Residuals Boxplot')

plt.subplot(1, 3, 2)
sns.boxplot(y=residuals_cc50)
plt.title('CC50 Residuals Boxplot')

plt.subplot(1, 3, 3)
sns.boxplot(y=residuals_si)
plt.title('SI Residuals Boxplot')

plt.tight_layout()
plt.savefig('analysis/residuals/residuals_boxplot.png')
plt.close()

# QQ-plots for residual analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
stats.probplot(residuals_ic50, dist="norm", plot=plt)
plt.title('IC50 Residuals QQ-plot')

plt.subplot(1, 3, 2)
stats.probplot(residuals_cc50, dist="norm", plot=plt)
plt.title('CC50 Residuals QQ-plot')

plt.subplot(1, 3, 3)
stats.probplot(residuals_si, dist="norm", plot=plt)
plt.title('SI Residuals QQ-plot')

plt.tight_layout()
plt.savefig('analysis/residuals/residuals_qqplot.png')
plt.close()

# For classification, show confusion matrices and analyze misclassified samples
cm_ic50 = confusion_matrix(test_ic50_class, pred_ic50_class)
cm_cc50 = confusion_matrix(test_cc50_class, pred_cc50_class)
cm_si = confusion_matrix(test_si_class, pred_si_class)
cm_si_8 = confusion_matrix(test_si_8_class, pred_si_8_class)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.heatmap(cm_ic50, annot=True, fmt='d', cmap='Blues')
plt.title('IC50 Confusion Matrix')

plt.subplot(2, 2, 2)
sns.heatmap(cm_cc50, annot=True, fmt='d', cmap='Blues')
plt.title('CC50 Confusion Matrix')

plt.subplot(2, 2, 3)
sns.heatmap(cm_si, annot=True, fmt='d', cmap='Blues')
plt.title('SI Confusion Matrix')

plt.subplot(2, 2, 4)
sns.heatmap(cm_si_8, annot=True, fmt='d', cmap='Blues')
plt.title('SI > 8 Confusion Matrix')

plt.tight_layout()
plt.savefig('analysis/residuals/confusion_matrices.png')
plt.close()

print("Residual analysis and error diagnostics completed.")