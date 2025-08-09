import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create analysis directory if it doesn't exist
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/subgroup', exist_ok=True)

# Load the prepared datasets
test_features = pd.read_csv('data/test_features.csv')
test_ic50 = pd.read_csv('data/test_ic50.csv')
test_cc50 = pd.read_csv('data/test_cc50.csv')
test_si = pd.read_csv('data/test_si.csv')

# Load predictions
pred_ic50 = pd.read_csv('data/pred_ic50.csv')
pred_cc50 = pd.read_csv('data/pred_cc50.csv')
pred_si = pd.read_csv('data/pred_si.csv')

# Calculate residuals
residuals_ic50 = test_ic50['IC50, mM'] - pred_ic50['pred_ic50']
residuals_cc50 = test_cc50['CC50, mM'] - pred_cc50['pred_cc50']
residuals_si = test_si['SI'] - pred_si['pred_si']

# Investigate how model quality and errors differ across key data subgroups
# Example: Plot actual vs. predicted for subgroups based on IC50 quartiles
quartiles = np.percentile(test_ic50['IC50, mM'], [25, 50, 75])

# Create subgroups
low_ic50 = test_ic50[test_ic50['IC50, mM'] <= quartiles[0]]
mid_low_ic50 = test_ic50[(test_ic50['IC50, mM'] > quartiles[0]) & (test_ic50['IC50, mM'] <= quartiles[1])]
mid_high_ic50 = test_ic50[(test_ic50['IC50, mM'] > quartiles[1]) & (test_ic50['IC50, mM'] <= quartiles[2])]
high_ic50 = test_ic50[test_ic50['IC50, mM'] > quartiles[2]]

# Get corresponding predictions
low_pred = pred_ic50.loc[low_ic50.index]
mid_low_pred = pred_ic50.loc[mid_low_ic50.index]
mid_high_pred = pred_ic50.loc[mid_high_ic50.index]
high_pred = pred_ic50.loc[high_ic50.index]

# Plot actual vs predicted for each subgroup
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(low_ic50['IC50, mM'], low_pred['pred_ic50'], alpha=0.5)
plt.plot([0, max(low_ic50['IC50, mM'])], [0, max(low_ic50['IC50, mM'])], 'r--')
plt.title('Low IC50: Actual vs Predicted')
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')

plt.subplot(2, 2, 2)
plt.scatter(mid_low_ic50['IC50, mM'], mid_low_pred['pred_ic50'], alpha=0.5)
plt.plot([0, max(mid_low_ic50['IC50, mM'])], [0, max(mid_low_ic50['IC50, mM'])], 'r--')
plt.title('Mid-Low IC50: Actual vs Predicted')
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')

plt.subplot(2, 2, 3)
plt.scatter(mid_high_ic50['IC50, mM'], mid_high_pred['pred_ic50'], alpha=0.5)
plt.plot([0, max(mid_high_ic50['IC50, mM'])], [0, max(mid_high_ic50['IC50, mM'])], 'r--')
plt.title('Mid-High IC50: Actual vs Predicted')
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')

plt.subplot(2, 2, 4)
plt.scatter(high_ic50['IC50, mM'], high_pred['pred_ic50'], alpha=0.5)
plt.plot([0, max(high_ic50['IC50, mM'])], [0, max(high_ic50['IC50, mM'])], 'r--')
plt.title('High IC50: Actual vs Predicted')
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')

plt.tight_layout()
plt.savefig('analysis/subgroup/ic50_subgroup_analysis.png')
plt.close()

# Calculate and plot residuals for each subgroup
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(low_ic50['IC50, mM'] - low_pred['pred_ic50'], kde=True)
plt.title('Low IC50 Residuals')

plt.subplot(2, 2, 2)
sns.histplot(mid_low_ic50['IC50, mM'] - mid_low_pred['pred_ic50'], kde=True)
plt.title('Mid-Low IC50 Residuals')

plt.subplot(2, 2, 3)
sns.histplot(mid_high_ic50['IC50, mM'] - mid_high_pred['pred_ic50'], kde=True)
plt.title('Mid-High IC50 Residuals')

plt.subplot(2, 2, 4)
sns.histplot(high_ic50['IC50, mM'] - high_pred['pred_ic50'], kde=True)
plt.title('High IC50 Residuals')

plt.tight_layout()
plt.savefig('analysis/subgroup/ic50_residuals_by_subgroup.png')
plt.close()

# Calculate and print error metrics for each subgroup
metrics = []
for name, subgroup, pred in [
    ('Low', low_ic50, low_pred),
    ('Mid-Low', mid_low_ic50, mid_low_pred),
    ('Mid-High', mid_high_ic50, mid_high_pred),
    ('High', high_ic50, high_pred)
]:
    residuals = subgroup['IC50, mM'] - pred['pred_ic50']
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    metrics.append({'Subgroup': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})

metrics_df = pd.DataFrame(metrics)
print("IC50 Subgroup Error Metrics:")
print(metrics_df)

print("Data subgroup analysis completed.")