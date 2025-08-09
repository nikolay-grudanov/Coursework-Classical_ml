import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import shap
import os

# Create analysis directory if it doesn't exist
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/feature_importance', exist_ok=True)

# Load the prepared datasets
train_features = pd.read_csv('data/train_features.csv')
test_features = pd.read_csv('data/test_features.csv')
train_ic50 = pd.read_csv('data/train_ic50.csv')
test_ic50 = pd.read_csv('data/test_ic50.csv')
train_cc50 = pd.read_csv('data/train_cc50.csv')
test_cc50 = pd.read_csv('data/test_cc50.csv')
train_si = pd.read_csv('data/train_si.csv')
test_si = pd.read_csv('data/test_si.csv')
train_ic50_class = pd.read_csv('data/train_ic50_class.csv')
test_ic50_class = pd.read_csv('data/test_ic50_class.csv')
train_cc50_class = pd.read_csv('data/train_cc50_class.csv')
test_cc50_class = pd.read_csv('data/test_cc50_class.csv')
train_si_class = pd.read_csv('data/train_si_class.csv')
test_si_class = pd.read_csv('data/test_si_class.csv')
train_si_8_class = pd.read_csv('data/train_si_8_class.csv')
test_si_8_class = pd.read_csv('data/test_si_8_class.csv')

# Train models for feature importance analysis
# Regression models
rf_ic50 = RandomForestRegressor(random_state=42)
rf_ic50.fit(train_features, train_ic50.values.ravel())

rf_cc50 = RandomForestRegressor(random_state=42)
rf_cc50.fit(train_features, train_cc50.values.ravel())

rf_si = RandomForestRegressor(random_state=42)
rf_si.fit(train_features, train_si.values.ravel())

# Classification models
rf_ic50_class = RandomForestClassifier(random_state=42)
rf_ic50_class.fit(train_features, train_ic50_class.values.ravel())

rf_cc50_class = RandomForestClassifier(random_state=42)
rf_cc50_class.fit(train_features, train_cc50_class.values.ravel())

rf_si_class = RandomForestClassifier(random_state=42)
rf_si_class.fit(train_features, train_si_class.values.ravel())

rf_si_8_class = RandomForestClassifier(random_state=42)
rf_si_8_class.fit(train_features, train_si_8_class.values.ravel())

# Plot feature importances for regression models
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.barplot(x=rf_ic50.feature_importances_, y=train_features.columns)
plt.title('IC50 Feature Importances')

plt.subplot(2, 2, 2)
sns.barplot(x=rf_cc50.feature_importances_, y=train_features.columns)
plt.title('CC50 Feature Importances')

plt.subplot(2, 2, 3)
sns.barplot(x=rf_si.feature_importances_, y=train_features.columns)
plt.title('SI Feature Importances')

plt.tight_layout()
plt.savefig('analysis/feature_importance/regression_feature_importance.png')
plt.close()

# Plot feature importances for classification models
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.barplot(x=rf_ic50_class.feature_importances_, y=train_features.columns)
plt.title('IC50 Classification Feature Importances')

plt.subplot(2, 2, 2)
sns.barplot(x=rf_cc50_class.feature_importances_, y=train_features.columns)
plt.title('CC50 Classification Feature Importances')

plt.subplot(2, 2, 3)
sns.barplot(x=rf_si_class.feature_importances_, y=train_features.columns)
plt.title('SI Classification Feature Importances')

plt.subplot(2, 2, 4)
sns.barplot(x=rf_si_8_class.feature_importances_, y=train_features.columns)
plt.title('SI > 8 Classification Feature Importances')

plt.tight_layout()
plt.savefig('analysis/feature_importance/classification_feature_importance.png')
plt.close()

# Use SHAP for advanced feature contribution analysis
# For IC50 regression
explainer = shap.Explainer(rf_ic50, train_features)
shap_values = explainer(test_features)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, test_features, plot_type="bar")
plt.title('IC50 SHAP Feature Importances')
plt.savefig('analysis/feature_importance/shap_ic50_importance.png')
plt.close()

# For CC50 regression
explainer = shap.Explainer(rf_cc50, train_features)
shap_values = explainer(test_features)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, test_features, plot_type="bar")
plt.title('CC50 SHAP Feature Importances')
plt.savefig('analysis/feature_importance/shap_cc50_importance.png')
plt.close()

# For SI regression
explainer = shap.Explainer(rf_si, train_features)
shap_values = explainer(test_features)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, test_features, plot_type="bar")
plt.title('SI SHAP Feature Importances')
plt.savefig('analysis/feature_importance/shap_si_importance.png')
plt.close()

print("Feature Importance and Model Interpretation completed.")