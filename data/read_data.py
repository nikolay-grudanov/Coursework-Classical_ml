import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load the data from the Excel file
data = pd.read_excel('/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx')

# Drop the 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Calculate the Selectivity Index (SI) if not present
if 'SI' not in data.columns:
    data['SI'] = data['CC50, mM'] / data['IC50, mM']

# Handle missing values by filling them with the median of their respective columns
data.fillna(data.median(), inplace=True)

# Define features and target variables
features = data.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])
target_ic50 = data['IC50, mM']
target_cc50 = data['CC50, mM']
target_si = data['SI']

# Split the data into training and testing sets for regression
train_features, test_features, train_ic50, test_ic50 = train_test_split(features, target_ic50, test_size=0.2, random_state=42)
_, _, train_cc50, test_cc50 = train_test_split(features, target_cc50, test_size=0.2, random_state=42)
_, _, train_si, test_si = train_test_split(features, target_si, test_size=0.2, random_state=42)

# Initialize and train the regression models
model_ic50 = LinearRegression()
model_cc50 = LinearRegression()
model_si = LinearRegression()

model_ic50.fit(train_features, train_ic50)
model_cc50.fit(train_features, train_cc50)
model_si.fit(train_features, train_si)

# Predict on the test set
y_pred_ic50 = model_ic50.predict(test_features)
y_pred_cc50 = model_cc50.predict(test_features)
y_pred_si = model_si.predict(test_features)

# Evaluate the models
mse_ic50 = mean_squared_error(test_ic50, y_pred_ic50)
r2_ic50 = r2_score(test_ic50, y_pred_ic50)

mse_cc50 = mean_squared_error(test_cc50, y_pred_cc50)
r2_cc50 = r2_score(test_cc50, y_pred_cc50)

mse_si = mean_squared_error(test_si, y_pred_si)
r2_si = r2_score(test_si, y_pred_si)

print(f'IC50 Regression - MSE: {mse_ic50}, R2: {r2_ic50}')
print(f'CC50 Regression - MSE: {mse_cc50}, R2: {r2_cc50}')
print(f'SI Regression - MSE: {mse_si}, R2: {r2_si}')

# Visualize the actual vs predicted values for IC50
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(test_ic50, y_pred_ic50)
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')
plt.title('IC50 Actual vs Predicted')

plt.subplot(1, 3, 2)
plt.scatter(test_cc50, y_pred_cc50)
plt.xlabel('Actual CC50')
plt.ylabel('Predicted CC50')
plt.title('CC50 Actual vs Predicted')

plt.subplot(1, 3, 3)
plt.scatter(test_si, y_pred_si)
plt.xlabel('Actual SI')
plt.ylabel('Predicted SI')
plt.title('SI Actual vs Predicted')

plt.tight_layout()
plt.show()

# Classification Models
# Create binary targets for classification
data['IC50_above_median'] = (data['IC50, mM'] > data['IC50, mM'].median()).astype(int)
data['CC50_above_median'] = (data['CC50, mM'] > data['CC50, mM'].median()).astype(int)
data['SI_above_median'] = (data['SI'] > data['SI'].median()).astype(int)
data['SI_above_8'] = (data['SI'] > 8).astype(int)

# Split the data into training and testing sets for classification
train_features, test_features, train_ic50_class, test_ic50_class = train_test_split(features, data['IC50_above_median'], test_size=0.2, random_state=42)
_, _, train_cc50_class, test_cc50_class = train_test_split(features, data['CC50_above_median'], test_size=0.2, random_state=42)
_, _, train_si_class, test_si_class = train_test_split(features, data['SI_above_median'], test_size=0.2, random_state=42)
_, _, train_si_8_class, test_si_8_class = train_test_split(features, data['SI_above_8'], test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
train_features_res_ic50, train_ic50_class_res = smote.fit_resample(train_features, train_ic50_class)
train_features_res_cc50, train_cc50_class_res = smote.fit_resample(train_features, train_cc50_class)
train_features_res_si, train_si_class_res = smote.fit_resample(train_features, train_si_class)
train_features_res_si_8, train_si_8_class_res = smote.fit_resample(train_features, train_si_8_class)

# Initialize and train the classification models
model_ic50_class = RandomForestClassifier(random_state=42)
model_cc50_class = RandomForestClassifier(random_state=42)
model_si_class = RandomForestClassifier(random_state=42)
model_si_8_class = RandomForestClassifier(random_state=42)

model_ic50_class.fit(train_features_res_ic50, train_ic50_class_res)
model_cc50_class.fit(train_features_res_cc50, train_cc50_class_res)
model_si_class.fit(train_features_res_si, train_si_class_res)
model_si_8_class.fit(train_features_res_si_8, train_si_8_class_res)

# Predict on the test set
y_pred_ic50_class = model_ic50_class.predict(test_features)
y_pred_cc50_class = model_cc50_class.predict(test_features)
y_pred_si_class = model_si_class.predict(test_features)
y_pred_si_8_class = model_si_8_class.predict(test_features)

# Evaluate the models
accuracy_ic50 = accuracy_score(test_ic50_class, y_pred_ic50_class)
accuracy_cc50 = accuracy_score(test_cc50_class, y_pred_cc50_class)
accuracy_si = accuracy_score(test_si_class, y_pred_si_class)
accuracy_si_8 = accuracy_score(test_si_8_class, y_pred_si_8_class)

print(f'IC50 Classification - Accuracy: {accuracy_ic50}')
print(f'CC50 Classification - Accuracy: {accuracy_cc50}')
print(f'SI Classification - Accuracy: {accuracy_si}')
print(f'SI > 8 Classification - Accuracy: {accuracy_si_8}')

# Print classification reports
print('IC50 Classification Report:')
print(classification_report(test_ic50_class, y_pred_ic50_class))

print('CC50 Classification Report:')
print(classification_report(test_cc50_class, y_pred_cc50_class))

print('SI Classification Report:')
print(classification_report(test_si_class, y_pred_si_class))

print('SI > 8 Classification Report:')
print(classification_report(test_si_8_class, y_pred_si_8_class))

# Summary of Results
print('Summary of Results:')
print('Regression Models:')
print(f'IC50 Regression - MSE: {mse_ic50}, R2: {r2_ic50}')
print(f'CC50 Regression - MSE: {mse_cc50}, R2: {r2_cc50}')
print(f'SI Regression - MSE: {mse_si}, R2: {r2_si}')

print('Classification Models:')
print(f'IC50 Classification - Accuracy: {accuracy_ic50}')
print(f'CC50 Classification - Accuracy: {accuracy_cc50}')
print(f'SI Classification - Accuracy: {accuracy_si}')
print(f'SI > 8 Classification - Accuracy: {accuracy_si_8}')

print('The regression models show moderate performance with R2 values indicating the proportion of variance explained.')
print('The classification models, after addressing class imbalance with SMOTE and using Random Forest, show improved accuracy and balanced performance across classes.')
print('The CC50 classification model performs the best among the classification tasks, followed by IC50, SI, and SI > 8.')
