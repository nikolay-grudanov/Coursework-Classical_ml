import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load the feature-engineered data
data = pd.read_csv('data/feature_engineered_data.csv')

# Create binary targets for classification
data['IC50_above_median'] = (data['IC50, mM'] > data['IC50, mM'].median()).astype(int)
data['CC50_above_median'] = (data['CC50, mM'] > data['CC50, mM'].median()).astype(int)
data['SI_above_median'] = (data['SI'] > data['SI'].median()).astype(int)
data['SI_above_8'] = (data['SI'] > 8).astype(int)

# Define features and target variables
features = data.drop(columns=['IC50, mM', 'CC50, mM', 'SI', 'IC50_above_median', 'CC50_above_median', 'SI_above_median', 'SI_above_8'])

# Split the data into training and testing sets for classification
train_features, test_features, train_ic50_class, test_ic50_class = train_test_split(features, data['IC50_above_median'], test_size=0.2, random_state=42)
train_features_cc50, test_features_cc50, train_cc50_class, test_cc50_class = train_test_split(features, data['CC50_above_median'], test_size=0.2, random_state=42)
train_features_si, test_features_si, train_si_class, test_si_class = train_test_split(features, data['SI_above_median'], test_size=0.2, random_state=42)
train_features_si_8, test_features_si_8, train_si_8_class, test_si_8_class = train_test_split(features, data['SI_above_8'], test_size=0.2, random_state=42)

# Save the datasets for later use
train_features.to_csv('data/train_features.csv', index=False)
test_features.to_csv('data/test_features.csv', index=False)
train_ic50_class.to_csv('data/train_ic50_class.csv', index=False)
test_ic50_class.to_csv('data/test_ic50_class.csv', index=False)
train_cc50_class.to_csv('data/train_cc50_class.csv', index=False)
test_cc50_class.to_csv('data/test_cc50_class.csv', index=False)
train_si_class.to_csv('data/train_si_class.csv', index=False)
test_si_class.to_csv('data/test_si_class.csv', index=False)
train_si_8_class.to_csv('data/train_si_8_class.csv', index=False)
test_si_8_class.to_csv('data/test_si_8_class.csv', index=False)

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

# Save predictions
pd.DataFrame(y_pred_ic50_class, columns=['pred_ic50_class']).to_csv('data/pred_ic50_class.csv', index=False)
pd.DataFrame(y_pred_cc50_class, columns=['pred_cc50_class']).to_csv('data/pred_cc50_class.csv', index=False)
pd.DataFrame(y_pred_si_class, columns=['pred_si_class']).to_csv('data/pred_si_class.csv', index=False)
pd.DataFrame(y_pred_si_8_class, columns=['pred_si_8_class']).to_csv('data/pred_si_8_class.csv', index=False)

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

print("Classification modeling completed.")