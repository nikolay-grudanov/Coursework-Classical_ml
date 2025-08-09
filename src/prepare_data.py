import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the reduced dataset
data = pd.read_csv('data/reduced_data.csv')

# Normalize/scale features
scaler = StandardScaler()
features = data.drop(columns=['IC50, mM', 'CC50, mM', 'SI', 'IC50_above_median', 'CC50_above_median', 'SI_above_median', 'SI_above_8'])
scaled_features = scaler.fit_transform(features)

# Create separate datasets for regression/classification
target_ic50 = data['IC50, mM']
target_cc50 = data['CC50, mM']
target_si = data['SI']
target_ic50_class = data['IC50_above_median']
target_cc50_class = data['CC50_above_median']
target_si_class = data['SI_above_median']
target_si_8_class = data['SI_above_8']

# Handle class imbalance using SMOTE if needed
smote = SMOTE(random_state=42)
scaled_features_res_ic50, target_ic50_class_res = smote.fit_resample(scaled_features, target_ic50_class)
scaled_features_res_cc50, target_cc50_class_res = smote.fit_resample(scaled_features, target_cc50_class)
scaled_features_res_si, target_si_class_res = smote.fit_resample(scaled_features, target_si_class)
scaled_features_res_si_8, target_si_8_class_res = smote.fit_resample(scaled_features, target_si_8_class)

# Split into train/test sets
train_features, test_features, train_ic50, test_ic50 = train_test_split(scaled_features, target_ic50, test_size=0.2, random_state=42)
_, _, train_cc50, test_cc50 = train_test_split(scaled_features, target_cc50, test_size=0.2, random_state=42)
_, _, train_si, test_si = train_test_split(scaled_features, target_si, test_size=0.2, random_state=42)
_, _, train_ic50_class, test_ic50_class = train_test_split(scaled_features_res_ic50, target_ic50_class_res, test_size=0.2, random_state=42)
_, _, train_cc50_class, test_cc50_class = train_test_split(scaled_features_res_cc50, target_cc50_class_res, test_size=0.2, random_state=42)
_, _, train_si_class, test_si_class = train_test_split(scaled_features_res_si, target_si_class_res, test_size=0.2, random_state=42)
_, _, train_si_8_class, test_si_8_class = train_test_split(scaled_features_res_si_8, target_si_8_class_res, test_size=0.2, random_state=42)

# Save the prepared datasets to new CSV files
pd.DataFrame(train_features).to_csv('data/train_features.csv', index=False)
pd.DataFrame(test_features).to_csv('data/test_features.csv', index=False)
pd.DataFrame(train_ic50).to_csv('data/train_ic50.csv', index=False)
pd.DataFrame(test_ic50).to_csv('data/test_ic50.csv', index=False)
pd.DataFrame(train_cc50).to_csv('data/train_cc50.csv', index=False)
pd.DataFrame(test_cc50).to_csv('data/test_cc50.csv', index=False)
pd.DataFrame(train_si).to_csv('data/train_si.csv', index=False)
pd.DataFrame(test_si).to_csv('data/test_si.csv', index=False)
pd.DataFrame(train_ic50_class).to_csv('data/train_ic50_class.csv', index=False)
pd.DataFrame(test_ic50_class).to_csv('data/test_ic50_class.csv', index=False)
pd.DataFrame(train_cc50_class).to_csv('data/train_cc50_class.csv', index=False)
pd.DataFrame(test_cc50_class).to_csv('data/test_cc50_class.csv', index=False)
pd.DataFrame(train_si_class).to_csv('data/train_si_class.csv', index=False)
pd.DataFrame(test_si_class).to_csv('data/test_si_class.csv', index=False)
pd.DataFrame(train_si_8_class).to_csv('data/train_si_8_class.csv', index=False)
pd.DataFrame(test_si_8_class).to_csv('data/test_si_8_class.csv', index=False)

print("Data preparation for ML completed.")