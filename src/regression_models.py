import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load the feature-engineered data
data = pd.read_csv('data/feature_engineered_data.csv')

# Define features and target variables
features = data.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])
target_ic50 = data['IC50, mM']
target_cc50 = data['CC50, mM']
target_si = data['SI']

# Split the data into training and testing sets
train_features, test_features, train_ic50, test_ic50 = train_test_split(features, target_ic50, test_size=0.2, random_state=42)
train_features_cc50, test_features_cc50, train_cc50, test_cc50 = train_test_split(features, target_cc50, test_size=0.2, random_state=42)
train_features_si, test_features_si, train_si, test_si = train_test_split(features, target_si, test_size=0.2, random_state=42)

# Save the datasets for later use
train_features.to_csv('data/train_features.csv', index=False)
test_features.to_csv('data/test_features.csv', index=False)
train_ic50.to_csv('data/train_ic50.csv', index=False)
test_ic50.to_csv('data/test_ic50.csv', index=False)
train_cc50.to_csv('data/train_cc50.csv', index=False)
test_cc50.to_csv('data/test_cc50.csv', index=False)
train_si.to_csv('data/train_si.csv', index=False)
test_si.to_csv('data/test_si.csv', index=False)

# Initialize and train the regression models
model_ic50 = LinearRegression()
model_cc50 = LinearRegression()
model_si = LinearRegression()

model_ic50.fit(train_features, train_ic50)
model_cc50.fit(train_features_cc50, train_cc50)
model_si.fit(train_features_si, train_si)

# Predict on the test set
y_pred_ic50 = model_ic50.predict(test_features)
y_pred_cc50 = model_cc50.predict(test_features_cc50)
y_pred_si = model_si.predict(test_features_si)

# Save predictions
pd.DataFrame(y_pred_ic50, columns=['pred_ic50']).to_csv('data/pred_ic50.csv', index=False)
pd.DataFrame(y_pred_cc50, columns=['pred_cc50']).to_csv('data/pred_cc50.csv', index=False)
pd.DataFrame(y_pred_si, columns=['pred_si']).to_csv('data/pred_si.csv', index=False)

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

print("Regression modeling completed.")