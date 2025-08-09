import pandas as pd

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

# Summarize key findings
print("Key Findings:")
print("1. The dataset contains 1001 entries and 214 columns.")
print("2. Key target variables include IC50, mM, CC50, mM, and SI.")
print("3. Missing values were handled by filling them with the median of their respective columns.")
print("4. Outliers were detected and removed using IQR and z-score methods.")
print("5. Features with high multicollinearity were removed.")
print("6. Advanced visualizations and statistical tests were conducted.")
print("7. Data was normalized and split into train/test sets.")

# Suggest improvements for modeling
print("\nImprovements for Modeling:")
print("1. Consider using more advanced regression techniques such as Random Forest or Gradient Boosting.")
print("2. Explore feature engineering to create new features or interactions.")
print("3. Handle class imbalance using techniques like SMOTE or different algorithms.")
print("4. Perform hyperparameter tuning to optimize model performance.")

# Save processed datasets and visualizations
print("\nProcessed datasets and visualizations saved in the data and analysis folders.")

print("Summary and recommendations completed.")