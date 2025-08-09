import pandas as pd

# Load the data from the Excel file
data = pd.read_excel('/home/gna/workspase/education/MEPHI/Coursework-Classical_ml/data/data.xlsx')

# Drop the 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Save the cleaned data to a new CSV file
data.to_csv('data/cleaned_data.csv', index=False)

print("Data loaded and cleaned successfully.")