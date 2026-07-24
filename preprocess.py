import pandas as pd
import numpy as np

# 1. Load the raw dataset
print("Loading dataset...")
df = pd.read_csv('heart_disease.csv')

# 2. Fill Missing Values (NaNs)
# Fill continuous numerical missing values with Median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with Mode (Most common value)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Convert Binary Text Columns into Numbers (0s and 1s)
binary_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Smoking': {'Yes': 1, 'No': 0},
    'Family Heart Disease': {'Yes': 1, 'No': 0},
    'Diabetes': {'Yes': 1, 'No': 0},
    'High Blood Pressure': {'Yes': 1, 'No': 0},
    'Low HDL Cholesterol': {'Yes': 1, 'No': 0},
    'High LDL Cholesterol': {'Yes': 1, 'No': 0},
    'Heart Disease Status': {'Yes': 1, 'No': 0}
}

for col, mapping in binary_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# 4. Convert Ordinal Categories (Low -> 0, Medium -> 1, High -> 2)
ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
ordinal_cols = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']

for col in ordinal_cols:
    if col in df.columns:
        df[col] = df[col].map(ordinal_mapping)

# 5. Verify no missing values remain
print("Missing values remaining:", df.isnull().sum().sum())

# 6. Save the clean dataset as a new CSV file
df.to_csv('cleaned_heart_disease.csv', index=False)
print("SUCCESS: 'cleaned_heart_disease.csv' has been generated!")