import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  HEART DISEASE DATASET - PREPROCESSING PIPELINE")
print("=" * 60)

# ---------------------------------------------------------
# STEP 1: Load Dataset
# ---------------------------------------------------------
raw_filename = 'heart_disease.csv'
df = pd.read_csv(raw_filename)

print(f"\n[INFO] Loaded '{raw_filename}' successfully!")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# STEP 2: Exploratory Groupby Analysis (Practical 3 & 4 Style)
# ---------------------------------------------------------
print("\n" + "-" * 50)
print("1. Practical 3 & 4 Groupby Insights")
print("-" * 50)

# Example 1: Average Health Vitals grouped by Heart Disease Status
print("\n>>> Mean Risk Factors Grouped by Heart Disease Status:")
vital_groupby = df.groupby('Heart Disease Status')[['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure']].mean()
print(vital_groupby.round(2))

# Example 2: Count of patients grouped by Gender and Heart Disease Status
print("\n>>> Patient Counts Grouped by Gender and Heart Disease Status:")
gender_groupby = df.groupby(['Gender', 'Heart Disease Status']).size().unstack()
print(gender_groupby)


# ---------------------------------------------------------
# STEP 3: Handle Missing Values using groupby()
# ---------------------------------------------------------
print("\n" + "-" * 50)
print("2. Handling Missing Values via groupby Imputation")
print("-" * 50)

print(f"Total Missing Values Before Imputation: {df.isnull().sum().sum()}")

# Identify numerical vs categorical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = [col for col in df.columns if col not in num_cols and col != 'Heart Disease Status']

# Impute numerical features using Median based on (Gender + Heart Disease Status) group
for col in num_cols:
    df[col] = df.groupby(['Gender', 'Heart Disease Status'])[col].transform(lambda x: x.fillna(x.median()))
    # Backup fallback if group had all NaNs
    df[col] = df[col].fillna(df[col].median())

# Impute categorical features using Mode based on Heart Disease Status group
for col in cat_cols:
    df[col] = df.groupby('Heart Disease Status')[col].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
    )
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Total Missing Values After Imputation: {df.isnull().sum().sum()}")


# ---------------------------------------------------------
# STEP 4: Feature Encoding (Categorical to Numerical)
# ---------------------------------------------------------
print("\n" + "-" * 50)
print("3. Categorical & Ordinal Feature Encoding")
print("-" * 50)

# Binary Mappings (No -> 0, Yes -> 1)
binary_map = {'No': 0, 'Yes': 1}
binary_cols = [
    'Smoking', 'Family Heart Disease', 'Diabetes', 
    'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol'
]

for col in binary_cols:
    df[col] = df[col].map(binary_map)

# Specific Mappings
gender_map = {'Female': 0, 'Male': 1}
target_map = {'No': 0, 'Yes': 1}

df['Gender'] = df['Gender'].map(gender_map)
df['Heart Disease Status'] = df['Heart Disease Status'].map(target_map)

# Ordinal Multi-level Mappings (Low -> 0, Medium -> 1, High -> 2)
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
ordinal_cols = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']

for col in ordinal_cols:
    df[col] = df[col].map(ordinal_map)

print("[INFO] Successfully encoded all categorical and ordinal variables into numerical integers.")


# ---------------------------------------------------------
# STEP 5: Export Unscaled & Scaled Clean Datasets
# ---------------------------------------------------------
print("\n" + "-" * 50)
print("4. Saving Cleaned Output Files")
print("-" * 50)

# Save the unscaled cleaned CSV for EDA and tree models (Decision Tree, Random Forest)
unscaled_filename = 'cleaned_heart_disease.csv'
df.to_csv(unscaled_filename, index=False)
print(f"✅ Saved unscaled clean dataset as: '{unscaled_filename}'")

# Create a scaled copy for distance-based models (Logistic Regression, KNN)
df_scaled = df.copy()
scaler = StandardScaler()

features_to_scale = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                     'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
                     'CRP Level', 'Homocysteine Level']

df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

scaled_filename = 'scaled_cleaned_heart_disease.csv'
df_scaled.to_csv(scaled_filename, index=False)
print(f"✅ Saved scaled clean dataset as: '{scaled_filename}'")

print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)