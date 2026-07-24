import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv('heart_disease.csv')


print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nInitial Data Types and Missing Value Summary:")
missing_summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Count': df.isnull().sum(),# to scan for empty values in the dataset
    'Missing Percentage (%)': (df.isnull().sum() / len(df)) * 100 
})
print(missing_summary)
# missing summary for data quality assessment

#for removing duplicates, we will check for duplicate rows and remove them if any exist
print("\n --- Removing Duplicates ---")
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

# if got duplicate rows, we will remove them and reset the index starting from 0
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed successfully.")


# 0 mean No Heart Disease, 1 means Yes Heart Disease
y = df['Heart Disease Status'].map({'No': 0, 'Yes': 1})
# delete the heart disease status column from the feature set to avoid data leakage
X = df.drop(columns=['Heart Disease Status'])


print("\n--- Handling Missing Values ---")

# replace 'NaN' to 'None' in Alcohol Consumption column 
X['Alcohol Consumption'] = X['Alcohol Consumption'].fillna('None')

# make two lists 
# pick all columns that are numbers
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# pick all columns that are text or object 
cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()

# create a tool to fill missing numbers using the median
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# fill missing values with the most frequent value (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

print("Missing values after imputation:", X.isnull().sum().sum())


print("\n--- Handling Categorical Features ---")

# encoding : make the Alpha to Numeric conversion 
# low = 0, medium = 1, high = 2
ordinal_mappings = {
    'Exercise Habits': ['Low', 'Medium', 'High'],
    'Alcohol Consumption': ['None', 'Low', 'Medium', 'High'],
    'Stress Level': ['Low', 'Medium', 'High'],
    'Sugar Consumption': ['Low', 'Medium', 'High']
}

for col, categories in ordinal_mappings.items():
    oe = OrdinalEncoder(categories=[categories])
    X[col] = oe.fit_transform(X[[col]])

# define the binary categorical features that will be encoded using LabelEncoder
binary_cols = [
    'Gender', 'Smoking', 'Family Heart Disease', 'Diabetes', 
    'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol'
]
#gender: female = 0, male = 1
#smoking: no = 0, yes = 1
#family heart disease: no = 0, yes = 1
#diabetes: no = 0, yes = 1
#high blood pressure: no = 0, yes = 1
#low hdl cholesterol: no = 0, yes = 1
#high ldl cholesterol: no = 0, yes = 1

# encode binary categorical features using LabelEncoder
# only 0 and 1 values will be assigned to the two categories
for col in binary_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("Sample encoded feature matrix:")
print(X.head(5))


print("\n--- Performing Train-Test Split ---")

# train test split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")
print("\nTarget Class Distribution (Train):")
print(y_train.value_counts(normalize=True))

#standardize the numerical features to have a mean of 0 and a standard deviation of 1
print("\n--- Performing Feature Scaling  ---")

scaler = StandardScaler()

# make copies of the training and testing sets to avoid modifying the original data
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()


# calculate the mean and standard deviation and scale the training set
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
# we use transform to prevent data leakage 
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print("Numerical features successfully scaled using StandardScaler.")

# save preprocessed datasets for modeling stage
# index=False to avoid saving the index column in the CSV files
# for training
X_train_scaled.to_csv('X_train_preprocessed.csv', index=False)
# for testing
X_test_scaled.to_csv('X_test_preprocessed.csv', index=False)
# for target variable = got answer for training dataset
y_train.to_csv('y_train.csv', index=False)
# for target variable = got answer for testing dataset
y_test.to_csv('y_test.csv', index=False)
# save the full preprocessed dataset for future reference 
df.to_csv('heart_disease_cleaned_full.csv', index=False)

print("\nPreprocessing Completed !")