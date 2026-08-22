import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42

TARGET = 'Heart Disease Status'

ORDINAL_MAPPINGS = {
    'Exercise Habits': ['Low', 'Medium', 'High'],
    'Alcohol Consumption': ['None', 'Low', 'Medium', 'High'],
    'Stress Level': ['Low', 'Medium', 'High'],
    'Sugar Consumption': ['Low', 'Medium', 'High'],
}

BINARY_COLS = [
    'Gender',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
]

def main():
    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    df = pd.read_csv(ROOT / 'heart_disease.csv')
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---------------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------------
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed: {duplicate_count}")

    # ---------------------------------------------------------
    # TARGET & TRAIN/TEST SPLIT
    # ---------------------------------------------------------
    y = df[TARGET].map({'No': 0, 'Yes': 1})
    X = df.drop(columns=[TARGET]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # ---------------------------------------------------------
    # MISSING VALUES & IMPUTATION
    # ---------------------------------------------------------
    X_train['Alcohol Consumption'] = X_train['Alcohol Consumption'].fillna('None')
    X_test['Alcohol Consumption'] = X_test['Alcohol Consumption'].fillna('None')

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()

    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # ---------------------------------------------------------
    # ENCODING (ORDINAL & BINARY)
    # ---------------------------------------------------------
    for col, categories in ORDINAL_MAPPINGS.items():
        encoder = OrdinalEncoder(
            categories=[categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_train[col] = encoder.fit_transform(X_train[[col]]).ravel()
        X_test[col] = encoder.transform(X_test[[col]]).ravel()

    for col in BINARY_COLS:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    # ---------------------------------------------------------
    # SCALE NUMERICAL FEATURES ONLY
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ---------------------------------------------------------
    # SAVE PREPROCESSED DATA & SCALER
    # ---------------------------------------------------------
    X_train.to_csv(ROOT / 'X_train_preprocessed.csv', index=False)
    X_test.to_csv(ROOT / 'X_test_preprocessed.csv', index=False)
    y_train.to_csv(ROOT / 'y_train.csv', index=False)
    y_test.to_csv(ROOT / 'y_test.csv', index=False)

    joblib.dump(scaler, ROOT / 'shared_scaler.pkl')
    df.to_csv(ROOT / 'heart_disease_cleaned_full.csv', index=False)

    print("\n✅ PREPROCESSING COMPLETED")

if __name__ == '__main__':
    main()