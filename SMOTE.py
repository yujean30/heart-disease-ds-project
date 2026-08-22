import pandas as pd
from imblearn.over_sampling import SMOTENC

# ---------------------------------------------------------
# 1. Load the Files Produced by preprocess.py
# ---------------------------------------------------------
# X_train_preprocessed.csv already has: numeric columns scaled (StandardScaler),
# categorical columns encoded as plain integers (NOT scaled). We only apply
# SMOTE here -- no re-imputing, re-encoding, or re-splitting.
X_train = pd.read_csv('X_train_preprocessed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

print(f"Training set before SMOTE: {X_train.shape[0]} rows")
print("Class balance before SMOTE:")
print(pd.Series(y_train).value_counts())

# ---------------------------------------------------------
# 2. Identify Categorical Column Positions
# ---------------------------------------------------------
# Same columns preprocess.py encoded with OrdinalEncoder / LabelEncoder.
# These must stay untouched by SMOTE's interpolation -- SMOTENC handles that
# by taking a majority vote among nearest neighbors for these columns instead
# of interpolating them, so no fractional/invalid categories get created.
ordinal_cols = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
binary_cols = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
               'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']
categorical_cols = binary_cols + ordinal_cols

categorical_indices = [X_train.columns.get_loc(c) for c in categorical_cols]

# Sanity check: confirm these columns really are still plain integers
# (i.e. preprocess.py did not scale them).
non_integer = X_train[categorical_cols].apply(lambda col: (col % 1 != 0).sum())
if non_integer.sum() > 0:
    print("\nWARNING: some categorical columns contain non-integer values --")
    print("this usually means they were scaled. Do not proceed until this is")
    print("resolved, or SMOTENC will still be pointed at the wrong columns.")
    print(non_integer[non_integer > 0])

# ---------------------------------------------------------
# 3. Apply SMOTENC -- Training Set Only
# ---------------------------------------------------------
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_balanced, y_train_balanced = smote_nc.fit_resample(X_train, y_train)

print("\nClass balance after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())
print(f"Training set after SMOTE: {X_train_balanced.shape[0]} rows")

# ---------------------------------------------------------
# 4. Save -- Overwrites the Training Files Only
# ---------------------------------------------------------
# X_test_preprocessed.csv and y_test.csv from preprocess.py are NOT touched --
# they stay as the original, real-world-imbalanced test set.
X_train_balanced.to_csv('X_train_preprocessed.csv', index=False)
pd.Series(y_train_balanced, name='Heart Disease Status').to_csv('y_train.csv', index=False)

print("\nSaved (overwritten with balanced versions):")
print("- X_train_preprocessed.csv")
print("- y_train.csv")
print("\nUnchanged (test set left as-is):")
print("- X_test_preprocessed.csv")
print("- y_test.csv")