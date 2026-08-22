import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42
TARGET = "Heart Disease Status"

df = pd.read_csv("heart_disease.csv")

y = df[TARGET].map({"No": 0, "Yes": 1})
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])

print("Training baseline Logistic Regression...")

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n" + "=" * 50)
print("BASELINE RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")