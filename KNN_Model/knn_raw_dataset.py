# ============================================================
# K-Nearest Neighbors (KNN) - Raw Heart Disease Dataset
# Leakage-safe diagnostic / final candidate version
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Change this only if your CSV has a different filename.
DATA_PATH = "heart_disease_cleaned_full.csv"

OUTPUT_DIR = "KNN_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("KNN - RAW HEART DISEASE DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Heart Disease Status"].value_counts())


# ============================================================
# 3. TARGET / FEATURES
# ============================================================

X = df.drop(columns=["Heart Disease Status"]).copy()
y = df["Heart Disease Status"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Target contains unexpected values.")

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 5. PREPROCESSING
#    Numeric -> median imputation + StandardScaler
#    Categorical -> most-frequent imputation + OneHotEncoder
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 6. KNN + 5-FOLD CROSS-VALIDATION
#    Test set is NOT used to select the hyperparameters.
# ============================================================

knn = KNeighborsClassifier()

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", knn)
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 18, 21, 25, 31],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Accuracy is used here only for selecting the KNN configuration.
# Final evaluation is performed separately on the untouched test set.
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    verbose=1
)

print("\n" + "=" * 70)
print("5-FOLD KNN HYPERPARAMETER TUNING")
print("=" * 70)

grid.fit(X_train, y_train)

print("\nBest parameters:")
print(grid.best_params_)

print(f"Best 5-Fold CV Accuracy: {grid.best_score_:.4f}")


# ============================================================
# 7. FINAL TEST-SET EVALUATION
# ============================================================

best_knn = grid.best_estimator_

y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 70)
print("FINAL KNN TEST-SET PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# 8. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": ["KNN"],
    "Best K": [grid.best_params_["knn__n_neighbors"]],
    "Weights": [grid.best_params_["knn__weights"]],
    "Metric": [grid.best_params_["knn__metric"]],
    "CV Accuracy": [grid.best_score_],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "ROC AUC": [roc_auc]
})

metrics_path = os.path.join(
    OUTPUT_DIR,
    "knn_raw_dataset_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False
)

plt.title(
    f"KNN Confusion Matrix "
    f"(K={grid.best_params_['knn__n_neighbors']})"
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "knn_raw_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 10. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC = {roc_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN ROC Curve")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "knn_raw_roc_curve.png"
)

plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "knn_raw_best_model.pkl"
)

joblib.dump(best_knn, model_path)

print("\nGenerated files:")
print(" -", metrics_path)
print(" -", cm_path)
print(" -", roc_path)
print(" -", model_path)

print("\nKNN run completed.")
