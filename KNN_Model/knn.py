# ============================================================
# K-Nearest Neighbors (KNN) Model
# Heart Disease Classification
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTENC


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 1.0
SMOTE_K_NEIGHBORS = 5

# Final KNN value selected from previous tuning
FINAL_K = 18

# Categorical columns used by SMOTENC
CATEGORICAL_COLS = [
    "Exercise Habits",
    "Alcohol Consumption",
    "Stress Level",
    "Sugar Consumption",
    "Gender",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
]


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_X_PATH = os.path.join(
    BASE_DIR, "X_train_preprocessed.csv"
)

TEST_X_PATH = os.path.join(
    BASE_DIR, "X_test_preprocessed.csv"
)

TRAIN_Y_PATH = os.path.join(
    BASE_DIR, "y_train.csv"
)

TEST_Y_PATH = os.path.join(
    BASE_DIR, "y_test.csv"
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_knn_model.pkl"
)

# Also save a joblib version for UI compatibility
MODEL_JOBLIB_PATH = os.path.join(
    OUTPUT_DIR, "knn_model.joblib"
)

SCALER_PATH = os.path.join(
    OUTPUT_DIR, "scaler.pkl"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR, "knn_baseline_metrics.csv"
)

CM_PATH = os.path.join(
    OUTPUT_DIR, "knn_confusion_matrix.png"
)

ROC_PATH = os.path.join(
    OUTPUT_DIR, "knn_roc_curve.png"
)


# ============================================================
# 3. LOAD PREPROCESSED DATA
# ============================================================

print("=" * 60)
print("KNN HEART DISEASE MODEL")
print("=" * 60)

print("\nLoading preprocessed datasets...")

X_train = pd.read_csv(TRAIN_X_PATH)
X_test = pd.read_csv(TEST_X_PATH)

y_train = pd.read_csv(TRAIN_Y_PATH)
y_test = pd.read_csv(TEST_Y_PATH)

# Convert target DataFrames to 1-dimensional arrays
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

print("\nOriginal data:")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 4. APPLY SMOTENC
# ============================================================

print("\nApplying SMOTENC...")

# Check that all categorical columns exist
missing_categorical = [
    col for col in CATEGORICAL_COLS
    if col not in X_train.columns
]

if missing_categorical:
    print("\nWARNING: The following categorical columns were")
    print("not found in X_train:")
    print(missing_categorical)

    # Use only columns that actually exist
    categorical_columns_used = [
        col for col in CATEGORICAL_COLS
        if col in X_train.columns
    ]
else:
    categorical_columns_used = CATEGORICAL_COLS


smote = SMOTENC(
    categorical_features=categorical_columns_used,
    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    k_neighbors=SMOTE_K_NEIGHBORS,
    random_state=RANDOM_STATE,
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTENC:")
print("X_train shape:", X_train_smote.shape)
print("y_train shape:", y_train_smote.shape)

print("\nClass distribution after SMOTENC:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# 5. STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_smote
)

X_test_scaled = scaler.transform(
    X_test
)

print("StandardScaler applied successfully.")


# ============================================================
# 6. KNN MODEL TUNING
# ============================================================

print("\n" + "=" * 60)
print("KNN PARAMETER TUNING")
print("=" * 60)

k_results = []

for k in range(1, 21):

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(
        X_train_scaled,
        y_train_smote
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    k_results.append({
        "K": k,
        "Accuracy": accuracy
    })

    print(
        f"K = {k:2d}, "
        f"Accuracy = {accuracy * 100:.2f}%"
    )


k_results_df = pd.DataFrame(k_results)


# ============================================================
# 7. TRAIN FINAL KNN MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"FINAL KNN MODEL: K = {FINAL_K}")
print("=" * 60)

knn = KNeighborsClassifier(
    n_neighbors=FINAL_K,
    metric="euclidean"
)

knn.fit(
    X_train_scaled,
    y_train_smote
)

y_pred = knn.predict(
    X_test_scaled
)

y_prob = knn.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nFinal KNN Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 9. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": ["KNN"],
    "Best K": [FINAL_K],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "ROC AUC": [roc_auc],
})

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

print("\nMetrics saved to:")
print(METRICS_PATH)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False
)

plt.title(
    f"KNN Confusion Matrix (K={FINAL_K})"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved.")


# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

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

plt.title(
    f"KNN ROC Curve (K={FINAL_K})"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved.")


# ============================================================
# 12. SAVE FINAL MODEL AND SCALER
# ============================================================

joblib.dump(
    knn,
    MODEL_PATH
)

# Save an additional joblib copy for UI compatibility
joblib.dump(
    knn,
    MODEL_JOBLIB_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\nModel files saved:")
print(" -", MODEL_PATH)
print(" -", MODEL_JOBLIB_PATH)
print(" -", SCALER_PATH)


# ============================================================
# 13. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KNN MODEL COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Final K: {FINAL_K}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nGenerated files:")

print("1. best_knn_model.pkl")
print("2. knn_model.joblib")
print("3. scaler.pkl")
print("4. knn_baseline_metrics.csv")
print("5. knn_confusion_matrix.png")
print("6. knn_roc_curve.png")

print("\nAll KNN outputs are ready for GitHub/UI integration.")