import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# 0. Suppress Parallelization Warnings
# ---------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# ---------------------------------------------------------
# 1. Create Output Folder
# ---------------------------------------------------------
output_dir = 'Random_Forest_Model'
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Load Preprocessed Data
# ---------------------------------------------------------
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# ---------------------------------------------------------
# 3. Fit & Save Scaler (numeric columns only)
# ---------------------------------------------------------
num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
            'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']

scaler = StandardScaler()
scaler.fit(X_train[num_cols])
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# ---------------------------------------------------------
# 4. Build Pipeline: SMOTE + Random Forest
# ---------------------------------------------------------
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# ---------------------------------------------------------
# 5. Hyperparameter Search
# ---------------------------------------------------------
param_dist = {
    "rf__n_estimators": [100, 200, 300, 500],
    "rf__max_depth": [None, 10, 20, 30, 50],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__class_weight": ["balanced", None]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv_strategy,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rand_search.fit(X_train, y_train)
best_model = rand_search.best_estimator_

# ---------------------------------------------------------
# 6. Save Trained Model (unchanged)
# ---------------------------------------------------------
joblib.dump(best_model, os.path.join(output_dir, 'random_forest_tuned.joblib'))

# ---------------------------------------------------------
# 7. Model Evaluation (probabilities)
# ---------------------------------------------------------
y_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

# ---------------------------------------------------------
# 8. Threshold Comparison
# ---------------------------------------------------------
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    results.append({
        "Threshold": t,
        "Accuracy": acc_t,
        "Precision": prec_t,
        "Recall": rec_t,
        "F1-Score": f1_t
    })

df_thresholds = pd.DataFrame(results)
df_thresholds.to_csv(os.path.join(output_dir, 'rf_threshold_comparison.csv'), index=False)

# Pick best threshold by F1-Score
best_threshold_idx = df_thresholds['F1-Score'].idxmax()
best_threshold = df_thresholds.loc[best_threshold_idx, 'Threshold']
best_acc = df_thresholds.loc[best_threshold_idx, 'Accuracy']
best_prec = df_thresholds.loc[best_threshold_idx, 'Precision']
best_rec = df_thresholds.loc[best_threshold_idx, 'Recall']
best_f1 = df_thresholds.loc[best_threshold_idx, 'F1-Score']

print(f"\n✅ Threshold comparison complete. Best threshold: {best_threshold} (F1 = {best_f1:.3f})")

# ---------------------------------------------------------
# 9. Save Metrics for Best Threshold
# ---------------------------------------------------------
rf_best_results = pd.DataFrame([{
    'Model': f'Random Forest (Threshold={best_threshold})',
    'Accuracy': best_acc,
    'Precision': best_prec,
    'Recall': best_rec,
    'F1-Score': best_f1,
    'ROC-AUC': auc
}])
rf_best_results.to_csv(os.path.join(output_dir, 'rf_metrics.csv'), index=False)

# ---------------------------------------------------------
# 10. Confusion Matrix (Best Threshold)
# ---------------------------------------------------------
y_pred_best = (y_proba >= best_threshold).astype(int)
plt.figure(figsize=(6, 5))
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title(f'Random Forest (Threshold={best_threshold}) - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_confusion_matrix.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 11. ROC Curve (probability-based)
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Random Forest (SMOTE in CV) - ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_roc_curve.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 12. Bar Chart of Metrics Across Thresholds
# ---------------------------------------------------------
df_melt = df_thresholds.melt(id_vars="Threshold", value_vars=["Accuracy","Precision","Recall","F1-Score"],
                             var_name="Metric", value_name="Score")

plt.figure(figsize=(8,6))
sns.barplot(data=df_melt, x="Threshold", y="Score", hue="Metric")
plt.title("Random Forest Performance Across Thresholds")
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_threshold_metrics_bar.png'), dpi=300)
plt.close()

print("\n✅ Best Random Forest metrics, threshold comparison, confusion matrix, ROC curve, and bar chart saved in 'random_forest_model' directory")
