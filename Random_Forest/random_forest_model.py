import os, sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path
from SMOTE import create_smote   # import shared helper

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================
# 0. Shared Setup -- loaded ONCE, used by both models below
# ===========================================================
BASE_DIR = os.path.dirname(__file__)  # path to Random_Forest folder
os.makedirs(os.path.join(BASE_DIR, 'random_forest_model'), exist_ok=True)           # SMOTE model
os.makedirs(os.path.join(BASE_DIR, 'random_forest_model_no_smote'), exist_ok=True)  # No-SMOTE model

# Raw, imbalanced training set -- used for honest CV (both models' search
# + threshold tuning) and as the actual training data for the No-SMOTE model.
X_train_raw = pd.read_csv('X_train_preprocessed.csv')
y_train_raw = pd.read_csv('y_train.csv').values.ravel()

# Real, untouched test set -- used for final evaluation of BOTH models, exactly once each.
X_test = pd.read_csv('X_test_preprocessed.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Raw imbalanced training set: {X_train_raw.shape[0]} rows, "
      f"{pd.Series(y_train_raw).value_counts().to_dict()}")
print(f"Test set (real, untouched):  {X_test.shape[0]} rows, "
      f"{pd.Series(y_test).value_counts().to_dict()}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# a place to collect both models' final numbers for the side-by-side comparison table printed at the very end
comparison_rows = []

def evaluate_and_report(model, model_label, output_dir, threshold,
                         cm_cmap, extra_title=""):
    """Shared evaluation + plotting + printed report for one fitted model.
    Runs the SAME evaluation logic for both the SMOTE and No-SMOTE models,
    so their numbers are directly comparable."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y_test, y_pred, digits=4)

    print("\n" + "=" * 50)
    print(f"{model_label} EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy Score : {acc * 100:.2f}%")
    print(f"Precision Score: {prec:.4f}")
    print(f"Recall Score   : {rec:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print(f"Best Threshold : {threshold}")
    print("\nCONFUSION MATRIX:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(report_text)

    # JSON metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            "model": model_label,
            "threshold": threshold,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": roc_auc,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp),
                                  "FN": int(fn), "TP": int(tp)},
            "classification_report": report_dict
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # CSV metrics (same shape UI.py already expects)
    pd.DataFrame([{
        'Model': f'{model_label} (Threshold={threshold})',
        'Threshold': threshold, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc_auc
    }]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cm_cmap, cbar=False,
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.title(f'{model_label} - Confusion Matrix{extra_title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'{model_label} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    comparison_rows.append({
        'Model': model_label, 'Threshold': threshold, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        'ROC-AUC': roc_auc
    })


def save_feature_importance(model, columns, output_dir, title, color):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, 'feature_importance.csv'), header=['Importance'])
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, color=color)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()


# ===========================================================
# PART A -- SMOTE MODEL (new standard)
# ===========================================================
print("\n" + "=" * 60)
print("= PART A: TRAINING SMOTE MODEL")
print("=" * 60)

search_pipeline = ImbPipeline([
    ('smote', create_smote()),   # dynamic SMOTENC
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

smote_param_dist = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 10, 20, 30, 50],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": ["balanced", None],
}

smote_search = RandomizedSearchCV(
    estimator=search_pipeline, param_distributions=smote_param_dist,
    n_iter=30, cv=cv_strategy, scoring="f1", n_jobs=-1,
    random_state=42, verbose=1
)
smote_search.fit(X_train_raw, y_train_raw)
print(f"\n[SMOTE model] Best hyperparameters (via honest CV): {smote_search.best_params_}")

# Threshold tuning
smote_oof_proba = cross_val_predict(
    smote_search.best_estimator_, X_train_raw, y_train_raw,
    cv=cv_strategy, method='predict_proba', n_jobs=-1
)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
smote_threshold_results = []
for t in thresholds:
    pred = (smote_oof_proba >= t).astype(int)
    smote_threshold_results.append({
        "Threshold": round(t, 2),
        "Precision": precision_score(y_train_raw, pred, zero_division=0),
        "Recall": recall_score(y_train_raw, pred, zero_division=0),
        "F1-Score": f1_score(y_train_raw, pred, zero_division=0),
    })
df_smote_thresholds = pd.DataFrame(smote_threshold_results)
df_smote_thresholds.to_csv(
    os.path.join(BASE_DIR, 'random_forest_model', 'threshold_comparison.csv'),
    index=False
)
smote_best_threshold = float(df_smote_thresholds.loc[df_smote_thresholds['F1-Score'].idxmax(), 'Threshold'])
print(f"[SMOTE model] Best threshold (via honest CV): {smote_best_threshold}")

# Final model: pipeline’s best_estimator_
smote_final_model = smote_search.best_estimator_
smote_final_model.fit(X_train_raw, y_train_raw)
joblib.dump(
    smote_final_model,
    os.path.join(BASE_DIR, 'random_forest_model', 'random_forest_tuned.joblib')
)

evaluate_and_report(
    smote_final_model, "RANDOM FOREST (SMOTE)", os.path.join(BASE_DIR, 'random_forest_model'),
    smote_best_threshold, cm_cmap='Greens'
)
save_feature_importance(
    smote_final_model.named_steps['classifier'], X_train_raw.columns, os.path.join(BASE_DIR, 'random_forest_model'),
    "Random Forest (SMOTE) Feature Importance", color='seagreen'
)

# ===========================================================
# PART B -- NO-SMOTE MODEL
# ===========================================================
# Trained directly on the raw imbalanced data; class_weight is part of the
# search space instead of SMOTE. No pipeline needed -- there's no
# resampling step, so plain CV is already honest here.
print("\n" + "=" * 60)
print("= PART B: TRAINING NO-SMOTE MODEL")
print("=" * 60)

no_smote_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample", None],
}
rf_plain = RandomForestClassifier(random_state=42, n_jobs=-1)

no_smote_search = RandomizedSearchCV(
    estimator=rf_plain, param_distributions=no_smote_param_dist,
    n_iter=30, cv=cv_strategy, scoring="f1", n_jobs=-1,
    random_state=42, verbose=1
)
no_smote_search.fit(X_train_raw, y_train_raw)
print(f"\n[No-SMOTE model] Best hyperparameters: {no_smote_search.best_params_}")

print("[No-SMOTE model] Generating out-of-fold probabilities for threshold tuning...")
no_smote_oof_proba = cross_val_predict(
    no_smote_search.best_estimator_, X_train_raw, y_train_raw,
    cv=cv_strategy, method='predict_proba', n_jobs=-1
)[:, 1]

no_smote_threshold_results = []
for t in thresholds:
    pred = (no_smote_oof_proba >= t).astype(int)
    no_smote_threshold_results.append({
        "Threshold": round(t, 2),
        "Precision": precision_score(y_train_raw, pred, zero_division=0),
        "Recall": recall_score(y_train_raw, pred, zero_division=0),
        "F1-Score": f1_score(y_train_raw, pred, zero_division=0),
    })
df_no_smote_thresholds = pd.DataFrame(no_smote_threshold_results)
df_no_smote_thresholds.to_csv(
    os.path.join(BASE_DIR, 'random_forest_model_no_smote', 'threshold_comparison.csv'),
    index=False
)
no_smote_best_threshold = float(df_no_smote_thresholds.loc[df_no_smote_thresholds['F1-Score'].idxmax(), 'Threshold'])
print(f"[No-SMOTE model] Best threshold (via honest CV): {no_smote_best_threshold}")

no_smote_final_model = no_smote_search.best_estimator_
no_smote_final_model.fit(X_train_raw, y_train_raw)
joblib.dump(no_smote_final_model, os.path.join(BASE_DIR, 'random_forest_model_no_smote', 'random_forest_no_smote.joblib'))

evaluate_and_report(
    no_smote_final_model, "RANDOM FOREST (NO SMOTE)", os.path.join(BASE_DIR, 'random_forest_model_no_smote'),
    no_smote_best_threshold, cm_cmap='Blues'
)
save_feature_importance(
    no_smote_final_model, X_train_raw.columns, os.path.join(BASE_DIR, 'random_forest_model_no_smote'),
    "Random Forest (No SMOTE) Feature Importance", color='steelblue'
)

# ===========================================================
# PART C -- Shared scaler + Final Side-by-Side Comparison
# ===========================================================
if os.path.exists('scaler.pkl'):
    scaler = joblib.load('scaler.pkl')
    joblib.dump(scaler, os.path.join(BASE_DIR, 'random_forest_model', 'scaler.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'random_forest_model_no_smote', 'scaler.pkl'))
else:
    print("\nWARNING: scaler.pkl not found -- run preprocess.py first.")

OUT_DIR = 'random_forest'
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(OUT_DIR, 'rf_smote_vs_no_smote_comparison.csv'), index=False)

print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON (same test set, evaluated once each)")
print("=" * 60)
print(comparison_df.to_string(index=False))
print("\nSaved comparison table to rf_smote_vs_no_smote_comparison.csv")
print("SMOTE model artifacts:    'random_forest_model/'")
print("No-SMOTE model artifacts: 'random_forest_model_no_smote/'")