import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# set the project and output folder paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# load preprocessed data
print('Loading preprocessed training and test data...')
X_train = pd.read_csv(os.path.join(project_dir, 'X_train_preprocessed.csv'))
X_test = pd.read_csv(os.path.join(project_dir, 'X_test_preprocessed.csv'))
# load the training targets and make them to 1D array
y_train = pd.read_csv(os.path.join(project_dir, 'y_train.csv')).values.ravel() 
y_test = pd.read_csv(os.path.join(project_dir, 'y_test.csv')).values.ravel() 

print(f'Training rows: {len(X_train)} | Test rows: {len(X_test)}')
print(f'Training class distribution: {pd.Series(y_train).value_counts().to_dict()}')

# setup Baseline Model & hyperparameter tuning
print('Training Logistic Regression with 5-fold cross-validation...')
lr_base = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'], # reduce overfitting 
    'solver': ['liblinear'] #suitable for small datasets and supports L1 penalty
}

grid_search = GridSearchCV(
    estimator=lr_base,
    param_grid=param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# select the best model from grid search
best_lr = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation recall: {grid_search.best_score_:.4f}')

# Save Trained Model Object (.pkl) for Streamlit Prototype
joblib.dump(best_lr, os.path.join(output_dir, 'best_lr_model.pkl'))

# Model Evaluation
y_pred = best_lr.predict(X_test)
y_proba = best_lr.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print('\nTEST SET RESULTS')
print(f'Accuracy : {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall   : {rec:.4f}')
print(f'F1-score : {f1:.4f}')
print(f'ROC-AUC  : {auc:.4f}')

# Save Metrics CSV
lr_results = pd.DataFrame([{
    'Model': 'Logistic Regression (Baseline)',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1-Score': f1,
    'ROC-AUC': auc
}])
lr_results.to_csv(os.path.join(output_dir, 'lr_baseline_metrics.csv'), index=False)

# Generate & Save Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Baseline Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lr_confusion_matrix.png'), dpi=300)
plt.close()

# Generate & Save ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Baseline Logistic Regression - ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lr_roc_curve.png'), dpi=300)
plt.close()

print(f'Outputs saved to: {output_dir}')


