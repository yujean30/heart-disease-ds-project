import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score

OUTPUT_DIR = "Check_Dataset_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("heart_disease.csv")
num_cols = df.select_dtypes(include='number').columns.tolist()

target = "Heart Disease Status"
y = df[target].map({"No": 0, "Yes": 1})

# ---------------------------------------------------------
# Text summaries (unchanged)
# ---------------------------------------------------------
print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df[target].value_counts())
print(df[target].value_counts(normalize=True))

print("\n" + "=" * 60)
print("NUMERIC FEATURES BY TARGET")
print("=" * 60)
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    print(f"\n--- {col} ---")
    print(df.groupby(target)[col].agg(["mean", "std", "min", "max"]).round(3))

print("\n" + "=" * 60)
print("NUMERIC CORRELATION WITH TARGET")
print("=" * 60)
for col in numeric_cols:
    corr = df[col].corr(y)
    print(f"{col:<25} {corr:.4f}")

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES VS TARGET")
print("=" * 60)
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if col == target:
        continue
    print(f"\n--- {col} ---")
    table = pd.crosstab(df[col], df[target], normalize="index")
    print(table.round(3))

# ---------------------------------------------------------
# Figure 1: distribution of every numeric feature by class.
# Pure EDA on raw data -- does NOT depend on which model you
# train, so this is already "general" and needs no changes.
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12))
axes1 = axes1.flatten()
for ax, col in zip(axes1, num_cols):
    for label, color in [('No', '#4C72B0'), ('Yes', '#DD8452')]:
        subset = df[df[target] == label][col].dropna()
        ax.hist(subset, bins=30, alpha=0.5, label=label, color=color, density=True)
    ax.set_title(col, fontsize=11)
    ax.legend(fontsize=8)
fig1.suptitle('REAL DATA: every numeric feature, split by class -- all overlap almost completely',
              fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/all_features_distribution.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved all_features_distribution.png (all 9 numeric features)")

# ---------------------------------------------------------
# Shared preprocessing (used by every model)
# ---------------------------------------------------------
X = df.drop(columns=[target]).copy()
X['Alcohol Consumption'] = X['Alcohol Consumption'].fillna('None')

ncols = X.select_dtypes(include='number').columns.tolist()
ccols = X.select_dtypes(include='object').columns.tolist()
X[ncols] = X[ncols].fillna(X[ncols].median())
X[ccols] = X[ccols].fillna(X[ccols].mode().iloc[0])

ordinal_mappings = {
    'Exercise Habits': ['Low', 'Medium', 'High'],
    'Alcohol Consumption': ['None', 'Low', 'Medium', 'High'],
    'Stress Level': ['Low', 'Medium', 'High'],
    'Sugar Consumption': ['Low', 'Medium', 'High']
}
for col, cats in ordinal_mappings.items():
    X[col] = OrdinalEncoder(categories=[cats]).fit_transform(X[[col]])

binary_cols = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
               'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']
for col in binary_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Model zoo.
# RF does not need feature scaling (splits are scale-invariant).
# LogisticRegression / KNN / SVM are distance- or gradient-based,
# so they are wrapped in a Pipeline with StandardScaler.
# SVC needs probability=True to expose predict_proba (slightly
# slower to fit, but keeps the downstream code identical).
# ---------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

model_colors = {
    "Random Forest": "#DD8452",
    "Logistic Regression": "#4C72B0",
    "KNN": "#55A868",
    "SVM": "#C44E52",
}


def get_positive_class_scores(fitted_model, X_eval):
    """
    General helper so ROC generation works for ANY sklearn classifier:
    use predict_proba when available (RF, LogReg, KNN, SVM with
    probability=True); fall back to decision_function otherwise
    (e.g. a plain SVC without probability=True). This is what makes
    a single shared ROC-plotting loop possible across model types.
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X_eval)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X_eval)
    else:
        raise ValueError("Model exposes neither predict_proba nor decision_function")


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    scores = get_positive_class_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    results[name] = {"fpr": fpr, "tpr": tpr, "auc": auc, "model": model}
    print(f"{name:<22} test AUC = {auc:.3f}")

# ---------------------------------------------------------
# Figure 2: correlation bar chart (general, data-based) +
# ROC curves for ALL FOUR models overlaid on one shared axis.
# The ROC-plotting loop is 100% general -- it only touches
# fpr/tpr/auc, which every classifier produces the same way,
# so RF/LogReg/KNN/SVM all reuse this exact code.
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

corr = df[num_cols].corrwith(y).sort_values()
axes2[0].barh(corr.index, corr.values, color='#55A868')
axes2[0].set_title('Correlation with target (all 9 numeric features)')
axes2[0].axvline(0, color='black', lw=0.8)
axes2[0].set_xlim(-0.15, 0.15)

for name, res in results.items():
    axes2[1].plot(
        res["fpr"], res["tpr"],
        color=model_colors[name], lw=2.2,
        label=f'{name} (AUC={res["auc"]:.2f})'
    )
axes2[1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random guessing (AUC=0.5)')
axes2[1].set_title('ROC curve -- RF vs Logistic Regression vs KNN vs SVM')
axes2[1].set_xlabel('False Positive Rate')
axes2[1].set_ylabel('True Positive Rate')
axes2[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/signal_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved signal_summary.png -- correlation bar chart + 4-model ROC comparison")

# ---------------------------------------------------------
# Figure 3: model-specific "what drives the prediction" plots.
# This CANNOT be one general chart, because each model exposes a
# different kind of internal signal:
#   - Random Forest       -> feature_importances_ (impurity-based)
#   - Logistic Regression -> coef_ (signed, standardized coefficients)
#   - KNN                  -> no built-in importance/coefficients
#   - SVM (rbf kernel)     -> no built-in importance/coefficients
#     (only a linear-kernel SVC exposes coef_)
# So this figure only plots RF importances and LogReg coefficients,
# and explicitly notes that KNN/SVM have no equivalent.
# ---------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
axes3[0].barh(importances.index, importances.values, color='#DD8452')
axes3[0].set_title('Random Forest: feature_importances_')

logreg_model = results["Logistic Regression"]["model"].named_steps["clf"]
coefs = pd.Series(logreg_model.coef_[0], index=X.columns).sort_values()
axes3[1].barh(coefs.index, coefs.values, color='#4C72B0')
axes3[1].axvline(0, color='black', lw=0.8)
axes3[1].set_title('Logistic Regression: standardized coefficients')

fig3.suptitle('Model-specific explainability (KNN / SVM have no native equivalent)', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_specific_explainability.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved model_specific_explainability.png (RF importances + LogReg coefficients)")