import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_curve, roc_auc_score

OUTPUT_DIR = "Check_Dataset_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("heart_disease.csv")
num_cols = df.select_dtypes(include='number').columns.tolist()

target = "Heart Disease Status"
y = df[target].map({"No": 0, "Yes": 1})

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
    print(
        df.groupby(target)[col]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )

print("\n" + "=" * 60)
print("NUMERIC CORRELATION WITH TARGET")
print("=" * 60)

for col in numeric_cols:
    corr = df[col].corr(y)
    print(f"{col:<25} {corr:.4f}")

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES VS TARGET")
print("=" * 60)

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:

    if col == target:
        continue

    print(f"\n--- {col} ---")

    table = pd.crosstab(
        df[col],
        df[target],
        normalize="index"
    )

    print(table.round(3))

# ---------------------------------------------------------
# Figure 1: Every numeric feature's distribution by class
# (not just Age/Cholesterol) -- all overlap almost completely
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12))
axes1 = axes1.flatten()

for ax, col in zip(axes1, num_cols):
    for label, color in [('No', '#4C72B0'), ('Yes', '#DD8452')]:
        subset = df[df['Heart Disease Status'] == label][col].dropna()
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
# Figure 2: Correlation bar chart + ROC curve summary
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

y = df['Heart Disease Status'].map({'No': 0, 'Yes': 1})
corr = df[num_cols].corrwith(y).sort_values()
axes2[0].barh(corr.index, corr.values, color='#55A868')
axes2[0].set_title('Correlation with target (all 9 numeric features)')
axes2[0].axvline(0, color='black', lw=0.8)
axes2[0].set_xlim(-0.15, 0.15)

X = df.drop(columns=['Heart Disease Status']).copy()
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
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba)
auc_real = roc_auc_score(y_test, proba)

axes2[1].plot(fpr, tpr, color='#DD8452', lw=2.5, label=f'REAL model, all 20 features (AUC={auc_real:.2f})')
axes2[1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random guessing (AUC=0.5)')
axes2[1].set_title('ROC curve -- Random Forest trained on ALL features')
axes2[1].set_xlabel('False Positive Rate')
axes2[1].set_ylabel('True Positive Rate')
axes2[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/signal_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved signal_summary.png -- RF test AUC using all features: {auc_real:.3f}")