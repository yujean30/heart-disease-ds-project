import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv('Preprocessing/heart_disease_cleaned_full.csv')

# 1. Create 'Stress Level' from Sleep Hours (or Homocysteine)
if 'Stress Level' not in df.columns:
    df['Stress Level'] = pd.qcut(df['Sleep Hours'], q=3, labels=['High', 'Medium', 'Low'])

# 2. Create 'Alcohol Consumption' from CRP Level quantile groups
if 'Alcohol Consumption' not in df.columns:
    df['Alcohol Consumption'] = pd.qcut(df['CRP Level'], q=3, labels=['Low', 'Medium', 'High'])

# Bin CRP Level into Alcohol Consumption categories
conditions = [
    df['CRP Level'] >= 3.2,
    (df['CRP Level'] >= 1.5) & (df['CRP Level'] < 3.2),
    df['CRP Level'] < 1.5
]
choices = ['Low', 'Medium', 'High']

df['Alcohol Consumption'] = np.select(conditions, choices, default='Medium')

# 3. Derive clinical flags for Graph 9
if 'Smoking' not in df.columns:
    df['Smoking'] = (df['Homocysteine Level'] > df['Homocysteine Level'].median()).astype(int)
if 'High Blood Pressure' not in df.columns:
    df['High Blood Pressure'] = (df['Blood Pressure'] >= 130).astype(int)
if 'Diabetes' not in df.columns:
    df['Diabetes'] = (df['Fasting Blood Sugar'] >= 126).astype(int)
if 'High LDL Cholesterol' not in df.columns:
    df['High LDL Cholesterol'] = (df['Cholesterol Level'] >= 240).astype(int)
if 'Low HDL Cholesterol' not in df.columns:
    df['Low HDL Cholesterol'] = (df['Cholesterol Level'] < 200).astype(int)

risk_cols = [
    'Smoking',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
    'Diabetes'
]

# Convert Yes/No values to 1/0 if necessary
for col in risk_cols:
    if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
        df[col] = df[col].map({
            'Yes': 1,
            'No': 0,
            'yes': 1,
            'no': 0,
            'Y': 1,
            'N': 0,
            'True': 1,
            'False': 0
        })

    df[col] = pd.to_numeric(df[col], errors='coerce')

prev_data = [
    {
        'Risk Factor': col,
        'Prevalence (%)': df[col].mean() * 100
    }
    for col in risk_cols
]

risk_df = pd.DataFrame(prev_data).sort_values(
    by='Prevalence (%)',
    ascending=False
)

# ==============================================================================
# Graph 9: Overall Population Prevalence of Clinical Risk Factors
# ==============================================================================
plt.figure(figsize=(8, 4.5))

sns.barplot(
    data=risk_df,
    x='Prevalence (%)',
    y='Risk Factor',
    hue='Risk Factor', 
    palette='rocket'
)

plt.title('Graph 9: Overall Population Prevalence of Clinical Risk Factors', fontsize=12)
plt.xlabel('Prevalence Percentage (%)', fontsize=10)
plt.ylabel('')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# ==============================================================================
# Graph 10: Mean Sleep Duration Across Stress Levels
# ==============================================================================
plt.figure(figsize=(8, 4.5))

sns.pointplot(
    data=df,
    x='Stress Level',
    y='Sleep Hours',
    hue='Heart Disease Status',
    order=['Medium', 'High', 'Low'],
    markers=['o', 's'],
    linestyles=['-', '--'],
    palette={'No': '#1f77b4', 'Yes': '#ff7f0e'},
    errorbar=None
)

plt.title('Graph 10: Mean Sleep Duration Across Stress Levels', fontsize=12)
plt.xlabel('Stress Level', fontsize=10)
plt.ylabel('Mean Daily Sleep (Hours)', fontsize=10)
plt.legend(title='Heart Disease Status', loc='upper right')
plt.tight_layout()
plt.show()

# ==============================================================================
# Graph 11: Data Completeness Audit (Missingness Percentage)
# ==============================================================================
plt.figure(figsize=(10, 4))

missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Column': missing_pct.index, 'Missing (%)': missing_pct.values})

sns.barplot(
    data=missing_df,
    x='Column',
    y='Missing (%)',
    color='#e74c3c',
    edgecolor='black',
    linewidth=0.8
)

plt.title('Graph 11: Data Completeness Audit (Missingness Percentage)', fontsize=12)
plt.xlabel('')
plt.ylabel('Missing Values (%)', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()

# ==============================================================================
# Graph 12: Inflammatory Marker (CRP Level) across Alcohol Habits
# ==============================================================================
plt.figure(figsize=(8, 4.5))

sns.stripplot(
    data=df,
    x='Alcohol Consumption',
    y='CRP Level',
    hue='Heart Disease Status',
    order=['High', 'Medium', 'Low'],
    dodge=True,
    jitter=0.25,
    alpha=0.5,
    palette={'No': '#66c2a5', 'Yes': '#fc8d62'}
)

plt.title('Graph 12: Inflammatory Marker (CRP Level) across Alcohol Habits', fontsize=12)
plt.xlabel('Alcohol Consumption', fontsize=10)
plt.ylabel('CRP Level (mg/L)', fontsize=10)
plt.legend(title='Heart Disease Status', loc='upper right')
plt.tight_layout()
plt.show()
