import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Ensure the output directory exists
output_dir = "Graph"
os.makedirs(output_dir, exist_ok=True)

# Set global aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 2. Load Dataset & Clean Headers
df = pd.read_csv("heart_disease_cleaned_full.csv")
df.columns = df.columns.str.strip()

# Helper function for fuzzy column matching (handles spaces, underscores, and lowercase)
def find_column(df_cols, keywords):
    cols_normalized = {col: col.lower().replace(" ", "").replace("_", "") for col in df_cols}
    for kw in keywords:
        kw_norm = kw.lower().replace(" ", "").replace("_", "")
        for orig_col, norm_col in cols_normalized.items():
            if kw_norm in norm_col:
                return orig_col
    return None

# Resolve column names dynamically
exercise_col = find_column(df.columns, ["exercise", "activity", "habit"])
target_col   = find_column(df.columns, ["heartdisease", "status", "disease", "target"])
bmi_col      = find_column(df.columns, ["bmi", "bodymass"])
age_col      = find_column(df.columns, ["age"])
bp_col       = find_column(df.columns, ["bloodpressure", "bp"])
chol_col     = find_column(df.columns, ["cholesterol", "chol"])

print(f"Detected columns -> Target: '{target_col}', Exercise: '{exercise_col}', BMI: '{bmi_col}'")

# Setup dynamic palette matching target data type
color_list = ["#2ecc71", "#e74c3c"]
if target_col and target_col in df.columns:
    unique_targets = sorted(df[target_col].unique())
    palette = {val: color_list[i % len(color_list)] for i, val in enumerate(unique_targets)}
else:
    raise KeyError(f"Target column for heart disease status could not be found. Available columns: {list(df.columns)}")

# -------------------------------------------------------------------------
# GRAPH 5: Split Violin Plot (BMI by Category & Heart Disease Status)
# -------------------------------------------------------------------------
if bmi_col and bmi_col in df.columns:
    df["BMI_Category"] = pd.cut(
        df[bmi_col],
        bins=[0, 18.5, 24.9, 29.9, df[bmi_col].max()],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df,
        x="BMI_Category",
        y=bmi_col,
        hue=target_col,
        split=True,
        inner="quartile",
        palette=palette,
    )
    plt.title("Graph 5: BMI Distribution by Category and Heart Disease Status")
    plt.ylabel("Body Mass Index (BMI)")
    plt.xlabel("BMI Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_5_violin_bmi.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("Skipping Graph 5: BMI column not found.")

# -------------------------------------------------------------------------
# GRAPH 6: Grouped Bar Chart (Exercise Habits vs. Disease Prevalence)
# -------------------------------------------------------------------------
if exercise_col and exercise_col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=exercise_col,
        hue=target_col,
        palette=palette,
    )
    plt.title("Graph 6: Impact of Physical Activity Level on Heart Disease")
    plt.xlabel("Exercise Activity Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_6_grouped_bar_exercise.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print(f"Skipping Graph 6: Exercise column not found. Available columns: {list(df.columns)}")

# -------------------------------------------------------------------------
# GRAPH 7: Stacked Bar Chart (Age Binned Risk Multiplier)
# -------------------------------------------------------------------------
if age_col and age_col in df.columns:
    df["Age_Group"] = pd.cut(
        df[age_col],
        bins=[18, 35, 50, 65, df[age_col].max() + 1],
        labels=["18-35", "36-50", "51-65", "66-80"],
    )
    age_hd = pd.crosstab(df["Age_Group"], df[target_col])

    fig, ax = plt.subplots(figsize=(8, 5))
    age_hd.plot(kind="bar", stacked=True, color=color_list, ax=ax, width=0.6)
    plt.title("Graph 7: Age Group Risk Proportion")
    plt.ylabel("Count")
    plt.xlabel("Age Group")
    plt.legend(title="Heart Disease")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_7_stacked_age_risk.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("Skipping Graph 7: Age column not found.")

# -------------------------------------------------------------------------
# GRAPH 8: Scatter Plot with Alpha Transparency (BP vs Cholesterol)
# -------------------------------------------------------------------------
if bp_col and chol_col and bp_col in df.columns and chol_col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=bp_col,
        y=chol_col,
        hue=target_col,
        alpha=0.4,
        palette=palette,
        s=30,
    )
    plt.title("Graph 8: Bivariate Relationship: Blood Pressure vs. Cholesterol")
    plt.xlabel("Blood Pressure (mmHg)")
    plt.ylabel("Cholesterol (mg/dL)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_8_scatter_bp_chol.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("Skipping Graph 8: Blood Pressure or Cholesterol column not found.")

print("Execution finished! Generated charts are saved in the 'Graph' directory.")