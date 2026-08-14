import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 1. Load Dataset
df = pd.read_csv("heart_disease_cleaned_full.csv")

# -------------------------------------------------------------------------
# GRAPH 5: Split Violin Plot (BMI by Category & Heart Disease Status)
# -------------------------------------------------------------------------
df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 24.9, 29.9, df["BMI"].max()],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="BMI_Category",
    y="BMI",
    hue="Heart Disease Status",
    split=True,
    inner="quartile",
    palette={"No": "#2ecc71", "Yes": "#e74c3c"}
)
plt.title("Graph 5: BMI Distribution by Category and Heart Disease Status")
plt.ylabel("Body Mass Index (BMI)")
plt.xlabel("BMI Category")
plt.tight_layout()
plt.savefig("graph_5_violin_bmi.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 6: Grouped Bar Chart (Exercise Habits vs. Disease Prevalence)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df, x="Exercise Habits", hue="Heart Disease Status", palette={"No": "#2ecc71", "Yes": "#e74c3c"}
)
plt.title("Graph 6: Impact of Physical Activity Level on Heart Disease")
plt.xlabel("Exercise Activity Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graph_6_grouped_bar_exercise.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 7: Stacked Bar Chart (Age Binned Risk Multiplier)
# -------------------------------------------------------------------------
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[18, 35, 50, 65, 80],
    labels=["18-35", "36-50", "51-65", "66-80"]
)
age_hd = pd.crosstab(df["Age_Group"], df["Heart Disease Status"])

fig, ax = plt.subplots(figsize=(8, 5))
age_hd.plot(
    kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax, width=0.6
)
plt.title("Graph 7: Age Group Risk Proportion")
plt.ylabel("Count")
plt.xlabel("Age Group")
plt.legend(title="Heart Disease")
plt.tight_layout()
plt.savefig("graph_7_stacked_age_risk.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 8: Scatter Plot with Alpha Transparency (BP vs Cholesterol)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Blood Pressure",
    y="Cholesterol Level",
    hue="Heart Disease Status",
    alpha=0.4,
    palette="Dark2",
    s=30,
)
plt.title("Graph 8: Bivariate Relationship: Blood Pressure vs. Cholesterol")
plt.xlabel("Blood Pressure (mmHg)")
plt.ylabel("Cholesterol (mg/dL)")
plt.tight_layout()
plt.savefig("graph_8_scatter_bp_chol.png")
plt.show()