import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 1. Load Dataset
df = pd.read_csv("heart_disease.csv")

# -------------------------------------------------------------------------
# GRAPH 1: Donut Chart (Target Class Imbalance)
# -------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
target_counts = df["Heart Disease Status"].value_counts()
plt.pie(
    target_counts,
    labels=target_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#4c72b0", "#c44e52"],
    wedgeprops=dict(width=0.4, edgecolor="w"),
)
plt.title("Graph 1: Target Class Ratio (Heart Disease Status)")
plt.tight_layout()
plt.savefig("graph_1_donut_target.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 2: Histogram with KDE Overlay (Age Distribution)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Age",
    hue="Heart Disease Status",
    kde=True,
    palette="Set2",
    element="step",
)
plt.title("Graph 2: Patient Age Distribution by Heart Disease Status")
plt.xlabel("Age (Years)")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.savefig("graph_2_hist_kde_age.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 3: Correlation Matrix Heatmap (Numerical Correlations)
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
num_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(
    df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True
)
plt.title("Graph 3: Pairwise Correlation Heatmap of Continuous Attributes")
plt.tight_layout()
plt.savefig("graph_3_heatmap_corr.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 4: Box Plot (Cholesterol Outliers & Interquartile Ranges)
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Heart Disease Status",
    y="Cholesterol Level",
    palette="Pastel1",
    notch=True,
)
plt.title("Graph 4: Cholesterol Level Distribution & Quartiles")
plt.ylabel("Serum Cholesterol (mg/dL)")
plt.tight_layout()
plt.savefig("graph_4_boxplot_cholesterol.png")
plt.show()

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

# -------------------------------------------------------------------------
# GRAPH 9: Horizontal Bar Chart (Prevalence of Risk Factors)
# -------------------------------------------------------------------------
risk_cols = [
    "Smoking",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
]
risk_percentages = [(df[col] == "Yes").mean() * 100 for col in risk_cols]

plt.figure(figsize=(8, 4.5))
sns.barplot(x=risk_percentages, y=risk_cols, palette="magma", orient="h")
plt.title("Graph 9: Overall Population Prevalence of Clinical Risk Factors")
plt.xlabel("Prevalence Percentage (%)")
plt.tight_layout()
plt.savefig("graph_9_hbar_risk_factors.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 10: Point Plot (Sleep Duration Interaction with Stress Level)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.pointplot(
    data=df,
    x="Stress Level",
    y="Sleep Hours",
    hue="Heart Disease Status",
    markers=["o", "s"],
    linestyles=["-", "--"],
    palette="tab10",
    errorbar=None,
)
plt.title("Graph 10: Mean Sleep Duration Across Stress Levels")
plt.ylabel("Mean Daily Sleep (Hours)")
plt.tight_layout()
plt.savefig("graph_10_pointplot_sleep_stress.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 11: Missing Value Percentage Bar Plot (Data Quality Audit)
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct_filtered = missing_pct[missing_pct > 0]
missing_pct_filtered.plot(kind="bar", color="#e74c3c", edgecolor="black")
plt.title("Graph 11: Data Completeness Audit (Missingness Percentage)")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graph_11_missing_values.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 12: Jitter / Strip Plot (CRP Level across Alcohol Use)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.stripplot(
    data=df,
    x="Alcohol Consumption",
    y="CRP Level",
    hue="Heart Disease Status",
    jitter=0.25,
    alpha=0.5,
    palette="Set2",
    dodge=True,
)
plt.title(
    "Graph 12: Inflammatory Marker (CRP Level) across Alcohol Habits"
)
plt.ylabel("CRP Level (mg/L)")
plt.tight_layout()
plt.savefig("graph_12_stripplot_crp_alcohol.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 13: Continuous Density Plot / KDE (Fasting Blood Sugar)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=df,
    x="Fasting Blood Sugar",
    hue="Heart Disease Status",
    common_norm=False,
    fill=True,
    alpha=0.35,
    palette="crest",
)
plt.title("Graph 13: Kernel Density Estimate (KDE) of Fasting Blood Sugar")
plt.xlabel("Fasting Blood Sugar (mg/dL)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("graph_13_kde_sugar.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 14: Cumulative Empirical CDF Plot (Homocysteine Risk Shift)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.ecdfplot(
    data=df, x="Homocysteine Level", hue="Heart Disease Status", palette="Set1"
)
plt.title("Graph 14: Empirical Cumulative Distribution of Homocysteine Levels")
plt.xlabel("Homocysteine Level (µmol/L)")
plt.ylabel("Cumulative Probability")
plt.tight_layout()
plt.savefig("graph_14_ecdf_homocysteine.png")
plt.show()

# -------------------------------------------------------------------------
# GRAPH 15: Hexbin Bivariate Density Plot (Triglycerides vs Glucose)
# -------------------------------------------------------------------------
clean_df = df[["Triglyceride Level", "Fasting Blood Sugar"]].dropna()

plt.figure(figsize=(8, 6))
hb = plt.hexbin(
    clean_df["Triglyceride Level"],
    clean_df["Fasting Blood Sugar"],
    gridsize=25,
    cmap="Blues",
    mincnt=1,
)
plt.colorbar(hb, label="Patient Count Density")
plt.title(
    "Graph 15: Hexbin Bivariate Density: Triglycerides vs. Fasting Sugar"
)
plt.xlabel("Triglyceride Level (mg/dL)")
plt.ylabel("Fasting Blood Sugar (mg/dL)")
plt.tight_layout()
plt.savefig("graph_15_hexbin_density.png")
plt.show()