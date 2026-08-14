import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global aesthetic theme for publication-quality visuals
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    }
)

# STEP 1: DATA PREPROCESSING & CLEANING (CLO2 - Excellent Criteria)

# Load cleaned dataset
df = pd.read_csv("heart_disease_cleaned_full.csv")

# Clean whitespace/formatting from categorical string columns
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

# Continuous numerical feature columns for heatmap and analysis
num_cols = [
    "Age",
    "Blood Pressure",
    "Cholesterol Level",
    "BMI",
    "Sleep Hours",
    "Triglyceride Level",
    "Fasting Blood Sugar",
    "CRP Level",
    "Homocysteine Level",
]

# Ensure correct data type casting
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# GRAPH 1: Target Class Ratio (Heart Disease Status - Donut Chart)
fig, ax = plt.subplots(figsize=(6, 6))
target_counts = df["Heart Disease Status"].value_counts()

# Pie / Donut Chart Parameters
wedges, texts, autotexts = ax.pie(
    target_counts,
    labels=target_counts.index,
    autopct="%1.1f%%",
    pctdistance=0.65,
    startangle=90,
    colors=["#4c72b0", "#c44e52"],  # Standard blue & red palette
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11),
)

# Fine-tune percentage text visibility
for autotext in autotexts:
    autotext.set_fontsize(10.5)

ax.set_title("Graph 1: Target Class Ratio (Heart Disease Status)", pad=15)
plt.savefig("graph_1_donut_target.png", dpi=300, bbox_inches="tight")
plt.close()


# GRAPH 2: Patient Age Distribution by Heart Disease Status (Hist + KDE)
fig, ax = plt.subplots(figsize=(8, 5))

# Seaborn Histplot with KDE overlay using Set2 Palette
sns.histplot(
    data=df.dropna(subset=["Age"]),
    x="Age",
    hue="Heart Disease Status",
    kde=True,
    palette="Set2",
    element="step",
    bins=20,
    ax=ax,
)

ax.set_title(
    "Graph 2: Patient Age Distribution by Heart Disease Status", pad=15
)
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Patient Count")
plt.savefig("graph_2_hist_kde_age.png", dpi=300, bbox_inches="tight")
plt.close()


# GRAPH 3: Pairwise Correlation Heatmap of Continuous Attributes
fig, ax = plt.subplots(figsize=(9, 7))

# Calculate Pearson correlation matrix
corr_matrix = df[num_cols].corr()

# Annotate correlation values rounded to 2 decimal places
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
)

ax.set_title(
    "Graph 3: Pairwise Correlation Heatmap of Continuous Attributes", pad=15
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig("graph_3_heatmap_corr.png", dpi=300, bbox_inches="tight")
plt.close()


# GRAPH 4: Cholesterol Level Distribution & Quartiles (Notched Box Plot)
fig, ax = plt.subplots(figsize=(7, 5))

# Notched box plot using Pastel1 palette
sns.boxplot(
    data=df.dropna(subset=["Cholesterol Level"]),
    x="Heart Disease Status",
    y="Cholesterol Level",
    palette="Pastel1",
    notch=True,
    ax=ax,
)

ax.set_title("Graph 4: Cholesterol Level Distribution & Quartiles", pad=15)
ax.set_xlabel("Heart Disease Status")
ax.set_ylabel("Serum Cholesterol (mg/dL)")
plt.savefig("graph_4_boxplot_cholesterol.png", dpi=300, bbox_inches="tight")
plt.close()