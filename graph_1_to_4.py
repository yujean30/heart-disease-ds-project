import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset
df = pd.read_csv("heart_disease_cleaned_full.csv")

# Ensure the "Graph" output folder exists
output_dir = "Graph"
os.makedirs(output_dir, exist_ok=True)

# GRAPH 1: Target Class Ratio (Heart Disease Status - Donut Chart)
fig, ax = plt.subplots(figsize=(6, 6))
target_counts = df["Heart Disease Status"].value_counts()

wedges, texts, autotexts = ax.pie(
    target_counts,
    labels=target_counts.index,
    autopct="%1.1f%%",
    pctdistance=0.65,
    startangle=90,
    colors=["#4c72b0", "#c44e52"],
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11),
)

for autotext in autotexts:
    autotext.set_fontsize(10.5)

ax.set_title("Graph 1: Target Class Ratio (Heart Disease Status)", pad=15)
plt.savefig(
    os.path.join(output_dir, "graph_1_donut_target.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# GRAPH 2: Patient Age Distribution by Heart Disease Status (Hist + KDE)
fig, ax = plt.subplots(figsize=(8, 5))

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
plt.savefig(
    os.path.join(output_dir, "graph_2_hist_kde_age.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# GRAPH 3: Pairwise Correlation Heatmap
fig, ax = plt.subplots(figsize=(9, 7))
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
corr_matrix = df[num_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
    vmin=-0.3,
    vmax=0.8,
)
ax.set_title(
    "Graph 3: Pairwise Correlation Heatmap of Continuous Attributes", pad=15
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.savefig(
    os.path.join(output_dir, "graph_3_heatmap_corr.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# GRAPH 4: Cholesterol Level Box Plot
fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Heart Disease Status",
    y="Cholesterol Level",
    palette="Pastel1",
    notch=True,
    ax=ax,
)

ax.set_title(
    "Graph 4: Serum Cholesterol Distribution by Heart Disease Status", pad=15
)
ax.set_xlabel("Heart Disease Status")
ax.set_ylabel("Serum Cholesterol (mg/dL)")
plt.savefig(
    os.path.join(output_dir, "graph_4_boxplot_cholesterol.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("All graphs generated successfully.")