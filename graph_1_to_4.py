import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure the "Graph" output folder exists
output_dir = "Graph"
os.makedirs(output_dir, exist_ok=True)

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

# STEP 1: GENERATE REALISTIC CLINICAL DATASET
np.random.seed(42)
n = 10000

# Target class: 80% No Heart Disease, 20% Yes Heart Disease
heart_disease = np.random.choice(["No", "Yes"], size=n, p=[0.8, 0.2])
is_hd = (heart_disease == "Yes").astype(int)

# Synthesize correlated clinical attributes based on medical research
age = (
    np.random.normal(loc=46 + 12 * is_hd, scale=11, size=n).clip(18, 80).round()
)
cholesterol = (
    np.random.normal(
        loc=205 + 40 * is_hd + 0.5 * (age - 45), scale=35, size=n
    )
    .clip(130, 380)
    .round()
)
systolic_bp = (
    np.random.normal(
        loc=120 + 18 * is_hd + 0.4 * (age - 45), scale=15, size=n
    )
    .clip(90, 200)
    .round()
)
bmi = (
    np.random.normal(
        loc=25.5 + 4.5 * is_hd + 0.05 * (age - 45), scale=4.5, size=n
    )
    .clip(16, 48)
    .round(1)
)
fbs = (
    np.random.normal(
        loc=100 + 25 * is_hd + 0.2 * (bmi - 25), scale=20, size=n
    )
    .clip(70, 220)
    .round()
)
triglycerides = (
    np.random.normal(
        loc=140
        + 50 * is_hd
        + 1.2 * (bmi - 25)
        + 0.3 * (cholesterol - 200),
        scale=45,
        size=n,
    )
    .clip(70, 450)
    .round()
)
crp = (
    np.random.normal(
        loc=1.8 + 3.5 * is_hd + 0.08 * (bmi - 25), scale=1.5, size=n
    )
    .clip(0.1, 15.0)
    .round(2)
)
homocysteine = (
    np.random.normal(
        loc=9.5 + 4.0 * is_hd + 0.05 * (age - 45), scale=2.5, size=n
    )
    .clip(3.0, 25.0)
    .round(2)
)
sleep_hours = (
    np.random.normal(loc=7.2 - 0.8 * is_hd, scale=1.1, size=n)
    .clip(3.0, 10.0)
    .round(1)
)

# Save updated dataset
df = pd.DataFrame(
    {
        "Age": age,
        "Blood Pressure": systolic_bp,
        "Cholesterol Level": cholesterol,
        "BMI": bmi,
        "Sleep Hours": sleep_hours,
        "Triglyceride Level": triglycerides,
        "Fasting Blood Sugar": fbs,
        "CRP Level": crp,
        "Homocysteine Level": homocysteine,
        "Heart Disease Status": heart_disease,
    }
)
df.to_csv("heart_disease_cleaned_full.csv", index=False)


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