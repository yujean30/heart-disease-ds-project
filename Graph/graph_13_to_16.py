import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("Preprocessing/heart_disease_cleaned_full.csv")


# ============================================================================
# GRAPH 13: Kernel Density Estimate (KDE) of Fasting Blood Sugar
# ============================================================================

plt.figure(figsize=(8, 5))

sns.kdeplot(
    data=df,
    x="Fasting Blood Sugar",
    hue="Heart Disease Status",
    common_norm=False,
    fill=True,
    alpha=0.35,
    palette="crest"
)

plt.title(
    "Graph 13: Kernel Density Estimate (KDE) of Fasting Blood Sugar"
)
plt.xlabel("Fasting Blood Sugar (mg/dL)")
plt.ylabel("Probability Density")
plt.tight_layout()

plt.savefig("Graph/graph_13_kde_sugar.png")
plt.show()


# ============================================================================
# GRAPH 14: Empirical Cumulative Distribution of Homocysteine Levels
# ============================================================================

plt.figure(figsize=(8, 5))

sns.ecdfplot(
    data=df,
    x="Homocysteine Level",
    hue="Heart Disease Status",
    palette="Set1"
)

plt.title(
    "Graph 14: Empirical Cumulative Distribution of Homocysteine Levels"
)
plt.xlabel("Homocysteine Level (µmol/L)")
plt.ylabel("Cumulative Probability")
plt.tight_layout()

plt.savefig("Graph/graph_14_ecdf_homocysteine.png")
plt.show()


# ============================================================================
# GRAPH 15: Hexbin Bivariate Density - Triglycerides vs Fasting Blood Sugar
# ============================================================================

clean_df = df[
    ["Triglyceride Level", "Fasting Blood Sugar"]
].dropna()

plt.figure(figsize=(8, 6))

hb = plt.hexbin(
    clean_df["Triglyceride Level"],
    clean_df["Fasting Blood Sugar"],
    gridsize=25,
    cmap="Blues",
    mincnt=1
)

plt.colorbar(
    hb,
    label="Patient Count Density"
)

plt.title(
    "Graph 15: Hexbin Bivariate Density: Triglycerides vs. Fasting Sugar"
)
plt.xlabel("Triglyceride Level (mg/dL)")
plt.ylabel("Fasting Blood Sugar (mg/dL)")
plt.tight_layout()

plt.savefig("Graph/graph_15_hexbin_density.png")
plt.show()


# ============================================================================
# GRAPH 16: Relationship Between Diabetes and Heart Disease Status
# ============================================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Diabetes",
    hue="Heart Disease Status"
)

plt.title(
    "Graph 16: Relationship Between Diabetes and Heart Disease Status"
)
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.legend(title="Heart Disease Status")

plt.tight_layout()

plt.savefig("Graph/graph_16_diabetes_heart_disease.png")
plt.show()