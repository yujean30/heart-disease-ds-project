import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 1. Ensure the output directory exists
output_dir = "Graph"
os.makedirs(output_dir, exist_ok=True)

# Set global aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 2. Load Dataset & Clean Headers
df = pd.read_csv("Preprocessing/heart_disease_cleaned_full.csv")
df.columns = df.columns.str.strip()


# Helper function for fuzzy column matching
def find_column(df_cols, keywords):
    cols_normalized = {
        col: col.lower().replace(" ", "").replace("_", "")
        for col in df_cols
    }

    for kw in keywords:
        kw_norm = kw.lower().replace(" ", "").replace("_", "")

        for orig_col, norm_col in cols_normalized.items():
            if kw_norm in norm_col:
                return orig_col

    return None


# Resolve column names dynamically
target_col = find_column(
    df.columns,
    ["heartdisease"]
)

smoking_col = find_column(
    df.columns,
    ["smoking", "smoke"]
)

bp_col = find_column(
    df.columns,
    ["bloodpressure", "bp"]
)

diabetes_col = find_column(
    df.columns,
    ["diabetes"]
)

ldl_col = find_column(
    df.columns,
    ["highldl", "ldl"]
)

hdl_col = find_column(
    df.columns,
    ["lowhdl", "hdl"]
)

sleep_col = find_column(
    df.columns,
    ["sleephours", "sleep"]
)

stress_col = find_column(
    df.columns,
    ["stresslevel", "stress"]
)

crp_col = find_column(
    df.columns,
    ["crp", "inflammatory"]
)

alcohol_col = find_column(
    df.columns,
    ["alcoholconsumption", "alcohol"]
)


# Force target to Heart Disease Status if available
if "Heart Disease Status" in df.columns:
    target_col = "Heart Disease Status"


print(
    f"Detected columns -> "
    f"Target: '{target_col}', "
    f"Smoking: '{smoking_col}', "
    f"Blood Pressure: '{bp_col}', "
    f"Diabetes: '{diabetes_col}', "
    f"LDL: '{ldl_col}', "
    f"HDL: '{hdl_col}', "
    f"Sleep: '{sleep_col}', "
    f"Stress: '{stress_col}', "
    f"CRP: '{crp_col}', "
    f"Alcohol: '{alcohol_col}'"
)


# -------------------------------------------------------------------------
# TARGET PALETTE
# -------------------------------------------------------------------------

if target_col and target_col in df.columns:

    unique_targets = df[target_col].astype(str).unique()

    color_list = ["#2ecc71", "#e74c3c"]

    palette = {
        val: color_list[i % len(color_list)]
        for i, val in enumerate(unique_targets)
    }

else:
    raise KeyError(
        "Target column for heart disease status could not be found. "
        f"Available columns: {list(df.columns)}"
    )


# =========================================================================
# GRAPH 9: Overall Population Prevalence of Clinical Risk Factors
# =========================================================================

risk_columns = [
    ("Smoking", "Smoking"),
    ("High Blood Pressure", "High Blood Pressure"),
    ("Low HDL Cholesterol", "Low HDL Cholesterol"),
    ("High LDL Cholesterol", "High LDL Cholesterol"),
    ("Diabetes", "Diabetes")
]

prevalence_data = []

for display_name, column in risk_columns:

    if column not in df.columns:
        print(f"Skipping {display_name}: column not found.")
        continue

    # Count "Yes" values
    yes_count = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .sum()
    )

    # Calculate percentage
    prevalence = (yes_count / len(df)) * 100

    prevalence_data.append({
        "Risk Factor": display_name,
        "Prevalence (%)": prevalence
    })


risk_df = pd.DataFrame(prevalence_data)

risk_df = risk_df.sort_values(
    by="Prevalence (%)",
    ascending=False
)


print("\n" + "=" * 60)
print("GRAPH 9 - RISK FACTOR PREVALENCE")
print("=" * 60)

print(risk_df.to_string(index=False))


# -------------------------------------------------------------------------
# Plot Graph 9
# -------------------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=risk_df,
    x="Prevalence (%)",
    y="Risk Factor",
    hue="Risk Factor",
    palette="rocket",
    legend=False
)

plt.title(
    "Graph 9: Overall Population Prevalence of Clinical Risk Factors"
)

plt.xlabel(
    "Prevalence Percentage (%)"
)

plt.ylabel("")

plt.xlim(0, 100)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "graph_9_clinical_risk_factors.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================================
# GRAPH 10: Stress Level and Daily Sleep Hours by Heart Disease Status
# =========================================================================

# Create Stress Level if it does not already exist
if stress_col is None and sleep_col:

    df["Stress Level"] = pd.qcut(
        df[sleep_col],
        q=3,
        labels=["High", "Medium", "Low"]
    )

    stress_col = "Stress Level"


if (
    stress_col
    and stress_col in df.columns
    and sleep_col
    and sleep_col in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.pointplot(
        data=df,
        x=stress_col,
        y=sleep_col,
        hue=target_col,
        order=["Low", "Medium", "High"],
        markers=["o", "s"],
        linestyles=["-", "--"],
        palette=palette,
        errorbar=None
    )

    plt.title(
        "Graph 10: Stress Level and Daily Sleep Hours by Heart Disease Status"
    )

    plt.xlabel("Stress Level")
    plt.ylabel("Daily Sleep (Hours)")

    plt.legend(
        title="Heart Disease Status",
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "graph_10_sleep_stress.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

else:
    print(
        "Skipping Graph 10: "
        "Sleep Hours or Stress Level column not found."
    )

# =========================================================================
# GRAPH 11: Missing Value
# =========================================================================

missing_pct = (
    df.isnull().sum() / len(df)
) * 100

missing_df = pd.DataFrame({
    "Column": missing_pct.index,
    "Missing (%)": missing_pct.values
})


print("\n" + "=" * 60)
print("GRAPH 11 - MISSING VALUE SUMMARY")
print("=" * 60)

print(missing_df.to_string(index=False))

plt.figure(figsize=(10, 5))

sns.barplot(
    data=missing_df,
    x="Column",
    y="Missing (%)",
    color="#e74c3c",
    edgecolor="black",
    linewidth=0.8
)

plt.title(
    "Graph 11: Missing Value"
)

plt.xlabel("")
plt.ylabel("Missing Values (%)")

plt.xticks(
    rotation=45,
    ha="right",
    fontsize=9
)

plt.ylim(
    0,
    max(
        1,
        missing_df["Missing (%)"].max() * 1.15
    )
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "graph_11_missing_values.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================================
# GRAPH 12: CRP Level across Alcohol Consumption Categories
# =========================================================================

if (
    alcohol_col
    and alcohol_col in df.columns
    and crp_col
    and crp_col in df.columns
):

    # Make sure CRP is numeric
    df[crp_col] = pd.to_numeric(
        df[crp_col],
        errors="coerce"
    )

    # Keep actual Alcohol Consumption values
    # and do NOT recreate them from CRP.
    alcohol_order = [
        "High",
        "Medium",
        "Low",
        "None"
    ]

    existing_alcohol = [
        category
        for category in alcohol_order
        if category in df[alcohol_col].astype(str).unique()
    ]

    # Add any other categories that may exist
    other_categories = [
        category
        for category in df[alcohol_col].dropna().astype(str).unique()
        if category not in existing_alcohol
    ]

    existing_alcohol.extend(other_categories)

    plt.figure(figsize=(8, 5))

    sns.stripplot(
        data=df,
        x=alcohol_col,
        y=crp_col,
        hue=target_col,
        order=existing_alcohol,
        dodge=True,
        jitter=0.25,
        alpha=0.5,
        palette=palette
    )

    plt.title(
        "Graph 12: CRP Level across Alcohol Consumption Categories"
    )

    plt.xlabel("Alcohol Consumption")
    plt.ylabel("CRP Level (mg/L)")

    plt.legend(
        title="Heart Disease Status",
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "graph_12_crp_alcohol.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

else:
    print(
        "Skipping Graph 12: "
        "Alcohol Consumption or CRP Level column not found."
    )


# =========================================================================
# FINAL CHECK
# =========================================================================

print("\n" + "=" * 60)
print("EXECUTION COMPLETED")
print("=" * 60)

print(f"Charts saved in: {output_dir}")

print("\nFiles generated:")

for filename in [
    "graph_9_clinical_risk_factors.png",
    "graph_10_sleep_stress.png",
    "graph_11_missing_values.png",
    "graph_12_crp_alcohol.png"
]:
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"{filename}")
    else:
        print(f"{filename}")