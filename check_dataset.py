import pandas as pd
import numpy as np

df = pd.read_csv("heart_disease.csv")

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