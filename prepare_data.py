import pandas as pd
from pathlib import Path

# Load original training data
df = pd.read_csv("data/train.csv")

print("Data Cleaning")
print(f"Original rows: {len(df)}")

# Step 1: Identify conflicting URLs
label_counts = df.groupby("url")["label"].transform("nunique")
conflicting_mask = label_counts > 1

conflicting_rows = conflicting_mask.sum()
conflicting_urls = df.loc[conflicting_mask, "url"].nunique()

print(f"Conflicting URLs: {conflicting_urls}")
print(f"Rows affected by conflicts: {conflicting_rows}")

# Step 2: Remove all conflicting URLs
df_clean = df.loc[~conflicting_mask].copy()

# Step 3: Remove exact duplicates
rows_before = len(df_clean)

df_clean = df_clean.drop_duplicates(subset=["url"])

duplicates_removed = rows_before - len(df_clean)

print(f"Duplicate rows removed: {duplicates_removed}")

# Step 4: Verify cleaned dataset
assert df_clean["url"].is_unique
assert df_clean["url"].notna().all()
assert df_clean["label"].isin([0, 1]).all()

# Step 5: Save cleaned dataset
output_path = Path("data/train_clean.csv")

df_clean.to_csv(output_path, index=False)

print("\nSummary:")
print(f"Original rows: {len(df)}")
print(f"Conflicting rows removed: {conflicting_rows}")
print(f"Duplicate rows removed: {duplicates_removed}")
print(f"Cleaned rows: {len(df_clean)}")

print("\nCleaned Label Distribution:")
print(df_clean["label"].value_counts())

print(f"\nSaved to: {output_path}")