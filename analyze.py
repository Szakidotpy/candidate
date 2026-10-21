import pandas as pd

# Load training dataset
df = pd.read_csv("data/train.csv")

# Basic information
print("Dataset Information:")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# First five rows
print("\nDataset Preview:")
print(df.head())

# Label distribution
print("\nLabel Distribution:")
print(df["label"].value_counts(dropna=False))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Empty URLs
empty_urls = df["url"].fillna("").astype(str).str.strip().eq("").sum()
print(f"\nEmpty URLs: {empty_urls}")

# Exact duplicate rows
print("\nDuplicate Rows:")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

# Duplicate URLs
print(f"Duplicate URLs: {df.duplicated(subset=['url']).sum()}")

# Conflicting labels (both 0 and 1 for the same URL)
conflicts = (
    df.groupby("url", dropna=False)["label"]
    .nunique(dropna=False)
    .gt(1)
    .sum()
)

print(f"URLs with conflicting labels: {conflicts}")

('''
 first observations:
 rows: 8651
 4330 phising labels
 4321 legitimate labels
 no missing values
 no empty urls
 exact duplicate rows: 30
 duplicate urls: 80
 conflicting labels: 50
 ''')

# Identify URLs with conflicting labels
label_counts = df.groupby("url")["label"].nunique()

conflicting_urls = label_counts[label_counts > 1].index

# Get all affected rows
conflicting_rows = df[df["url"].isin(conflicting_urls)]

print("\nAnalyzing conflicting URLs and their labels")

print(f"Unique conflicting URLs: {len(conflicting_urls)}")

print(f"Total affected rows: {len(conflicting_rows)}")

print("\nFirst 10 conflicting URLs:")

for url in conflicting_urls[:10]:
    labels = df.loc[df["url"] == url, "label"].tolist()

    print(f"\nURL: {url}")
    print(f"Labels: {labels}")
    
('''
 Final decisions based on observations:
 delete all exact duplicate rows
 delete all rows with conflicting labels (BOTH SIDES)
 Why? Because we cannot determine the correct label for these URLs, therefore, we cannot judge them based on the labels provided.
    The model will be trained on the remaining data, which has clear labels and no conflicts.
 This way it is much safer, as for these conflicting URLs, we cannot determine the correct label,
    and it is better to remove them from the training set to avoid introducing noise into the model.
    
Logical model to use: TF-IDF with Logistic Regression
 ''')