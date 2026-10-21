import pandas as pd
from urllib.parse import urlsplit


# 1 Load cleaned training data

df = pd.read_csv("data/train_clean.csv")

print("Phishing URL Detection")
print(f"Total rows: {len(df)}")


# 2 Feature Extraction

# Length of the URL
df["url_length"] = df["url"].str.len()

# Number of digits
df["digit_count"] = df["url"].str.count(r"\d")

# Number of hyphens
df["hyphen_count"] = df["url"].str.count("-")

# Number of dots
df["dot_count"] = df["url"].str.count(r"\.")

# Does the URL contain @?
df["has_at"] = df["url"].str.contains("@", regex=False)

# Does the URL contain a query string marker?
df["has_query"] = df["url"].str.contains("?", regex=False)

# Does the URL use HTTPS?
df["uses_https"] = (
    df["url"].str.lower().str.startswith("https://")
)


# 3 Numerical Features

print("\nNumerical Features by Class")

numeric_features = [
    "url_length",
    "digit_count",
    "hyphen_count",
    "dot_count"
]

summary = df.groupby("label")[numeric_features].agg(
    ["mean", "median"]
).round(2)

print(summary.to_string())


# 4 Boolean Features

print("\nBoolean Features (% of URLs)")

boolean_features = [
    "has_at",
    "has_query",
    "uses_https"
]

for feature in boolean_features:

    percentages = (
        df.groupby("label")[feature].mean() * 100
    ).round(2)

    print(f"\n{feature}:")
    print(percentages.to_string())


# 5 URL Length Distribution

print("\nURL Length Distribution")

for label in [0, 1]:

    subset = df[df["label"] == label]

    name = "PHISHING" if label == 1 else "CLEAN"

    print(f"\n{name}")
    print(f"Mean: {subset['url_length'].mean():.2f}")
    print(f"Median: {subset['url_length'].median():.2f}")
    print(f"90th percentile: {subset['url_length'].quantile(0.9):.2f}")
    print(f"Maximum: {subset['url_length'].max()}")


# 6 Domain Analysis

def extract_hostname(url):

    try:
        text = str(url).strip()

        if "://" not in text:
            text = "http://" + text

        return (urlsplit(text).hostname or "").lower()

    except ValueError:
        return ""


df["hostname"] = df["url"].apply(extract_hostname)

print("\nDomain Analysis")

print(f"Unique hostnames: {df['hostname'].nunique()}")

print(
    "URLs with unparsed hostnames:",
    (df["hostname"] == "").sum()
)

print("\nTop 10 most frequent hostnames:")

print(
    df.loc[df["hostname"] != "", "hostname"]
    .value_counts()
    .head(10)
    .to_string()
)


# 7 Hostname Analysis

valid_hosts = df[df["hostname"] != ""]

mixed_hosts = (
    valid_hosts.groupby("hostname")["label"]
    .nunique()
)

print("\nHostnames appearing in both classes:")
print((mixed_hosts > 1).sum())


print("\n=== EDA COMPLETED ===")

('''
 Results:
Phishing URL Detection
Total rows: 8521

Numerical Features by Class
      url_length            digit_count                 hyphen_count                dot_count       
            mean   median            mean  median               mean median             mean median
label                                                                          
0          47.91   42.0              1.62    0.0                1.22    0.0             2.21    2.0
1          75.85   55.0              9.75    2.0                0.80    0.0             2.78    2.0

Boolean Features (% of URLs)

has_at:
label
0    0.00
1    4.55

has_query:
label
0     3.41
1    24.13

uses_https:
label
0    44.02
1    32.97

URL Length Distribution

CLEAN
Mean: 47.91
Median: 42.00
90th percentile: 80.00
Maximum: 557

PHISHING
Mean: 75.85
Median: 55.00
90th percentile: 129.00
Maximum: 1641

nDomain Analysis
Unique hostnames: 6310
URLs with unparsed hostnames: 0

Top 10 most frequent hostnames:
hostname
support-appleld.com.secureupdate.duilawyeryork.com    161
en.wikipedia.org                                      106
secureupdate.appleld.com.duilawyeryork.com             56
www.thefreedictionary.com                              33
en.academic.ru                                         33
drive.google.com                                       31
docs.google.com                                        29
www.youtube.com                                        28
shadetreetechnology.com                                28
www.facebook.com                                       27

Hostnames appearing in both classes:
7

=== EDA COMPLETED ===


Observations:
- Phishing URLs tend to be longer than clean URLs.
- Phishing URLs have more digits on average.
- Phishing URLs are more likely to contain a query string.
- Phishing URLs tend to contain more hyphens than legitimate URLs.
- A small percentage of phishing URLs contain the "@" symbol.
- Clean URLs are more likely to use HTTPS than phishing URLs.

Important Note: There are 8521 URLs, but only 6310 unique hostnames. If we split the dataset for training and validation randomly,
    there could be very similar URLs in both sets, which may lead to data leakage. Consider using GroupShuffleSplit.
 ''')