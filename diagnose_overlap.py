
import re
import joblib
import pandas as pd

from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_auc_score


# 1. Configuration

TRAIN_PATH = "data/train_clean.csv"
MODEL_PATH = "artifacts/model.joblib"

DATASETS = {
    "TEST": "data/test.csv",
    "EXTERNAL TEST": "data/external_test.csv"
}


# 2. URL utilities

def get_hostname(url):
    text = str(url).strip()

    if "://" not in text:
        text = "http://" + text

    try:
        return (urlsplit(text).hostname or "").lower()
    except ValueError:
        return ""


def normalize_url(url):
    text = str(url).strip().lower()
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)

    if text.startswith("www."):
        text = text[4:]

    return text.rstrip("/")


# 3. Load trained model

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle["threshold"]

print(f"Loaded model. Threshold: {threshold}")


# 4. Reconstruct the actual fitted training subset

train_df = pd.read_csv(TRAIN_PATH)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

groups = train_df["url"].apply(get_hostname)

train_idx, _ = next(
    splitter.split(
        train_df["url"],
        train_df["label"],
        groups=groups
    )
)

fitted_train = train_df.iloc[train_idx]

train_urls = set(
    fitted_train["url"].apply(normalize_url)
)

train_hosts = set(
    fitted_train["url"].apply(get_hostname)
)

train_hosts.discard("")


# 5. Metrics helper

def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def evaluate_group(df, name):
    print(f"\n--- {name} ---")
    print(f"Samples: {len(df)}")

    if df.empty:
        return

    X = df["url"].astype(str)
    y = df["label"]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None

    precision = (
        tp / (tp + fp) if tp + fp else None
    )

    auc = (
        roc_auc_score(y, probabilities)
        if y.nunique() == 2 else None
    )

    print(f"Legitimate: {tn + fp}")
    print(f"Phishing: {tp + fn}")

    print(f"Recall: {format_metric(recall)}")
    print(f"FPR: {format_metric(fpr)}")
    print(f"Precision: {format_metric(precision)}")
    print(f"ROC-AUC: {format_metric(auc)}")

    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")


# 6. Diagnostic comparison

for dataset_name, dataset_path in DATASETS.items():

    df = pd.read_csv(dataset_path)

    normalized_urls = df["url"].apply(normalize_url)
    hostnames = df["url"].apply(get_hostname)

    seen_url = normalized_urls.isin(train_urls)
    seen_host = hostnames.isin(train_hosts)

    # Mutually exclusive categories
    url_overlap = seen_url
    hostname_only = ~seen_url & seen_host
    unseen_hostname = ~seen_url & ~seen_host

    print(f"\n{'=' * 50}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 50}")

    print(f"Normalized URL overlap: {url_overlap.sum()}")
    print(f"Hostname-only overlap: {hostname_only.sum()}")
    print(f"Unseen hostname: {unseen_hostname.sum()}")

    assert (
        url_overlap.sum()
        + hostname_only.sum()
        + unseen_hostname.sum()
    ) == len(df)

    evaluate_group(
        df.loc[url_overlap],
        "SEEN URL"
    )

    evaluate_group(
        df.loc[hostname_only],
        "SEEN HOSTNAME ONLY"
    )

    evaluate_group(
        df.loc[unseen_hostname],
        "UNSEEN HOSTNAME"
    )


print("\nDiagnostic analysis completed.")


('''
 Loaded model. Threshold: 0.8

==================================================
DATASET: TEST
==================================================
Normalized URL overlap: 120
Hostname-only overlap: 678
Unseen hostname: 2060

--- SEEN URL ---
Samples: 120
Legitimate: 1
Phishing: 119
Recall: 0.4958
FPR: 0.0000
Precision: 1.0000
ROC-AUC: 1.0000
TP: 59 | FP: 0 | TN: 1 | FN: 60

--- SEEN HOSTNAME ONLY ---
Samples: 678
Legitimate: 315
Phishing: 363
Recall: 0.7190
FPR: 0.0000
Precision: 1.0000
ROC-AUC: 0.9977
TP: 261 | FP: 0 | TN: 315 | FN: 102

--- UNSEEN HOSTNAME ---
Samples: 2060
Legitimate: 1113
Phishing: 947
Recall: 0.4530
FPR: 0.0027
Precision: 0.9931
ROC-AUC: 0.9563
TP: 429 | FP: 3 | TN: 1110 | FN: 518

==================================================
DATASET: EXTERNAL TEST
==================================================
Normalized URL overlap: 0
Hostname-only overlap: 17
Unseen hostname: 2841

--- SEEN URL ---
Samples: 0

--- SEEN HOSTNAME ONLY ---
Samples: 17
Legitimate: 5
Phishing: 12
Recall: 0.5833
FPR: 0.0000
Precision: 1.0000
ROC-AUC: 1.0000
TP: 7 | FP: 0 | TN: 5 | FN: 5

--- UNSEEN HOSTNAME ---
Samples: 2841
Legitimate: 1424
Phishing: 1417
Recall: 0.1708
FPR: 0.0000
Precision: 1.0000
ROC-AUC: 0.9093
TP: 242 | FP: 0 | TN: 1424 | FN: 1175

Diagnostic analysis completed.
''')