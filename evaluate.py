
import re
import joblib
import numpy as np
import pandas as pd

from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score
)
from scipy.stats import beta


# 1. Configuration

MODEL_PATH = "artifacts/model.joblib"
TRAIN_PATH = "data/train_clean.csv"
TEST_PATH = "data/test.csv"
EXTERNAL_PATH = "data/external_test.csv"

DEPLOYMENT_PREVALENCE = 0.001


# 2. URL helpers

def get_hostname(url):
    text = str(url).strip()

    if "://" not in text:
        text = "http://" + text

    try:
        return (urlsplit(text).hostname or "").lower()
    except ValueError:
        return ""


def normalize_url(url):
    """
    Approximate URL normalization for overlap detection:
    - lowercase
    - remove scheme
    - remove leading www.
    - remove trailing slashes

    This does not guarantee page equivalence.
    """
    text = str(url).strip().lower()
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)

    if text.startswith("www."):
        text = text[4:]

    return text.rstrip("/")


# 3. Load saved model

print("Loading model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle["threshold"]

print("Model loaded successfully.")
print(f"Selected threshold: {threshold}")


# 4. Reconstruct actual training subset
# Same cleaned data, grouping and random seed as train.py.
# No model fitting takes place here.

train_df = pd.read_csv(TRAIN_PATH)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

groups = train_df["url"].apply(get_hostname)

train_idx, val_idx = next(
    splitter.split(
        train_df["url"],
        train_df["label"],
        groups=groups
    )
)

fitted_train = train_df.iloc[train_idx]

training_urls = set(
    fitted_train["url"].apply(normalize_url)
)

training_hosts = set(
    fitted_train["url"].apply(get_hostname)
)
training_hosts.discard("")

print(f"Actual model training rows: {len(fitted_train)}")


# 5. Overlap analysis

def analyze_overlap(df, name):

    print(f"\n{name}: Overlap Analysis:")

    normalized = df["url"].apply(normalize_url)
    hostnames = df["url"].apply(get_hostname)

    exact_overlap = normalized.isin(training_urls)

    hostname_overlap = (
        hostnames.isin(training_hosts) & (hostnames != "")
    )

    print(f"Total URLs: {len(df)}")
    print(f"Normalized URL overlaps: {exact_overlap.sum()}")
    print(f"Hostname overlaps: {hostname_overlap.sum()}")

    print(
        f"Hostname overlap rate: "
        f"{hostname_overlap.mean() * 100:.2f}%"
    )

    return hostname_overlap


# 6. Evaluation

def evaluate_dataset(df, name):

    print(f"\n{name}: Model Evaluation")

    X = df["url"].fillna("").astype(str)
    y_true = df["label"]

    if not y_true.isin([0, 1]).all():
        raise ValueError("Unexpected labels in evaluation data")

    y_prob = model.predict_proba(X)[:, 1]

    if not np.isfinite(y_prob).all():
        raise ValueError("Model returned non-finite probabilities")

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    fpr = fp / (fp + tn)

    auc = roc_auc_score(y_true, y_prob)

    # Estimate precision under real-world class prevalence
    denominator = (
        recall * DEPLOYMENT_PREVALENCE
        + fpr * (1 - DEPLOYMENT_PREVALENCE)
    )

    deployed_precision = (
        recall * DEPLOYMENT_PREVALENCE / denominator
        if denominator > 0 else float("nan")
    )

    # One-sided 95% binomial upper bound for FPR.
    # Assumes independent, representative negative examples.
    negatives = fp + tn

    fpr_upper = (
        beta.ppf(0.95, fp + 1, tn)
        if fp < negatives else 1.0
    )

    print(f"Samples: {len(df)}")
    print(f"Threshold: {threshold}")

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")

    print("\nDeployment estimate (0.1% prevalence):")
    print(f"Estimated precision: {deployed_precision:.4f}")

    print("\nUncertainty:")
    print(f"FPR one-sided 95% upper bound: {fpr_upper:.4f}")

    if fp == 0:
        print(
            "WARNING: Zero observed false positives "
            "does not prove zero deployment FPR."
        )

    return y_prob


# 7. Load evaluation datasets

test_df = pd.read_csv(TEST_PATH)
external_df = pd.read_csv(EXTERNAL_PATH)


# 8. Test set

test_overlap = analyze_overlap(test_df, "TEST")

test_probabilities = evaluate_dataset(
    test_df, "TEST"
)


# 9. External test set

external_overlap = analyze_overlap(
    external_df, "EXTERNAL TEST"
)

external_probabilities = evaluate_dataset(
    external_df, "EXTERNAL TEST"
)


# 10. Export required predictions.csv
# Keep exactly the original test URL order.

predictions = pd.DataFrame({
    "url": test_df["url"],
    "probability": test_probabilities.astype(float)
})

assert len(predictions) == len(test_df)
assert list(predictions.columns) == ["url", "probability"]

predictions.to_csv("predictions.csv", index=False)

print("\nExport:")
print("Saved predictions.csv")
print(f"Rows exported: {len(predictions)}")

print("\nEvaluation completed.")



('''
 Loading model...
Model loaded successfully.
Selected threshold: 0.8
Actual model training rows: 6903

TEST: Overlap Analysis:
Total URLs: 2858
Normalized URL overlaps: 120
Hostname overlaps: 766
Hostname overlap rate: 26.80%

TEST: Model Evaluation
Samples: 2858
Threshold: 0.8

Precision: 0.9960
Recall: 0.5241
FPR: 0.0021
ROC-AUC: 0.9717

Confusion Matrix:
TP: 749
FP: 3
TN: 1426
FN: 680

Deployment estimate (0.1% prevalence):
Estimated precision: 0.1999

Uncertainty:
FPR one-sided 95% upper bound: 0.0054

EXTERNAL TEST: Overlap Analysis:
Total URLs: 2858
Normalized URL overlaps: 0
Hostname overlaps: 17
Hostname overlap rate: 0.59%

EXTERNAL TEST: Model Evaluation
Samples: 2858
Threshold: 0.8

Precision: 1.0000
Recall: 0.1742
FPR: 0.0000
ROC-AUC: 0.9101

Confusion Matrix:
TP: 249
FP: 0
TN: 1429
FN: 1180

Deployment estimate (0.1% prevalence):
Estimated precision: 1.0000

Uncertainty:
FPR one-sided 95% upper bound: 0.0021
WARNING: Zero observed false positives does not prove zero deployment FPR.

Export:
Saved predictions.csv
Rows exported: 2858

Evaluation completed.
''')