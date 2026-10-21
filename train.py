import pandas as pd

from urllib.parse import urlsplit

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


# 1. Load cleaned training data

df = pd.read_csv("data/train_clean.csv")

X = df["url"]
y = df["label"]


# 2. Extract hostnames

def get_hostname(url):

    text = str(url).strip()

    if "://" not in text:
        text = "http://" + text

    try:
        return (urlsplit(text).hostname or "").lower()
    except ValueError:
        return ""


groups = X.apply(get_hostname)


# 3. Train-validation split using GroupShuffleSplit to avoid data leakage

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print("Data Split:")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nValidation label distribution:")
print(y_val.value_counts())

train_hosts = set(groups.iloc[train_idx])
val_hosts = set(groups.iloc[val_idx])

print(
    f"\nOverlapping hostnames: "
    f"{len(train_hosts & val_hosts)}"
)


# 4. Baseline Model

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


# 5. Train the model

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# 6. Validation Predictions

y_prob = model.predict_proba(X_val)[:, 1]

threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)


# 7. Evaluation

precision = precision_score(
    y_val, y_pred, zero_division=0
)

recall = recall_score(
    y_val, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_val, y_prob
)

tn, fp, fn, tp = confusion_matrix(
    y_val, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0.0


print("\nBaseline Model Evaluation:")

print(f"Threshold: {threshold}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")


# 8. Threshold Analysis
# Evaluate different operating thresholds using validation data.
# Assume phishing prevalence of 1 in 1000 incoming URLs.

prevalence = 0.001

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

print("\nThreshold Analysis:")

print(
    f"{'Threshold':<12}"
    f"{'Recall':<10}"
    f"{'FPR':<10}"
    f"{'Val. Prec.':<12}"
    f"{'Deploy Prec.':<14}"
    f"{'FP':<6}"
    f"{'TP':<6}"
)

for threshold in thresholds:

    # Apply the current threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    # Recall / True Positive Rate
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else float("nan")
    )

    # False Positive Rate
    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else float("nan")
    )

    # Precision measured on validation data
    val_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else float("nan")
    )

    # Expected precision at deployment prevalence
    denominator = (
        recall * prevalence
        + fpr * (1 - prevalence)
    )

    deployed_precision = (
        (recall * prevalence) / denominator
        if denominator > 0 else float("nan")
    )

    print(
        f"{threshold:<12.2f}"
        f"{recall:<10.4f}"
        f"{fpr:<10.4f}"
        f"{val_precision:<12.4f}"
        f"{deployed_precision:<14.4f}"
        f"{fp:<6}"
        f"{tp:<6}"
    )


print("\nThreshold analysis completed.")


# 9. Save the trained model and selected threshold

from pathlib import Path
import joblib

selected_threshold = 0.80

Path("artifacts").mkdir(exist_ok=True)

model_bundle = {
    "model": model,
    "threshold": selected_threshold
}

joblib.dump(model_bundle, "artifacts/model.joblib")

print("\nModel saved to artifacts/model.joblib")
print(f"Selected threshold: {selected_threshold}")


('''
Data Split:
Training samples: 6903
Validation samples: 1618

Training label distribution:
label
0    3511
1    3392
Name: count, dtype: int64

Validation label distribution:
label
1    872
0    746
Name: count, dtype: int64

Overlapping hostnames: 0

Training model...
Training completed.

Baseline Model Evaluation:
Threshold: 0.5
Precision: 0.8948
Recall: 0.8876
False Positive Rate: 0.1220
ROC-AUC: 0.9608

Confusion Matrix:
True Positives: 774
False Positives: 91
True Negatives: 655
False Negatives: 98

Not nescessarily the worst outcomes, but for a large dataset, or a real case, this would be a problem. 
We need a baseline analysis to see what we can do to improve the model.

Threshold Analysis:

Threshold   Recall    FPR       Val. Prec.  Deploy Prec.  FP    TP    
0.50        0.8876    0.1220    0.8948      0.0072        91    774   
0.60        0.8211    0.0442    0.9559      0.0182        33    716   
0.70        0.7156    0.0094    0.9889      0.0709        7     624   
0.80        0.4920    0.0013    0.9977      0.2687        1     429   
0.90        0.2523    0.0000    1.0000      1.0000        0     220   
0.95        0.1204    0.0000    1.0000      1.0000        0     105   
0.97        0.0505    0.0000    1.0000      1.0000        0     44    
0.99        0.0034    0.0000    1.0000      1.0000        0     3  


A threshold of 0.80 will be selected using the validation set as a provisional operating point for prioritising suspicious URLs for review.
At this threshold, the model achieves 49.20% recall and a measured false-positive rate of approximately 0.13%.
Assuming a phishing prevalence of 0.1% and unchanged error rates, the estimated deployment precision is 26.87%.
However, this estimate is highly uncertain because only one false positive was observed among 746 legitimate validation URLs.
The model is not yet suitable for automatic blocking without further independent evaluation.
 ''')