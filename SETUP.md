# Setup and Reproducibility

## Project Overview

This project implements a URL-only phishing detection baseline using character-level TF-IDF features and Logistic Regression.

The objective is to evaluate the model's generalization performance and assess whether its results justify deployment in a real-world cybersecurity environment.

The implementation uses scikit-learn and runs on CPU without requiring external services or additional datasets.

## Project Structure

```text
candidate/
├── data/                      # Provided datasets (not committed)
│   ├── train.csv
│   ├── train_clean.csv
│   ├── test.csv
│   └── external_test.csv
├── artifacts/
│   └── model.joblib
├── reports/
│   └── report.md
├── analyze.py
├── prepare_data.py
├── eda.py
├── train.py
├── evaluate.py
├── diagnose_overlap.py
├── predict.py
├── predictions.csv
├── requirements.txt
├── SETUP.md
└── .gitignore
```

## Requirements

- Python 3, compatible with the pinned dependencies
- CPU-based execution
- Dependencies listed in `requirements.txt`

No GPU, model downloads, or network access are required for training or inference once the dependencies are installed.

## Installation

Clone the repository and navigate to the project root.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

Place the three original, supplied CSV files inside the `data/` directory:

- `train.csv`
- `test.csv`
- `external_test.csv`

The supplied datasets are deliberately excluded from version control, as required by the assignment.

No additional external training data was used.

## Reproducing the Results

Run the following commands from the project root.

**Step 1 — Data cleaning**

```bash
python prepare_data.py
```

This creates `data/train_clean.csv`.

The cleaning procedure removes exact duplicate records and URLs associated with conflicting labels.

**Step 2 — Exploratory analysis (optional)**

```bash
python eda.py
```

This reports statistical differences between legitimate and phishing URLs.

**Step 3 — Model training**

```bash
python train.py
```

This script performs a hostname-grouped training/validation split using a fixed random seed.

It trains a character-level TF-IDF + Logistic Regression baseline, evaluates validation performance, analyzes operating thresholds, and saves the trained pipeline and selected threshold to `artifacts/model.joblib`.

The selected operating threshold is 0.80.

**Step 4 — Final evaluation**

```bash
python evaluate.py
```

This evaluates the saved model against both provided evaluation datasets.

Reported metrics include precision, recall, false-positive rate, ROC-AUC, confusion matrix counts, and estimated deployment precision under a 0.1% phishing prevalence assumption.

It also measures normalized URL and hostname overlap with the fitted training subset.

The script generates `predictions.csv`, containing `url` and `probability` in the original test-set order.

**Step 5 — Additional overlap diagnostics (optional)**

```bash
python diagnose_overlap.py
```

This compares model performance across previously seen normalized URLs, previously seen hostnames, and unseen hostnames. These diagnostics do not alter the trained model or operating threshold.

## Single-URL Inference

The prediction function can be imported directly:

```python
from predict import predict

result = predict(
    "https://www.PayPal-secure.login.tk/webscr?cmd=x"
)

print(result)
```

The function returns exactly two fields:

```python
{
    "label": "phishing",
    "probability": 0.88
}
```

The output above is illustrative.

The `label` is determined using the saved 0.80 threshold. The `probability` is a finite Python float between 0 and 1.

Inference loads the saved model lazily and does not trigger training or downloads.

### Invalid URL Handling

The classifier operates on raw URL strings.

Empty strings and malformed URLs are accepted and scored as text without attempting to open or validate websites.

This avoids crashes but does not establish whether a supplied URL is syntactically valid or safe.

A `clean` prediction means only that the model's estimated phishing score is below the operating threshold.

Non-string inputs raise a `TypeError`.

## Model Limitations

The model uses URL text only and cannot inspect webpage content, redirects, certificates, or other runtime security signals.

Its probability outputs come from a classifier trained on approximately balanced data and have not been independently calibrated for real-world deployment prevalence.

The validation and evaluation datasets are also approximately balanced, unlike the assumed 0.1% phishing prevalence in deployment.

Observed differences between the standard and external evaluation sets indicate generalization limitations.

The selected threshold is provisional and does not justify automatic blocking without further representative evaluation.

For the complete analysis, methodology, decision log, uncertainty discussion, and deployment recommendation, see `reports/Report.md`.
