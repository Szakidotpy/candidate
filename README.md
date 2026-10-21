# Phishing URL Detection, OPSWAT Take-Home Assignment

A URL-only phishing detection baseline developed using Python, character-level TF-IDF and Logistic Regression.

The project focuses on data quality, evaluation design, generalization, threshold selection and deployment limitations rather than maximizing classification accuracy.

## Key Findings

At the selected threshold of 0.80:

| Metric | Test | External Test |
|---|---:|---:|
| Recall | 52.41% | 17.42% |
| False Positive Rate | 0.21% | 0.00% |
| Precision | 99.60% | 100.00% |
| ROC-AUC | 0.9717 | 0.9101 |

The substantial recall reduction on the external dataset highlights generalization limitations. The model is not recommended for automatic blocking.

Observed precision on approximately balanced evaluation datasets does not represent expected precision under real-world phishing prevalence.

## Documentation

- [Setup and Reproducibility](SETUP.md)
- [Technical Report](reports/Report.md)
- [Original Assignment](readme/README.md)

## Inference

```python
from predict import predict

result = predict("https://example.com")
print(result)
```

The model is loaded from `artifacts/model.joblib`. Inference runs on CPU without retraining.

## Dataset Availability

The original datasets are intentionally excluded from this repository, as requested in the assignment. Reviewers can use their supplied datasets to reproduce the results.

## Implementation

Python, pandas, NumPy, scikit-learn, SciPy and joblib.
