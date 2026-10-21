# URL-only phishing detection, Technical report


## Executive recommendation

Do not deploy this classifier as an automatic URL blocker. At the preselected **0.80 threshold** it detects **249 of 1,429** phishing URLs in the external evaluation set (**17.42% recall**), missing **1,180**. Its external **ROC-AUC is 0.9101**, indicating some ranking value, but zero observed false positives in 1,429 legitimate URLs is not evidence of zero real-world false-positive risk. I would defer production deployment and consider only a supervised review-prioritization pilot after collecting representative traffic and estimating review costs.

## Data quality and URL observations

The supplied training set contained **8,651 rows**. I removed **30 identical duplicate rows** and both records for each of **50 URLs with conflicting labels (100 rows)**, retaining **8,521** unique, consistently labelled URLs. Both supplied evaluation sets (**2,858 rows each**) were left unchanged. No external data was added.

In the cleaned training set, phishing URLs were longer (**mean 75.85 vs 47.91** characters; **median 55 vs 42**), contained more digits on average (**9.75 vs 1.62**), and more often contained a query marker (**24.13% vs 3.41%**). My initial hypothesis that phishing URLs would contain more hyphens was contradicted by the means (**0.80 phishing vs 1.22 legitimate**); both medians were zero. These are dataset associations, not universal phishing rules.

## Model and validation protocol

Baseline: character-level **TF-IDF** (3–5-grams, `min_df=2`, sublinear TF) followed by **Logistic Regression**; CPU-only, no neural model. Hostname-grouped splitting with seed 42 produced **6,903 fitting** and **1,618 validation** rows, with zero exact hostname overlap between the two subsets. The operating threshold **0.80** was selected on validation only; the supplied test sets were reserved for final evaluation. A second model was not prioritized over careful diagnostics and reproducibility.

## Final evaluation at threshold 0.80

| Dataset | Recall | FPR | Precision | ROC-AUC | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Validation (n=1,618) | 49.20% | 0.134% | 99.77% | 0.9608 | 429 / 1 / 443 |
| Test (n=2,858) | 52.41% | 0.210% | 99.60% | 0.9717 | 749 / 3 / 680 |
| External (n=2,858) | 17.42% | 0.000% | 100%* | 0.9101 | 249 / 0 / 1,180 |

*100% is the observed positive predictive value on the balanced external sample, not a deployment guarantee. FPR for external evaluation is zero observed false positives among 1,429 negatives.*

## Threshold, prevalence and uncertainty

At 0.50 the grouped validation false-positive rate was **12.20%** (recall **88.76%**). Increasing the threshold to 0.80 reduced false positives from **91 to 1**, but also reduced true positives from **774 to 429**. Threshold 0.90 produced zero observed false positives yet only **25.23% recall**; this is not automatically safer in operation. The 0.80 point is a provisional trade-off for review prioritization, not a validated blocking threshold.

Assuming phishing prevalence π = 0.001 and unchanged class-conditional error rates, expected deployed precision is **Recall × π / [Recall × π + FPR × (1 − π)]**. This gives **26.87%** using validation and **19.99%** using the standard test. Plugging in external FPR = 0 produces 100%, but this is an unstable estimate: with **0/1,429** false positives, the one-sided 95% binomial upper bound on FPR is about **0.21%** (under independent representative sampling). The test set’s **3/1,429** false positives give an upper bound of about **0.54%**. Balanced public datasets and uncalibrated model scores do not establish performance or probabilities for a 0.1%-prevalence deployment population.

## Train–evaluation relationships

Overlap was assessed against the actual **6,903 fitted training rows**. Normalization lowercased URLs and removed the scheme, leading `www.` and trailing slashes. The standard test contained **120 normalized URL matches** and **766 hostname matches**; the external set had **0** and **17**, respectively. These counts use separate definitions and may overlap. Standard-test recall was **71.90%** on nonidentical URLs from seen hostnames versus **45.30%** on unseen hostnames. External unseen-hostname recall was only **17.08%**. This is consistent with sample familiarity contributing to optimistic test results, but does not prove causality or fully explain the external gap.

The overlap rule misses equivalent URLs involving percent encoding, redirects, default ports or alternate hostnames, and case-folding can incorrectly merge case-sensitive paths. Exact hostname grouping also misses relationships among distinct subdomains of the same registered domain.

## Decision log: observation → action → effect

1. **Label quality:** Conflicting labels and duplicates → remove affected rows → 8,521 consistent training rows; no measured claim that cleaning improved accuracy.
2. **Validation leakage:** Repeated hostnames → use `GroupShuffleSplit` → zero exact hostname overlap inside validation, though related registered domains can remain.
3. **Operating point:** 0.50 produced 12.20% validation FPR → select 0.80 on validation → 0.134% measured FPR with reduced recall (49.20%).
4. **Generalization:** External recall far below standard-test recall → audit URL/hostname overlap and unseen-hostname performance → identify a substantial unresolved cross-source generalization gap; no post-test retuning.

## Limitations and next steps

URL text alone excludes page content, redirects and runtime context. Small numbers of negative errors create wide uncertainty; probability calibration under deployment prevalence was not assessed. I did not complete temporal or registered-domain-held-out evaluation, prospective production testing, or an optional second model. Before deployment, obtain temporally newer and prevalence-representative URLs, verify labels, quantify alert/review costs, monitor drift and calibration, then choose a threshold on fresh validation data without reusing these final test sets.
