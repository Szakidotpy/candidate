# Take-home: phishing URL detection

Build a URL-only phishing classifier and assess whether its results justify
using it in production. We care most about your evaluation, decisions, and
ability to explain limitations.

**Time budget: at most four hours.** Prioritise the required work and describe
anything unfinished. An incomplete submission with sound reasoning is welcome.

## The data

| file | rows | columns |
|---|---|---|
| `data/train.csv` | 8,651 | `url`, `label` |
| `data/test.csv` | 2,858 | `url`, `label` |
| `data/external_test.csv` | 2,858 | `url`, `label` |

`label` is `1` for phishing and `0` for legitimate. All three files are
approximately balanced. The training collection merged multiple feeds and
has not been tidied for you. Inspect it before fitting anything.

The external file comes from a separate public collection. It holds urls and
labels only, was deduplicated with conflicting labels removed, sampled to
1,429 per class, and filtered to share no url with the other two files after
normalising case, scheme, leading `www.` and trailing slashes. That
normalisation is our rule for building it, not a preprocessing requirement
for you, and it is only an approximation of "the same page" — say what you
think it misses.

So the external file has had cleaning that `train.csv` has not. A separate
source does not guarantee independence or deployment realism.

**Reserve both evaluation files for final assessment**, and create validation
splits from `train.csv` as needed.

You do not need additional data. If you want to add some — for volume, class
balance, or a different distribution — you are welcome to, provided you:

- **exclude anything that appears in `test.csv` or `external_test.csv`.**
  Both are drawn from public collections, so a public dataset you download
  may well contain our evaluation rows. Check, and state how you checked.
- report where it came from, its size and class balance, and the overlap you
  measured;
- say what it changed. A result that got worse is a finding, not a failure.

## What to build

**1. A baseline.** Train a simple, fast, non-neural model that you could explain
to an engineer in two minutes.

**Optional:** Try one improvement motivated by your analysis. Keeping the
baseline is acceptable if you explain why; a second model is not required.

**2. An evaluation.** Evaluate on both supplied evaluation sets. Report recall,
false-positive rate, precision, and ROC-AUC, naming your operating threshold.
Measure overlap with training and explain your overlap definition.

Explain what each set tells you about future performance. Investigate whether
relationships between training and evaluation examples make the results
optimistic. You may construct an additional validation split from training.
Distinguish observations from hypotheses about why results differ.

**3. A single-URL prediction function.** Exact contract:

```python
from predict import predict

predict("https://www.PayPal-secure.login.tk/webscr?cmd=x")
# Example output, not an expected prediction:
# {"label": "phishing", "probability": 0.97}
```

Accept a raw string and return exactly these two keys. `label` is `"phishing"`
or `"clean"`, using your chosen threshold. `probability` is a finite Python
number between 0 and 1 estimating phishing probability. Explain its calibration
limitations and which population it refers to; it need not be calibrated to
deployment prevalence. Apply all required preprocessing inside the function.
It must handle unusual strings (including empty or malformed URLs) without
crashing and run on CPU. Document your treatment of invalid inputs.

**4. A reproducible submission.** Include dependency versions, any required
model artifacts, and commands to reproduce training, evaluation, and inference
on another machine. `uv` is convenient, but other working setups are welcome.
Inference should load a saved model, without training or downloading on import.

## What to write

A short report, approximately 1–2 pages, covering:

- **The result you would tell your manager.** Which metric and evaluation
  would you stand behind, and what claims do they support?
- **Two or three observations about URL characteristics**, supported by
  numbers. Include one hypothesis your analysis weakened or contradicted.
  Distinguish dataset patterns from general claims about phishing.
- **Threshold and deployment.** Assume roughly one in 1,000 incoming URLs
  is phishing. Choose a threshold using validation data and estimate deployed
  precision from measured true-positive and false-positive rates. State your
  assumptions and what the evaluation sample size allows you to conclude.
  Would you block URLs, warn users, prioritise review, or defer deployment?
  Explain what additional evidence you need; no fixed error costs are assumed.
- **Data handling.** Describe issues found, any cleaning rules, and how many
  rows they affected. Multiple treatments can be defensible.
- **Three to five consequential decisions:** observation → action → effect.
  Include unsuccessful changes if you made any; extra experiments are not
  required to fill the log.
- **Limitations and next steps**, including what you left unfinished.

## How we grade

| area | weight |
|---|---|
| Evaluation design and interpretation | 30% |
| Threshold, base rate, and uncertainty | 25% |
| Data inspection and evidence-backed observations | 20% |
| Reproducibility and prediction function | 15% |
| Modelling decisions and diagnostic reasoning | 10% |

There is no accuracy leaderboard or minimum accuracy target. We assess whether
your model behaves plausibly and whether your reasoning supports your claims.
The optional improvement earns no automatic bonus.

## Submitting

Send a repository link containing your code, `predict.py`, report, setup
instructions, and `predictions.csv` (`url,probability`) with exactly one row
per URL in `data/test.csv`. Public or private repositories are welcome; for a
private submission, contact us to arrange reviewer access or a private upload.
Do not include supplied datasets in your repository; reviewers already have them.

- **Send the link to:** Ruxandra Cuzman — ruxandra.cuzman@opswat.com
- **Questions are welcome.** Asking for clarification is not a penalty.

Use any libraries, tooling, or AI assistance you like. Be ready to explain
and defend your choices in the follow-up conversation.
