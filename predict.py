
import math
from functools import lru_cache
from pathlib import Path

import joblib


# 1. Model configuration

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "model.joblib"
)


# 2. Load saved model
# The model is loaded once, on the first prediction.
# No training or downloading happens on import.

@lru_cache(maxsize=1)
def load_model():
    bundle = joblib.load(MODEL_PATH)

    return bundle["model"], bundle["threshold"]


# 3. Single URL prediction

def predict(url: str) -> dict:
    """
    Predict whether a URL is phishing or clean.

    Args:
        url: Raw URL string.

    Returns:
        {
            "label": "phishing" or "clean",
            "probability": float
        }

    Notes:
        - Uses a saved CPU-compatible model.
        - Accepts empty and malformed URL strings.
        - Does not contact or validate websites.
        - Probabilities are model estimates and
          are not calibrated to deployment prevalence.
    """

    if not isinstance(url, str):
        raise TypeError("url must be a string")

    model, threshold = load_model()

    # Apply the same vectorization used during training.
    # The saved pipeline contains the TF-IDF preprocessing.
    probability = float(
        model.predict_proba([url])[0, 1]
    )

    if not math.isfinite(probability):
        raise ValueError("Non-finite model probability")

    probability = max(0.0, min(1.0, probability))

    label = (
        "phishing"
        if probability >= threshold
        else "clean"
    )

    return {
        "label": label,
        "probability": probability
    }


# 4. Manual test examples

if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "https://www.PayPal-secure.login.tk/webscr?cmd=x",
        "",
        "not-a-valid-url",
        "://malformed"
    ]

    for url in test_urls:
        print(f"\nURL: {repr(url)}")
        print(f"Prediction: {predict(url)}")
