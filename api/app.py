"""Flask API serving the packaged heart-disease pipeline."""
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.config import FEATURES, LABELS

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/heart_model.joblib"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
log.info("Loaded model from %s", MODEL_PATH)


def parse_payload(payload):
    """Return (DataFrame, None) for a valid payload, or (None, error message)."""
    if not isinstance(payload, dict):
        return None, "Body must be a JSON object of feature values."
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return None, f"Missing features: {missing}"
    row = {}
    for f in FEATURES:
        try:
            row[f] = float(payload[f])
        except (TypeError, ValueError):
            return None, f"Feature '{f}' must be numeric, got {payload[f]!r}"
    return pd.DataFrame([row], columns=FEATURES), None


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/")
def index():
    return jsonify(service="heart-disease-api", endpoints=["GET /health", "POST /predict"],
                   required_features=FEATURES)


@app.post("/predict")
def predict():
    start = time.perf_counter()
    X, error = parse_payload(request.get_json(silent=True))
    if error:
        log.warning("Rejected request: %s", error)
        return jsonify(error=error), 400
    proba = model.predict_proba(X)[0]
    pred = int(proba.argmax())
    result = {
        "prediction": pred,
        "label": LABELS[pred],
        "confidence": round(float(proba[pred]), 4),
        "probability_disease": round(float(proba[1]), 4),
    }
    log.info("predict input=%s output=%s latency_ms=%.1f", X.iloc[0].to_dict(), result,
             (time.perf_counter() - start) * 1000)
    return jsonify(result)
