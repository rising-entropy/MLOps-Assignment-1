"""Flask API serving the packaged heart-disease pipeline, with logging and Prometheus metrics."""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.config import FEATURES, LABELS

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/heart_model.joblib"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])
CONFIDENCE = Histogram("api_prediction_confidence", "Confidence of predicted class",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
INVALID = Counter("api_invalid_requests_total", "Requests rejected by validation")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def jlog(level, event, **fields):
    """One JSON object per line so Loki/Grafana can parse the fields."""
    log.log(level, json.dumps({"event": event, **fields}, default=str))


jlog(logging.INFO, "model_loaded", path=str(MODEL_PATH))


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


@app.before_request
def start_timer():
    g.start = time.perf_counter()


@app.after_request
def record_request(resp):
    endpoint = request.url_rule.rule if request.url_rule else "unmatched"
    elapsed = time.perf_counter() - g.start
    if endpoint != "/metrics":
        REQUESTS.labels(request.method, endpoint, resp.status_code).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        jlog(logging.INFO, "request", method=request.method, endpoint=endpoint,
             status=resp.status_code, latency_ms=round(elapsed * 1000, 2),
             client=request.remote_addr)
    return resp


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.get("/")
def index():
    return jsonify(service="heart-disease-api", endpoints=["GET /health", "POST /predict",
                                                           "GET /metrics"],
                   required_features=FEATURES)


@app.post("/predict")
def predict():
    X, error = parse_payload(request.get_json(silent=True))
    if error:
        INVALID.inc()
        jlog(logging.WARNING, "invalid_request", error=error)
        return jsonify(error=error), 400
    proba = model.predict_proba(X)[0]
    pred = int(proba.argmax())
    result = {
        "prediction": pred,
        "label": LABELS[pred],
        "confidence": round(float(proba[pred]), 4),
        "probability_disease": round(float(proba[1]), 4),
    }
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    jlog(logging.INFO, "prediction", input=X.iloc[0].to_dict(), **result)
    return jsonify(result)
