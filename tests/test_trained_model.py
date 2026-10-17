"""Checks on the saved model artifact (run in CI right after training)."""
import json

import joblib
import numpy as np
import pytest

from src import config

MODEL = config.MODEL_DIR / "heart_model.joblib"
META = config.MODEL_DIR / "model_metadata.json"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="no trained model yet")


def test_metadata_matches_feature_contract():
    meta = json.loads(META.read_text())
    assert meta["input_features"] == config.FEATURES
    assert len(meta["selected_features"]) >= 1


def test_model_quality_gate():
    """Fail the pipeline if a retrain regresses below an agreed floor."""
    meta = json.loads(META.read_text())
    assert meta["cv_metrics"]["roc_auc"] >= 0.85
    assert meta["test_metrics"]["roc_auc"] >= 0.85
    assert meta["test_metrics"]["recall"] >= 0.80


def test_saved_model_scores_raw_records(clean_df):
    model = joblib.load(MODEL)
    proba = model.predict_proba(clean_df[config.FEATURES])[:, 1]
    assert proba.shape == (len(clean_df),) and not np.isnan(proba).any()
    # sanity: sick patients should score higher on average than healthy ones
    sick = clean_df[config.TARGET] == 1
    assert proba[sick].mean() > proba[~sick].mean()
