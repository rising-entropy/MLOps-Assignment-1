"""Training utilities, MLflow logging and packaging (tiny grid, isolated paths)."""
import json

import joblib
import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config, features, train

TINY_GRID = {"fe__add_features": [True], "select__k": ["all"]}


@pytest.fixture
def tiny_models():
    return {"logistic_regression": (LogisticRegression(max_iter=1000, random_state=0),
                                    {"clf__C": [0.1, 1.0]})}


@pytest.fixture
def small_search(monkeypatch):
    monkeypatch.setattr(train, "SHARED_GRID", TINY_GRID)


def test_split_is_stratified_and_disjoint(clean_df):
    X_tr, X_te, y_tr, y_te = train.split(clean_df)
    assert len(X_te) == round(len(clean_df) * config.TEST_SIZE)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert list(X_tr.columns) == config.FEATURES


def test_split_is_reproducible(clean_df):
    a = train.split(clean_df)[1].index
    b = train.split(clean_df)[1].index
    assert list(a) == list(b)


def test_candidates_cover_three_model_families():
    assert set(train.candidates()) == {"logistic_regression", "random_forest", "gradient_boosting"}
    for _, grid in train.candidates().values():
        assert all(k.startswith("clf__") for k in grid)


def test_baseline_cv_auc_is_reasonable(clean_df):
    """Quality floor: a plain pipeline must clear 0.80 ROC-AUC in CV."""
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = features.build_pipeline(LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    assert cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean() > 0.80


def test_metrics_are_complete_and_bounded(clean_df):
    X_tr, X_te, y_tr, y_te = train.split(clean_df)
    model = features.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    m = train.test_metrics(model, X_te, y_te)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())


def test_nested_cv_reports_mean_and_std(clean_df, small_search):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    est, grid = LogisticRegression(max_iter=1000), {"clf__C": [0.1, 1.0]}
    res = train.nested_cv(est, grid, X, y, outer_splits=3, inner_splits=2)
    assert set(res) == set(train.SCORING)
    for mean, std in res.values():
        assert 0 <= mean <= 1 and std >= 0


def test_pick_winner_uses_cv_not_test():
    results = [
        {"model": "a", "cv_roc_auc": 0.90, "test_roc_auc": 0.80},
        {"model": "b", "cv_roc_auc": 0.85, "test_roc_auc": 0.99},
    ]
    assert train.pick_winner(results)["model"] == "a"


def test_train_all_logs_runs_and_package_saves_model(
        clean_df, isolated_paths, small_search, tiny_models, one_patient):
    X_tr, X_te, y_tr, y_te = train.split(clean_df)
    train.setup_mlflow()
    results, summary, winner = train.train_all(X_tr, X_te, y_tr, y_te, models=tiny_models)

    assert list(summary["model"]) == ["logistic_regression"]
    assert winner["model"] == "logistic_regression"

    # tracking: parent + child run, with params, metrics and artifacts
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT)
    assert len(client.search_runs([exp.experiment_id])) == 2
    child = client.get_run(winner["run_id"])
    assert child.data.params["model"] == "logistic_regression"
    assert {"cv_roc_auc", "test_roc_auc", "test_recall"} <= set(child.data.metrics)
    assert {a.path for a in client.list_artifacts(winner["run_id"])} >= {"plots", "tuning"}

    # packaging: file + metadata + registry
    meta = train.package(winner, summary)
    saved = joblib.load(config.MODEL_DIR / "heart_model.joblib")
    assert 0 <= saved.predict_proba(one_patient)[0, 1] <= 1
    np.testing.assert_allclose(saved.predict_proba(X_te), winner["estimator"].predict_proba(X_te))
    on_disk = json.loads((config.MODEL_DIR / "model_metadata.json").read_text())
    assert on_disk["input_features"] == config.FEATURES
    assert on_disk["mlflow_run_id"] == meta["mlflow_run_id"]
    assert client.get_registered_model(config.REGISTERED_MODEL).name == config.REGISTERED_MODEL
