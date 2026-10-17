"""Model development: tuning, nested CV, MLflow tracking and packaging.

Run end to end:  python -m src.train
"""
import json
import platform
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from src import config, data, features  # noqa: E402

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}

# search dimensions shared by every model: do the engineered features help,
# and how many (post-encoding) features should be kept? CV decides.
SHARED_GRID = {
    "fe__add_features": [False, True],
    "select__k": [8, 12, 16, "all"],
}


def candidates():
    """name -> (estimator, model-specific grid). Seeds fixed for reproducibility."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.SEED),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=1),
            {"clf__max_depth": [3, 5, None],
             "clf__min_samples_leaf": [1, 3, 5],
             "clf__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=config.SEED),
            {"clf__n_estimators": [50, 100, 200],
             "clf__learning_rate": [0.03, 0.1],
             "clf__max_depth": [2, 3],
             "clf__subsample": [0.8, 1.0]},
        ),
    }


def split(df):
    """Stratified train/test split of the cleaned frame."""
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.SEED)


def make_search(estimator, grid, inner_splits=5):
    """Grid search over the whole pipeline, so selection + FE live inside CV."""
    cv = StratifiedKFold(inner_splits, shuffle=True, random_state=config.SEED)
    return GridSearchCV(features.build_pipeline(estimator), {**SHARED_GRID, **grid},
                        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)


def nested_cv(estimator, grid, X, y, outer_splits=5, inner_splits=3):
    """Tuning is repeated inside every outer training fold -> unbiased estimate."""
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=config.SEED + 1)
    res = cross_validate(make_search(estimator, grid, inner_splits), X, y,
                         cv=outer, scoring=SCORING)
    return {k.replace("test_", ""): (v.mean(), v.std())
            for k, v in res.items() if k.startswith("test_")}


def test_metrics(model, X, y):
    pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba)}


def _save(fig, name, outdir):
    path = Path(outdir) / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_cm(model, X, y, name, outdir):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve (test set): {name}")
    roc = _save(fig, f"{name}_roc.png", outdir)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax,
                                          display_labels=["healthy", "disease"], colorbar=False)
    ax.set_title(f"Confusion matrix (test set): {name}")
    return roc, _save(fig, f"{name}_confusion.png", outdir)


def plot_importance(model, name, outdir):
    """Coefficients for linear models, impurity importances for trees."""
    clf = model.named_steps["clf"]
    vals = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    s = pd.Series(vals, index=features.selected_features(model)).sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(s) + 1))
    s.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_title(f"Feature weights: {name}")
    return _save(fig, f"{name}_importance.png", outdir)


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT) is None:
        mlflow.create_experiment(config.EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS)
    mlflow.set_experiment(config.EXPERIMENT)


def run_model(name, estimator, grid, X_tr, X_te, y_tr, y_te):
    """One MLflow run per candidate: tune, nested-CV, test, log everything."""
    with mlflow.start_run(run_name=name, nested=True):
        search = make_search(estimator, grid)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv = nested_cv(estimator, grid, X_tr, y_tr)
        test = test_metrics(best, X_te, y_te)
        kept = features.selected_features(best)

        mlflow.log_params({
            "model": name, "seed": config.SEED, "test_size": config.TEST_SIZE,
            "outer_cv": "5-fold stratified", "inner_cv": "5-fold stratified (3 inside nested)",
            "n_candidates": len(search.cv_results_["params"]), "n_features_kept": len(kept),
            **{k: str(v) for k, v in search.best_params_.items()},
        })
        mlflow.log_metric("tuning_best_cv_roc_auc", search.best_score_)
        for m, (mu, sd) in cv.items():
            mlflow.log_metric(f"cv_{m}", mu)
            mlflow.log_metric(f"cv_{m}_std", sd)
        for m, v in test.items():
            mlflow.log_metric(f"test_{m}", v)

        with tempfile.TemporaryDirectory() as tmp:
            for p in (*plot_roc_cm(best, X_te, y_te, name, tmp), plot_importance(best, name, tmp)):
                mlflow.log_artifact(str(p), "plots")
            table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            table.to_csv(Path(tmp) / "cv_results.csv", index=False)
            mlflow.log_artifact(str(Path(tmp) / "cv_results.csv"), "tuning")
        mlflow.log_dict({"selected_features": kept}, "selected_features.json")
        info = mlflow.sklearn.log_model(
            best, name="model", serialization_format="cloudpickle",
            signature=infer_signature(X_tr, best.predict_proba(X_tr)[:, 1]),
            input_example=X_tr.head(3))
        run_id = mlflow.active_run().info.run_id

    return {"model": name, "run_id": run_id, "model_uri": info.model_uri,
            "best_params": search.best_params_, "features": kept, "estimator": best,
            "cv": cv, "test": test, "cv_roc_auc": cv["roc_auc"][0], "test_roc_auc": test["roc_auc"]}


def pick_winner(results):
    """Winner chosen on nested-CV ROC-AUC only, so the test set never influences it."""
    return max(results, key=lambda r: r["cv_roc_auc"])


def train_all(X_tr, X_te, y_tr, y_te, models=None):
    """Tune + log every candidate under one parent run. Returns (results, summary, winner)."""
    models = models or candidates()
    results = []
    with mlflow.start_run(run_name="model-selection"):
        mlflow.log_params({"train_rows": len(X_tr), "test_rows": len(X_te),
                           "selection_metric": "nested-CV ROC-AUC"})
        for name, (est, grid) in models.items():
            print(f"[{name}] tuning ...", flush=True)
            results.append(run_model(name, est, grid, X_tr, X_te, y_tr, y_te))

        summary = pd.DataFrame([
            {"model": r["model"],
             **{f"cv_{k}": v[0] for k, v in r["cv"].items()},
             **{f"test_{k}": v for k, v in r["test"].items()}} for r in results])
        winner = pick_winner(results)
        mlflow.set_tag("winner", winner["model"])
        with tempfile.TemporaryDirectory() as tmp:
            summary.to_csv(Path(tmp) / "model_comparison.csv", index=False)
            mlflow.log_artifact(str(Path(tmp) / "model_comparison.csv"))
    return results, summary, winner


def package(winner, summary):
    """Save the winning pipeline (preprocessing included) + metadata, and register it."""
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(winner["estimator"], config.MODEL_DIR / "heart_model.joblib")
    meta = {
        "model": winner["model"],
        "best_params": {k: str(v) for k, v in winner["best_params"].items()},
        "input_features": config.FEATURES,
        "selected_features": winner["features"],
        "cv_metrics": {k: round(v[0], 4) for k, v in winner["cv"].items()},
        "test_metrics": {k: round(v, 4) for k, v in winner["test"].items()},
        "mlflow_run_id": winner["run_id"],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (config.MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    summary.round(4).to_csv(config.MODEL_DIR / "model_comparison.csv", index=False)
    mlflow.register_model(winner["model_uri"], config.REGISTERED_MODEL)
    return meta


def main():
    setup_mlflow()
    X_tr, X_te, y_tr, y_te = split(data.load_clean())
    results, summary, winner = train_all(X_tr, X_te, y_tr, y_te)
    package(winner, summary)
    print(summary.round(4).to_string(index=False))
    print("\nwinner:", winner["model"], winner["best_params"])
    return results, summary, winner


if __name__ == "__main__":
    main()
