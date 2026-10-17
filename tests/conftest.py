"""Shared fixtures: a small raw frame shaped like the UCI file, and isolated paths."""
import pandas as pd
import pytest

from src import config


@pytest.fixture
def raw_frame():
    """Raw-style frame: '?' placeholders, severity target 0-4, one duplicate row."""
    rows = [
        # age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal target
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "0.0", "6.0", 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, "3.0", "3.0", 2],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, "0.0", "3.0", 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, "?", "3.0", 0],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, "2.0", "?", 3],
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "0.0", "6.0", 0],  # duplicate of row 0
    ]
    return pd.DataFrame(rows, columns=config.COLUMNS)


@pytest.fixture(scope="session")
def clean_df():
    """The real cleaned dataset (committed in data/processed)."""
    return pd.read_csv(config.CLEAN_PATH)


@pytest.fixture
def one_patient():
    return pd.DataFrame([{
        "age": 60.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 250.0, "fbs": 0.0,
        "restecg": 0.0, "thalach": 120.0, "exang": 1.0, "oldpeak": 2.0, "slope": 2.0,
        "ca": 2.0, "thal": 7.0,
    }])[config.FEATURES]


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Point every output path at tmp_path so tests never touch real artifacts."""
    monkeypatch.setattr(config, "RAW_PATH", tmp_path / "raw" / "heart.data")
    monkeypatch.setattr(config, "CLEAN_PATH", tmp_path / "processed" / "heart_clean.csv")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MLFLOW_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACTS", (tmp_path / "mlartifacts").as_posix())
    return tmp_path
