"""Central configuration: paths, column groups and fixed seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

# MLflow: sqlite backend so the model registry works without a server
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlartifacts").as_posix()
EXPERIMENT = "heart-disease-risk"
REGISTERED_MODEL = "heart-disease-classifier"

# UCI Cleveland subset (303 rows, the 14 attributes used in the literature)
DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data")
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
TARGET = "target"
LABELS = {0: "No disease", 1: "Disease"}

# column groups used by the preprocessing pipeline
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]   # nominal -> one-hot
BINARY_ORDINAL = ["sex", "fbs", "exang", "ca"]     # already 0/1 or 0-3 counts
ENGINEERED = ["hr_pct_max", "oldpeak_x_exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY_ORDINAL

SEED = 42
TEST_SIZE = 0.20
