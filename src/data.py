"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
import pandas as pd
import requests

from src import config


def download(force=False):
    """Download the raw Cleveland file once; skip if it already exists."""
    if config.RAW_PATH.exists() and not force:
        return config.RAW_PATH
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(config.DATA_URL, timeout=30)
    resp.raise_for_status()
    config.RAW_PATH.write_bytes(resp.content)
    return config.RAW_PATH


def load_raw():
    return pd.read_csv(download(), header=None, names=config.COLUMNS)


def clean(raw):
    """'?' -> NaN, numeric types, drop duplicates, binarise the target.

    Missing values stay NaN here and are imputed inside the model pipeline,
    so the imputer is fit on training folds only (no leakage).
    """
    df = raw.replace("?", pd.NA).apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    # original target: 0 = healthy, 1-4 = disease severity -> binary
    df[config.TARGET] = (df[config.TARGET] > 0).astype(int)
    # float features keep the MLflow model signature stable when NaN appears
    feats = [c for c in df.columns if c != config.TARGET]
    df[feats] = df[feats].astype("float64")
    return df


def build_clean_dataset():
    """Download (if needed), clean and save the dataset; returns the cleaned frame."""
    df = clean(load_raw())
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_PATH, index=False)
    return df


def load_clean():
    """Return the cleaned dataset, building it first if needed."""
    if not config.CLEAN_PATH.exists():
        return build_clean_dataset()
    return pd.read_csv(config.CLEAN_PATH)


if __name__ == "__main__":
    out = build_clean_dataset()
    print(f"saved {config.CLEAN_PATH} {out.shape}")
