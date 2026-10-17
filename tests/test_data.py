"""Data acquisition and cleaning."""
import numpy as np
import pandas as pd

from src import config, data


def test_clean_marks_placeholders_as_nan(raw_frame):
    df = data.clean(raw_frame)
    assert df["ca"].isna().sum() == 1
    assert df["thal"].isna().sum() == 1


def test_clean_binarises_target(raw_frame):
    df = data.clean(raw_frame)
    assert set(df[config.TARGET].unique()) <= {0, 1}
    # severities 2 and 3 in the fixture must both become 1
    assert df[config.TARGET].sum() == 2


def test_clean_drops_duplicates_and_resets_index(raw_frame):
    df = data.clean(raw_frame)
    assert len(df) == len(raw_frame) - 1
    assert list(df.index) == list(range(len(df)))


def test_clean_features_are_float(raw_frame):
    df = data.clean(raw_frame)
    feats = [c for c in df.columns if c != config.TARGET]
    assert all(df[c].dtype == np.float64 for c in feats)


def test_clean_does_not_modify_input(raw_frame):
    before = raw_frame.copy()
    data.clean(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)


def test_download_skips_when_file_exists(isolated_paths, monkeypatch):
    config.RAW_PATH.parent.mkdir(parents=True)
    config.RAW_PATH.write_text("cached")

    def boom(*args, **kwargs):
        raise AssertionError("network must not be used when the file exists")

    monkeypatch.setattr(data.requests, "get", boom)
    assert data.download() == config.RAW_PATH


def test_download_writes_file(isolated_paths, monkeypatch):
    class Resp:
        content = b"1,2,3"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Resp())
    assert data.download().read_bytes() == b"1,2,3"


def test_build_clean_dataset_saves_csv(isolated_paths, raw_frame, monkeypatch):
    monkeypatch.setattr(data, "load_raw", lambda: raw_frame)
    df = data.build_clean_dataset()
    assert config.CLEAN_PATH.exists()
    assert pd.read_csv(config.CLEAN_PATH).shape == df.shape


def test_committed_dataset_shape_and_balance(clean_df):
    assert clean_df.shape == (303, 14)
    assert list(clean_df.columns) == config.COLUMNS
    assert 0.4 < clean_df[config.TARGET].mean() < 0.6
    assert clean_df[config.FEATURES].isna().sum().sum() == 6
