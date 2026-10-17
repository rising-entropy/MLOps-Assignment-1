"""Feature engineering and the modelling pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src import config, features


def test_feature_engineer_adds_expected_columns(one_patient):
    out = features.FeatureEngineer(add_features=True).transform(one_patient)
    assert set(config.ENGINEERED) <= set(out.columns)
    assert out.loc[0, "hr_pct_max"] == pytest.approx(120 / (220 - 60))
    assert out.loc[0, "oldpeak_x_exang"] == pytest.approx(2.0 * 1.0)


def test_feature_engineer_off_is_identity(one_patient):
    out = features.FeatureEngineer(add_features=False).transform(one_patient)
    pd.testing.assert_frame_equal(out, one_patient)


def test_feature_engineer_does_not_mutate_input(one_patient):
    before = one_patient.copy()
    features.FeatureEngineer().transform(one_patient)
    pd.testing.assert_frame_equal(one_patient, before)


def test_feature_engineer_is_clonable():
    fe = features.FeatureEngineer(add_features=False)
    assert clone(fe).add_features is False


def test_numeric_cols_follow_engineered_columns(one_patient):
    on = features.FeatureEngineer(True).transform(one_patient)
    off = features.FeatureEngineer(False).transform(one_patient)
    assert features.numeric_cols(on) == config.NUMERIC + config.ENGINEERED
    assert features.numeric_cols(off) == config.NUMERIC


@pytest.fixture
def fitted(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = features.build_pipeline(LogisticRegression(max_iter=1000))
    return pipe.fit(X, y)


def test_pipeline_predicts_valid_probabilities(fitted, clean_df):
    proba = fitted.predict_proba(clean_df[config.FEATURES])
    assert proba.shape == (len(clean_df), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_handles_missing_values(fitted, one_patient):
    one_patient.loc[0, ["ca", "thal", "chol"]] = np.nan
    p = fitted.predict_proba(one_patient)[0, 1]
    assert 0 <= p <= 1 and not np.isnan(p)


def test_pipeline_ignores_unseen_category(fitted, one_patient):
    one_patient.loc[0, "thal"] = 5.0  # never seen in training
    assert not np.isnan(fitted.predict_proba(one_patient)).any()


def test_pipeline_rejects_missing_input_column(fitted, one_patient):
    with pytest.raises(Exception):
        fitted.predict(one_patient.drop(columns=["age"]))


def test_selected_features_respect_k(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = features.build_pipeline(LogisticRegression(max_iter=1000))
    pipe.set_params(select__k=8, fe__add_features=True).fit(X, y)
    names = features.selected_features(pipe)
    assert len(names) == 8
    assert all("__" not in n for n in names)


def test_all_features_encoded_width(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = features.build_pipeline(LogisticRegression(max_iter=1000))
    pipe.set_params(select__k="all", fe__add_features=False).fit(X, y)
    # 5 numeric + one-hot (4+3+3+3) + 4 ordinal = 22
    assert len(features.selected_features(pipe)) == 22
