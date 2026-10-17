"""Feature engineering and the preprocessing / modelling pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds two clinically motivated features; switchable so CV can judge them."""

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.add_features:
            X["hr_pct_max"] = X["thalach"] / (220.0 - X["age"])
            X["oldpeak_x_exang"] = X["oldpeak"] * X["exang"]
        return X


def numeric_cols(df):
    # engineered columns only exist when FeatureEngineer is switched on
    return [c for c in config.NUMERIC + config.ENGINEERED if c in df.columns]


def build_pipeline(estimator):
    """engineer -> impute/scale/one-hot -> select top-k -> classifier."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    nominal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ordinal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("nom", nominal, config.CATEGORICAL),
        ("ord", ordinal, config.BINARY_ORDINAL),
    ])
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("prep", prep),
        ("select", SelectKBest(score_func=f_classif, k="all")),
        ("clf", estimator),
    ])


def selected_features(fitted):
    """Names of the features that survived selection in a fitted pipeline."""
    names = fitted.named_steps["prep"].get_feature_names_out()
    mask = fitted.named_steps["select"].get_support()
    return [n.split("__", 1)[1] for n in np.asarray(names)[mask]]
