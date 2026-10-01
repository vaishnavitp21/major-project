"""Leakage-safe feature selection, splitting, and sklearn preprocessing."""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LEAKAGE_COLUMNS, RANDOM_STATE, TEST_SIZE
from .data_validation import check_target_leakage


def identify_feature_types(
    df: pd.DataFrame,
    numerical_candidates: Iterable[str],
    categorical_candidates: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return available numeric and nominal feature names from candidate lists."""
    numerical = [c for c in numerical_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]
    overlap = set(numerical) & set(categorical)
    if overlap:
        raise ValueError(f"Features cannot be both numerical and categorical: {sorted(overlap)}")
    return numerical, categorical


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[Iterable[str]] = None,
    exclude_columns: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select X/y without transforming or removing rows, and check leakage fields."""
    if target_column not in df:
        raise KeyError(f"Target column {target_column!r} is absent.")
    excluded = set(exclude_columns or ()) | set(LEAKAGE_COLUMNS) | {target_column}
    features = list(feature_columns) if feature_columns is not None else [c for c in df.columns if c not in excluded]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Selected feature columns are absent: {missing}")
    check_target_leakage(features, target_column, forbidden_columns=excluded)
    return df.loc[:, features].copy(), df[target_column].copy()


def create_train_test_split(
    X,
    y,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    train_size=None,
):
    """Create a reproducible split; no preprocessing is fitted here."""
    stratify_values = y if stratify else None
    return train_test_split(X, y, test_size=test_size, train_size=train_size,
                            random_state=random_state, stratify=stratify_values)


def build_preprocessor(
    numerical_features: Iterable[str],
    categorical_features: Iterable[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a train-fit impute/scale/one-hot ColumnTransformer."""
    numerical_features = list(numerical_features)
    categorical_features = list(categorical_features)
    if set(numerical_features) & set(categorical_features):
        raise ValueError("Numerical and categorical feature lists overlap.")
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numerical_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numerical_features))
    if categorical_features:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn before 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError("At least one numerical or categorical feature is required.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def build_model_pipeline(estimator, numerical_features, categorical_features, scale_numeric=False) -> Pipeline:
    """Combine a fresh preprocessing transformer and estimator in a sklearn Pipeline."""
    return Pipeline([
        ("preprocessing", build_preprocessor(numerical_features, categorical_features, scale_numeric)),
        ("classifier", clone(estimator)),
    ])


class EncodedFeatureSubset(BaseEstimator, TransformerMixin):
    """Select fixed positions from a transformed feature matrix."""

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        if not self.indices:
            raise ValueError("At least one encoded feature must be selected.")
        if max(self.indices) >= X.shape[1] or min(self.indices) < 0:
            raise IndexError("An encoded feature index is outside the transformed matrix.")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices]


def build_encoded_subset_pipeline(
    estimator,
    numerical_features,
    categorical_features,
    all_encoded_feature_names,
    selected_encoded_feature_names,
    scale_numeric=False,
) -> Pipeline:
    """Build a train-fitted preprocessing/model pipeline for chosen encoded fields.

    The full preprocessing schema is fit on the pipeline's training data, then
    the fixed encoded columns are selected. If requested, only selected numeric
    fields are scaled; one-hot columns remain unchanged.
    """
    all_names = list(all_encoded_feature_names)
    selected_names = list(selected_encoded_feature_names)
    missing = [name for name in selected_names if name not in all_names]
    if missing:
        raise ValueError(f"Selected encoded features are absent from the full schema: {missing}")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Selected encoded feature names must be unique.")
    indices = [all_names.index(name) for name in selected_names]
    numeric_positions = [i for i, name in enumerate(selected_names) if name.startswith("numeric__")]
    categorical_positions = [i for i, name in enumerate(selected_names) if not name.startswith("numeric__")]
    steps = [
        ("preprocessing", build_preprocessor(numerical_features, categorical_features, scale_numeric=False)),
        ("feature_subset", EncodedFeatureSubset(indices)),
    ]
    if scale_numeric:
        transformers = []
        if numeric_positions:
            transformers.append(("numeric_scaler", StandardScaler(), numeric_positions))
        if categorical_positions:
            transformers.append(("onehot_passthrough", "passthrough", categorical_positions))
        steps.append(("selected_feature_scaling", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)))
    steps.append(("classifier", clone(estimator)))
    return Pipeline(steps)
