"""Fold-local preprocessing with explicit rare-level and winsorization controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency: int | float = 10, other_label: str = "__RARE__"):
        self.min_frequency = min_frequency
        self.other_label = other_label

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).astype("string")
        self.allowed_: list[set[str]] = []
        n = len(frame)
        threshold = (
            int(np.ceil(float(self.min_frequency) * n))
            if isinstance(self.min_frequency, float) and self.min_frequency < 1
            else int(self.min_frequency)
        )
        for column in frame.columns:
            counts = frame[column].value_counts(dropna=False)
            self.allowed_.append(set(counts[counts >= max(threshold, 1)].index.astype(str)))
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).astype("string")
        for index, column in enumerate(frame.columns):
            allowed = self.allowed_[index]
            values = frame[column].astype(str)
            frame[column] = values.where(values.isin(allowed), self.other_label)
        return frame.to_numpy(dtype=object)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)


@dataclass(frozen=True)
class FeatureTypes:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def infer_feature_types(df: pd.DataFrame, feature_columns: Iterable[str]) -> FeatureTypes:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return FeatureTypes(tuple(numeric), tuple(categorical))


def build_preprocessor(
    feature_types: FeatureTypes,
    *,
    scale_numeric: bool,
    rare_min_frequency: int | float = 10,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> ColumnTransformer:
    numeric_steps = [
        ("winsorize", Winsorizer(winsor_lower, winsor_upper)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(rare_min_frequency)),
            ("onehot", _one_hot_encoder()),
        ]
    )
    transformers = []
    if feature_types.numeric:
        transformers.append(("numeric", numeric_pipeline, list(feature_types.numeric)))
    if feature_types.categorical:
        transformers.append(("categorical", categorical_pipeline, list(feature_types.categorical)))
    if not transformers:
        raise ValueError("No model features available")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)
