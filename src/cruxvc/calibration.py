"""Probability calibration and calibrated-log-odds model wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .metrics import binary_prediction_metrics, clip_probabilities, logit


class PlattCalibrator(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)

    def fit(self, probabilities, y):
        self.model.fit(logit(probabilities).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def predict(self, probabilities):
        return self.model.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]


class BetaCalibrator(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)

    @staticmethod
    def _features(probabilities):
        p = clip_probabilities(probabilities)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, probabilities, y):
        self.model.fit(self._features(probabilities), np.asarray(y, dtype=int))
        return self

    def predict(self, probabilities):
        return self.model.predict_proba(self._features(probabilities))[:, 1]


class IsotonicCalibrator(BaseEstimator):
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, probabilities, y):
        self.model.fit(np.asarray(probabilities, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict(self, probabilities):
        return self.model.predict(np.asarray(probabilities, dtype=float))


def make_calibrator(name: str):
    if name == "platt":
        return PlattCalibrator()
    if name == "beta":
        return BetaCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator()
    raise KeyError(name)


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: Any
    feature_columns: tuple[str, ...]
    clip_epsilon: float = 1e-6

    def raw_probability(self, X):
        return np.asarray(self.base_model.predict_proba(X[list(self.feature_columns)]), dtype=float)[:, 1]

    def predict_proba(self, X):
        p = clip_probabilities(self.calibrator.predict(self.raw_probability(X)), self.clip_epsilon)
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        return logit(self.predict_proba(X)[:, 1], self.clip_epsilon)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def cross_fitted_calibration_predictions(
    raw_probabilities,
    y,
    method: str,
    *,
    folds: int = 5,
    seed: int = 260826,
) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=float)
    labels = np.asarray(y, dtype=int)
    output = np.full(len(labels), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(raw, labels):
        calibrator = make_calibrator(method)
        calibrator.fit(raw[train_index], labels[train_index])
        output[validation_index] = calibrator.predict(raw[validation_index])
    if np.isnan(output).any():
        raise RuntimeError("Cross-fitted calibration left missing predictions")
    return clip_probabilities(output)


def choose_calibrator(
    raw_probabilities,
    y,
    *,
    seed: int = 260826,
    slope_interval: tuple[float, float] = (0.8, 1.2),
) -> tuple[str, Any, list[dict[str, float]]]:
    methods = ["platt", "beta", "isotonic"]
    rows = []
    predictions = {}
    for method in methods:
        crossfit = cross_fitted_calibration_predictions(raw_probabilities, y, method, seed=seed)
        predictions[method] = crossfit
        metrics = binary_prediction_metrics(y, crossfit)
        rows.append({"method": method, **metrics})
    losses = np.array([row["log_loss"] for row in rows], dtype=float)
    best_loss = float(np.nanmin(losses))
    # A conservative one-SE approximation from per-case log loss.
    y_array = np.asarray(y, dtype=int)
    best_method = rows[int(np.nanargmin(losses))]["method"]
    best_p = predictions[best_method]
    case_losses = -(y_array * np.log(best_p) + (1 - y_array) * np.log(1 - best_p))
    one_se = float(np.std(case_losses, ddof=1) / np.sqrt(len(case_losses)))
    eligible = [row for row in rows if row["log_loss"] <= best_loss + one_se]
    platt = next(row for row in eligible if row["method"] == "platt") if any(
        row["method"] == "platt" for row in eligible
    ) else None
    if platt and slope_interval[0] <= platt["calibration_slope"] <= slope_interval[1]:
        selected = "platt"
    else:
        selected = next((name for name in ["beta", "isotonic", "platt"] if any(r["method"] == name for r in eligible)), best_method)
    calibrator = make_calibrator(selected).fit(raw_probabilities, y)
    return selected, calibrator, rows
