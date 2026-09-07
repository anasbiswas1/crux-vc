"""Model factories, blocked validation, and empirical near-optimal set selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metrics import binary_prediction_metrics
from .preprocessing import build_preprocessor, infer_feature_types


def stable_config_id(family: str, parameters: Mapping[str, Any]) -> str:
    payload = json.dumps({"family": family, "parameters": parameters}, sort_keys=True, default=str)
    return f"{family}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def load_model_grid(path: str | Path, max_per_family: int | None = None) -> pd.DataFrame:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = []
    for _, specification in payload.items():
        family = specification["family"]
        complexity_rank = int(specification.get("complexity_rank", 99))
        parameters = specification["parameters"]
        if max_per_family is not None:
            parameters = parameters[: int(max_per_family)]
        for order, params in enumerate(parameters):
            rows.append(
                {
                    "family": family,
                    "complexity_rank": complexity_rank,
                    "grid_order": order,
                    "parameters": params,
                    "config_id": stable_config_id(family, params),
                }
            )
    return pd.DataFrame(rows)


def estimator_for(family: str, parameters: Mapping[str, Any], seed: int):
    params = dict(parameters)
    if family == "logistic_elastic_net":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight=None,
            **params,
        )
    if family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install the xgboost extra to use the xgboost family") from exc
        return XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            reg_lambda=1.0,
            **params,
        )
    if family == "explainable_boosting":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            raise ImportError("Install interpret to use ExplainableBoostingClassifier") from exc
        return ExplainableBoostingClassifier(
            random_state=seed,
            n_jobs=-1,
            validation_size=0.15,
            outer_bags=8,
            inner_bags=0,
            **params,
        )
    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm to use the lightgbm sensitivity") from exc
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **params)
    raise KeyError(f"Unknown model family {family}")


def build_model_pipeline(
    X: pd.DataFrame,
    feature_columns: Iterable[str],
    family: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    feature_columns = list(feature_columns)
    types = infer_feature_types(X, feature_columns)
    scale = family == "logistic_elastic_net"
    preprocessor = build_preprocessor(types, scale_numeric=scale)
    estimator = estimator_for(family, parameters, seed)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def fit_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    feature_columns: Iterable[str],
    family: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    model = build_model_pipeline(X, feature_columns, family, parameters, seed)
    model.fit(X[list(feature_columns)], np.asarray(y, dtype=int))
    return model


def predict_positive(model: Any, X: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> np.ndarray:
    frame = X[list(feature_columns)] if feature_columns is not None else X
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary classifier must return an n x 2 probability matrix")
    return probabilities[:, 1]


@dataclass
class BlockedSelectionResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    registry: pd.DataFrame


def blocked_model_selection(
    features: pd.DataFrame,
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    grid: pd.DataFrame,
    outcomes: list[str],
    seeds: list[int],
    feature_columns: list[str],
    *,
    model_dir: str | Path | None = None,
) -> BlockedSelectionResult:
    data = features.merge(cohort[["case_id", *outcomes]], on="case_id", how="inner")
    rows_pred: list[pd.DataFrame] = []
    rows_metric: list[dict[str, Any]] = []
    rows_registry: list[dict[str, Any]] = []
    model_dir = Path(model_dir) if model_dir else None
    for outcome in outcomes:
        for grid_row in grid.itertuples(index=False):
            for seed in seeds:
                model_id = f"{grid_row.config_id}-seed{seed}"
                config_fold_predictions = []
                for fold_id in sorted(folds["fold_id"].unique()):
                    fold = folds[folds["fold_id"].eq(fold_id)]
                    train_ids = set(fold.loc[fold["role"].eq("train"), "case_id"])
                    val_ids = set(fold.loc[fold["role"].eq("validation"), "case_id"])
                    train = data[data["case_id"].isin(train_ids)]
                    validation = data[data["case_id"].isin(val_ids)]
                    if train.empty or validation.empty:
                        raise RuntimeError(f"Empty fold {fold_id} for {model_id}")
                    model = fit_model(
                        train,
                        train[outcome],
                        feature_columns,
                        grid_row.family,
                        grid_row.parameters,
                        seed,
                    )
                    probability = predict_positive(model, validation, feature_columns)
                    fold_pred = pd.DataFrame(
                        {
                            "case_id": validation["case_id"].to_numpy(),
                            "outcome": outcome,
                            "fold_id": fold_id,
                            "family": grid_row.family,
                            "config_id": grid_row.config_id,
                            "model_id": model_id,
                            "seed": seed,
                            "y_true": validation[outcome].to_numpy(dtype=int),
                            "probability": probability,
                        }
                    )
                    rows_pred.append(fold_pred)
                    config_fold_predictions.append(fold_pred)
                    metric = binary_prediction_metrics(fold_pred["y_true"], fold_pred["probability"])
                    rows_metric.append(
                        {
                            "outcome": outcome,
                            "fold_id": fold_id,
                            "family": grid_row.family,
                            "config_id": grid_row.config_id,
                            "model_id": model_id,
                            "seed": seed,
                            **metric,
                        }
                    )
                pooled = pd.concat(config_fold_predictions, ignore_index=True)
                pooled_metric = binary_prediction_metrics(pooled["y_true"], pooled["probability"])
                model_path = None
                if model_dir is not None:
                    # Refit on all development rows after selection data generation. This artifact is
                    # for reproducibility/auditing; 2008+ cannot influence its parameters.
                    development_ids = set(folds["case_id"])
                    development = data[data["case_id"].isin(development_ids)]
                    final_model = fit_model(
                        development,
                        development[outcome],
                        feature_columns,
                        grid_row.family,
                        grid_row.parameters,
                        seed,
                    )
                    model_path = model_dir / outcome / f"{model_id}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(final_model, model_path)
                rows_registry.append(
                    {
                        "outcome": outcome,
                        "family": grid_row.family,
                        "complexity_rank": grid_row.complexity_rank,
                        "grid_order": grid_row.grid_order,
                        "config_id": grid_row.config_id,
                        "model_id": model_id,
                        "seed": seed,
                        "parameters_json": json.dumps(grid_row.parameters, sort_keys=True),
                        "development_model_path": str(model_path) if model_path else None,
                        **{f"pooled_{k}": v for k, v in pooled_metric.items()},
                    }
                )
    return BlockedSelectionResult(
        pd.concat(rows_pred, ignore_index=True),
        pd.DataFrame(rows_metric),
        pd.DataFrame(rows_registry),
    )


def startup_bootstrap_metric_se(
    predictions: pd.DataFrame,
    *,
    metric: str = "log_loss",
    repetitions: int = 1000,
    seed: int = 260826,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["outcome", "family", "config_id", "model_id", "seed"]
    for key, frame in predictions.groupby(keys, dropna=False):
        ids = frame["case_id"].to_numpy()
        strata = frame["fold_id"].to_numpy()
        values = []
        for _ in range(repetitions):
            sampled_indices = []
            for stratum in np.unique(strata):
                candidates = np.where(strata == stratum)[0]
                sampled_indices.extend(rng.choice(candidates, len(candidates), replace=True))
            sample = frame.iloc[sampled_indices]
            values.append(binary_prediction_metrics(sample["y_true"], sample["probability"])[metric])
        rows.append(
            {
                **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                f"{metric}_bootstrap_se": float(np.nanstd(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def select_near_optimal_models(
    registry: pd.DataFrame,
    metric_se: pd.DataFrame,
    *,
    cap_per_family: int = 10,
    ap_lift_min: float = 1.5,
    require_positive_brier_skill: bool = True,
) -> pd.DataFrame:
    keys = ["outcome", "family", "config_id", "model_id", "seed"]
    merged = registry.merge(metric_se, on=keys, how="left")
    rows = []
    for outcome, frame in merged.groupby("outcome"):
        eligible = frame[frame["pooled_ap_lift"].ge(ap_lift_min)].copy()
        if require_positive_brier_skill:
            eligible = eligible[eligible["pooled_brier_skill"].gt(0)]
        eligible = eligible[
            eligible["pooled_calibration_slope"].between(0.25, 4.0, inclusive="both")
            | eligible["pooled_calibration_slope"].isna()
        ]
        if eligible.empty:
            frame = frame.copy()
            frame["near_optimal_global"] = False
            frame["near_optimal_family"] = False
            frame["membership_reason"] = "no_model_passed_skill_gate"
            rows.append(frame)
            continue
        best_index = eligible["pooled_log_loss"].idxmin()
        best = eligible.loc[best_index]
        threshold = float(best["pooled_log_loss"] + best.get("log_loss_bootstrap_se", 0.0))
        eligible["near_optimal_global"] = eligible["pooled_log_loss"].le(threshold)
        family_members = []
        for family, family_frame in eligible.groupby("family"):
            family_best_index = family_frame["pooled_log_loss"].idxmin()
            family_best = family_frame.loc[family_best_index]
            family_threshold = float(
                family_best["pooled_log_loss"] + family_best.get("log_loss_bootstrap_se", 0.0)
            )
            selected = family_frame[family_frame["pooled_log_loss"].le(family_threshold)].copy()
            selected = selected.sort_values(
                ["pooled_log_loss", "complexity_rank", "grid_order", "seed", "model_id"]
            ).head(cap_per_family)
            family_members.extend(selected["model_id"].tolist())
        eligible["near_optimal_family"] = eligible["model_id"].isin(family_members)
        eligible["membership_reason"] = np.select(
            [eligible["near_optimal_global"], eligible["near_optimal_family"]],
            ["global_one_standard_error", "family_one_standard_error"],
            default="skill_pass_but_outside_set",
        )
        excluded = frame[~frame["model_id"].isin(eligible["model_id"])].copy()
        excluded["near_optimal_global"] = False
        excluded["near_optimal_family"] = False
        excluded["membership_reason"] = "failed_skill_or_calibration_gate"
        rows.extend([eligible, excluded])
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["outcome", "family", "pooled_log_loss", "model_id"]).reset_index(drop=True)
