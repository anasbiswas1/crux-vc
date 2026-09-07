"""Development-only semi-synthetic controls and IID risk-control simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _standardized_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    median = np.nanmedian(values)
    values = np.where(np.isfinite(values), values, median)
    scale = np.std(values)
    return (values - np.mean(values)) / (scale if scale > 0 else 1.0)


def semi_synthetic_outcomes(
    features: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    *,
    shared_groups: Sequence[str],
    outcome_specific_groups: Mapping[str, Sequence[str]],
    seed: int = 260826,
    target_prevalence: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    output = features[["case_id"]].copy()
    truth_rows = []

    def group_signal(group_name: str) -> np.ndarray:
        columns = [c for c in groups[group_name] if c in features.columns]
        if not columns:
            return np.zeros(len(features))
        signals = []
        for column in columns:
            if pd.api.types.is_numeric_dtype(features[column]):
                signals.append(_standardized_numeric(features[column]))
            else:
                codes, _ = pd.factorize(features[column].astype("string"), sort=True)
                signals.append((codes - codes.mean()) / (codes.std() or 1))
        return np.mean(np.vstack(signals), axis=0)

    shared = np.zeros(len(features))
    for group in shared_groups:
        weight = float(rng.uniform(0.5, 1.5) * rng.choice([-1, 1]))
        shared += weight * group_signal(group)
        truth_rows.append({"outcome": "shared", "feature_group": group, "coefficient": weight, "role": "shared"})

    for outcome, specific_groups in outcome_specific_groups.items():
        score = shared.copy()
        for group in specific_groups:
            weight = float(rng.uniform(0.75, 1.75) * rng.choice([-1, 1]))
            score += weight * group_signal(group)
            truth_rows.append(
                {"outcome": outcome, "feature_group": group, "coefficient": weight, "role": "outcome_specific"}
            )
        score += rng.normal(0, 0.75, len(score))
        intercept = np.quantile(score, 1 - target_prevalence)
        probability = 1 / (1 + np.exp(-(score - intercept)))
        output[outcome] = rng.binomial(1, probability).astype("int8")
        output[f"{outcome}_probability"] = probability
    return output, pd.DataFrame(truth_rows)


def inject_future_round_count_positive_control(
    development_features: pd.DataFrame,
    development_labels: pd.DataFrame,
    *,
    source_outcome: str = "F36",
    seed: int = 260826,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = development_features.copy()
    labels = development_labels.set_index("case_id")[source_outcome]
    signal = result["case_id"].map(labels).astype(float)
    result["future_round_count_36"] = signal * rng.integers(1, 4, len(result)) + rng.binomial(1, 0.05, len(result))
    result["matrix_id"] = "DEVELOPMENT_POSITIVE_CONTROL_FUTURE_ONLY"
    return result


def randomize_labels(labels: Sequence[int], seed: int = 260826) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(labels, dtype=int).copy()
    rng.shuffle(values)
    return values


@dataclass
class IIDPolicySimulationResult:
    repetitions: pd.DataFrame
    summary: dict[str, Any]


def simulate_iid_policy_violations(
    policy_builder,
    *,
    repetitions: int = 1000,
    calibration_n: int = 1000,
    population_n: int = 100_000,
    delta: float = 0.05,
    seed: int = 260826,
) -> IIDPolicySimulationResult:
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        calibration_x = rng.normal(size=(calibration_n, 4))
        calibration_y = rng.binomial(1, 1 / (1 + np.exp(-calibration_x[:, 0])))
        population_x = rng.normal(size=(population_n, 4))
        population_y = rng.binomial(1, 1 / (1 + np.exp(-population_x[:, 0])))
        result = policy_builder(calibration_x, calibration_y, population_x, population_y, repetition)
        rows.append({"repetition": repetition, **result})
    frame = pd.DataFrame(rows)
    violations = int(frame["any_constraint_violation"].sum())
    # Exact Clopper–Pearson upper interval.
    upper = 1.0 if violations == repetitions else float(
        __import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.975, violations + 1, repetitions - violations)
    )
    summary = {
        "repetitions": repetitions,
        "violations": violations,
        "violation_rate": violations / repetitions,
        "exact_95pct_upper": upper,
        "compatible_with_delta": upper <= delta,
    }
    return IIDPolicySimulationResult(frame, summary)
