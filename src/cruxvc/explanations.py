"""Common grouped-attribution engine and conditional-imputation sensitivity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .metrics import logit


def load_feature_groups(path: str | Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {str(name): list(columns) for name, columns in payload["groups"].items()}


def active_feature_groups(
    groups: Mapping[str, Sequence[str]], feature_columns: Sequence[str]
) -> dict[str, list[str]]:
    available = set(feature_columns)
    active = {name: [column for column in columns if column in available] for name, columns in groups.items()}
    return {name: columns for name, columns in active.items() if columns}


def model_log_odds(model: Any, X: pd.DataFrame, clip_epsilon: float = 1e-6) -> np.ndarray:
    if hasattr(model, "decision_function"):
        values = np.asarray(model.decision_function(X), dtype=float)
        if values.ndim == 2:
            values = values[:, -1]
        return values.ravel()
    probabilities = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return logit(probabilities, clip_epsilon)


def _path_contributions(
    model: Any,
    x: pd.Series,
    donor: pd.Series,
    ordered_groups: list[str],
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], float, float]:
    states = []
    current = donor.copy()
    states.append(current[list(feature_columns)].copy())
    for group in ordered_groups:
        current = current.copy()
        for column in groups[group]:
            current[column] = x[column]
        states.append(current[list(feature_columns)].copy())
    batch = pd.DataFrame(states, columns=list(feature_columns))
    values = model_log_odds(model, batch)
    contributions = {
        group: float(values[index + 1] - values[index]) for index, group in enumerate(ordered_groups)
    }
    return contributions, float(values[0]), float(values[-1])


def grouped_permutation_shap_one(
    model: Any,
    x: pd.Series,
    background: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    *,
    n_orderings: int = 20,
    seed: int = 0,
    antithetic: bool = True,
) -> dict[str, Any]:
    group_names = list(groups)
    if not group_names:
        raise ValueError("No active semantic feature groups")
    rng = np.random.default_rng(seed)
    sums = {group: 0.0 for group in group_names}
    baseline_values = []
    endpoint_values = []
    paths = 0
    while paths < n_orderings:
        ordering = list(rng.permutation(group_names))
        donor = background.iloc[int(rng.integers(0, len(background)))]
        contributions, baseline, endpoint = _path_contributions(
            model, x, donor, ordering, groups, feature_columns
        )
        for group, value in contributions.items():
            sums[group] += value
        baseline_values.append(baseline)
        endpoint_values.append(endpoint)
        paths += 1
        if antithetic and paths < n_orderings:
            reverse = list(reversed(ordering))
            contributions, baseline, endpoint = _path_contributions(
                model, x, donor, reverse, groups, feature_columns
            )
            for group, value in contributions.items():
                sums[group] += value
            baseline_values.append(baseline)
            endpoint_values.append(endpoint)
            paths += 1
    phi = {group: value / paths for group, value in sums.items()}
    baseline_mean = float(np.mean(baseline_values))
    prediction = float(model_log_odds(model, pd.DataFrame([x[list(feature_columns)]]))[0])
    reconstruction = baseline_mean + sum(phi.values())
    return {
        "phi": phi,
        "baseline_log_odds": baseline_mean,
        "prediction_log_odds": prediction,
        "reconstruction_log_odds": reconstruction,
        "efficiency_residual": float(prediction - reconstruction),
        "endpoint_sd": float(np.std(endpoint_values, ddof=1)) if len(endpoint_values) > 1 else 0.0,
        "n_paths": paths,
    }


@dataclass
class GowerKNNDonorSampler:
    reference: pd.DataFrame
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    k: int = 30

    @classmethod
    def fit(
        cls,
        reference: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        k: int = 30,
    ) -> "GowerKNNDonorSampler":
        feature_columns = tuple(feature_columns)
        numeric = tuple(c for c in feature_columns if pd.api.types.is_numeric_dtype(reference[c]))
        categorical = tuple(c for c in feature_columns if c not in numeric)
        obj = cls(reference[list(feature_columns)].reset_index(drop=True).copy(), feature_columns, numeric, categorical, k)
        obj._numeric_min = obj.reference[list(numeric)].min(numeric_only=True) if numeric else pd.Series(dtype=float)
        obj._numeric_range = (
            obj.reference[list(numeric)].max(numeric_only=True) - obj._numeric_min
            if numeric
            else pd.Series(dtype=float)
        )
        if numeric:
            obj._numeric_range = obj._numeric_range.replace(0, 1).fillna(1)
        return obj

    def distances(self, x: pd.Series, observed_columns: Sequence[str]) -> np.ndarray:
        observed = [column for column in observed_columns if column in self.feature_columns]
        if not observed:
            return np.zeros(len(self.reference), dtype=float)
        components = []
        for column in observed:
            ref = self.reference[column]
            value = x[column]
            if column in self.numeric_columns:
                if pd.isna(value):
                    component = np.where(ref.isna(), 0.0, 1.0)
                else:
                    denominator = float(self._numeric_range.get(column, 1.0))
                    component = np.abs(pd.to_numeric(ref, errors="coerce").to_numpy() - float(value)) / denominator
                    component = np.where(np.isfinite(component), np.minimum(component, 1.0), 1.0)
            else:
                ref_string = ref.astype("string")
                component = np.where(ref_string.isna() & pd.isna(value), 0.0, (ref_string.astype(str) != str(value)).astype(float))
            components.append(np.asarray(component, dtype=float))
        return np.mean(np.vstack(components), axis=0)

    def donor_indices(
        self,
        x: pd.Series,
        observed_columns: Sequence[str],
        *,
        n: int = 1,
        seed: int = 0,
    ) -> np.ndarray:
        distances = self.distances(x, observed_columns)
        k = min(max(int(self.k), 1), len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / (distances[nearest] + 1e-6)
        weights = weights / weights.sum()
        rng = np.random.default_rng(seed)
        return rng.choice(nearest, size=n, replace=True, p=weights)

    def replace_groups(
        self,
        x: pd.Series,
        groups_to_replace: Sequence[str],
        groups: Mapping[str, Sequence[str]],
        *,
        seed: int = 0,
    ) -> pd.Series:
        replace_columns = {column for group in groups_to_replace for column in groups[group]}
        observed = [column for column in self.feature_columns if column not in replace_columns]
        donor_index = int(self.donor_indices(x, observed, n=1, seed=seed)[0])
        donor = self.reference.iloc[donor_index]
        result = x.copy()
        for column in replace_columns:
            result[column] = donor[column]
        return result

    def diagnostics(self, sample: pd.DataFrame, seed: int = 0) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        distances = []
        effective_counts = []
        support_violations = 0
        for _, x in sample.iterrows():
            observed = list(self.feature_columns)
            d = self.distances(x, observed)
            k = min(self.k, len(d))
            nearest = np.partition(d, k - 1)[:k]
            distances.append(float(np.mean(nearest)))
            weights = 1 / (nearest + 1e-6)
            weights /= weights.sum()
            effective_counts.append(float(1 / np.sum(weights**2)))
            for column in self.numeric_columns:
                value = pd.to_numeric(pd.Series([x[column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    ref = pd.to_numeric(self.reference[column], errors="coerce")
                    support_violations += int(value < ref.min() or value > ref.max())
        return {
            "mean_neighbour_distance": float(np.mean(distances)) if distances else np.nan,
            "p95_neighbour_distance": float(np.quantile(distances, 0.95)) if distances else np.nan,
            "mean_effective_donor_count": float(np.mean(effective_counts)) if effective_counts else np.nan,
            "support_violations": int(support_violations),
            "n_cases": int(len(sample)),
        }


def grouped_conditional_imputation_shap_one(
    model: Any,
    x: pd.Series,
    sampler: GowerKNNDonorSampler,
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    *,
    n_orderings: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    group_names = list(groups)
    rng = np.random.default_rng(seed)
    sums = {group: 0.0 for group in group_names}
    baseline_values = []
    paths = 0
    while paths < n_orderings:
        ordering = list(rng.permutation(group_names))
        selected: list[str] = []
        previous_groups = list(group_names)
        previous = sampler.replace_groups(x, previous_groups, groups, seed=seed + paths * 1000)
        previous_value = float(model_log_odds(model, pd.DataFrame([previous[list(feature_columns)]]))[0])
        baseline_values.append(previous_value)
        for position, group in enumerate(ordering):
            selected.append(group)
            remaining = [candidate for candidate in group_names if candidate not in selected]
            current = sampler.replace_groups(
                x,
                remaining,
                groups,
                seed=seed + paths * 1000 + position + 1,
            )
            current_value = float(model_log_odds(model, pd.DataFrame([current[list(feature_columns)]]))[0])
            sums[group] += current_value - previous_value
            previous_value = current_value
        paths += 1
    phi = {group: value / paths for group, value in sums.items()}
    prediction = float(model_log_odds(model, pd.DataFrame([x[list(feature_columns)]]))[0])
    baseline = float(np.mean(baseline_values))
    return {
        "phi": phi,
        "baseline_log_odds": baseline,
        "prediction_log_odds": prediction,
        "reconstruction_log_odds": baseline + sum(phi.values()),
        "efficiency_residual": prediction - baseline - sum(phi.values()),
        "n_paths": paths,
        "method_label": "gower_knn_conditional_imputation_sensitivity",
    }


def explain_dataset(
    model: Any,
    cases: pd.DataFrame,
    background: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    *,
    model_metadata: Mapping[str, Any],
    background_id: str,
    approximation_seed: int,
    n_orderings: int,
    method: str = "interventional",
    conditional_sampler: GowerKNNDonorSampler | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in cases.itertuples(index=False):
        x = pd.Series(row._asdict())
        if method == "interventional":
            result = grouped_permutation_shap_one(
                model,
                x,
                background,
                groups,
                feature_columns,
                n_orderings=n_orderings,
                seed=approximation_seed + int(x.get("case_id", 0)),
            )
            method_label = "grouped_permutation_shap_interventional"
        elif method == "conditional_imputation":
            if conditional_sampler is None:
                raise ValueError("conditional_sampler is required")
            result = grouped_conditional_imputation_shap_one(
                model,
                x,
                conditional_sampler,
                groups,
                feature_columns,
                n_orderings=n_orderings,
                seed=approximation_seed + int(x.get("case_id", 0)),
            )
            method_label = result.get("method_label", "conditional_imputation")
        else:
            raise KeyError(method)
        for group, value in result["phi"].items():
            records.append(
                {
                    "case_id": int(x["case_id"]),
                    **dict(model_metadata),
                    "background_id": background_id,
                    "approximation_seed": int(approximation_seed),
                    "explanation_method": method_label,
                    "feature_group": group,
                    "phi_log_odds": float(value),
                    "absolute_phi": abs(float(value)),
                    "baseline_log_odds": result["baseline_log_odds"],
                    "prediction_log_odds": result["prediction_log_odds"],
                    "efficiency_residual": result["efficiency_residual"],
                    "n_paths": result["n_paths"],
                }
            )
    return pd.DataFrame(records)


def proportional_stratified_sample(
    frame: pd.DataFrame,
    *,
    n: int,
    strata: Sequence[str],
    seed: int = 260826,
) -> pd.DataFrame:
    """Sample exactly ``n`` rows with deterministic proportional allocation.

    Allocations are capped by stratum size and, when feasible, every observed
    stratum receives at least one row. Residual slots are assigned by largest
    fractional remainder with stable stratum ordering.
    """
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    n = min(int(n), len(frame))
    available = [column for column in strata if column in frame.columns]
    if not available:
        return frame.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)
    groups = list(frame.groupby(available, dropna=False, sort=True))
    sizes = np.asarray([len(group) for _, group in groups], dtype=int)
    ideal = n * sizes / sizes.sum()
    allocation = np.floor(ideal).astype(int)
    if n >= len(groups):
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, sizes)
    # If minimum-one allocation overshoots, remove from the smallest fractional
    # remainders while preserving at least one when feasible.
    while allocation.sum() > n:
        candidates = [i for i in range(len(groups)) if allocation[i] > (1 if n >= len(groups) else 0)]
        index = min(candidates, key=lambda i: (ideal[i] - np.floor(ideal[i]), sizes[i], i))
        allocation[index] -= 1
    while allocation.sum() < n:
        candidates = [i for i in range(len(groups)) if allocation[i] < sizes[i]]
        index = max(candidates, key=lambda i: (ideal[i] - allocation[i], sizes[i], -i))
        allocation[index] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for (_, group), count in zip(groups, allocation):
        if count:
            chosen = rng.choice(group.index.to_numpy(), size=int(count), replace=False)
            selected.extend(chosen.tolist())
    result = frame.loc[selected].copy()
    if len(result) != n:
        raise RuntimeError(f"Stratified sampler selected {len(result)} rows, expected {n}")
    sort_columns = [column for column in ["t0", "case_id"] if column in result.columns]
    return result.sort_values(sort_columns).reset_index(drop=True) if sort_columns else result.sort_index().reset_index(drop=True)


def sample_background_ids(
    development: pd.DataFrame,
    *,
    n_sets: int,
    n_per_set: int,
    strata: Sequence[str] = ("landmark_round_type",),
    seed: int = 260826,
) -> pd.DataFrame:
    records = []
    for set_index in range(n_sets):
        sampled = proportional_stratified_sample(
            development,
            n=n_per_set,
            strata=strata,
            seed=seed + set_index,
        )
        for order, case_id in enumerate(sampled["case_id"]):
            records.append({"background_id": f"bg_{set_index + 1}", "order": order, "case_id": int(case_id)})
    return pd.DataFrame(records)


def sample_local_audit_cases(
    test: pd.DataFrame,
    *,
    n: int,
    outcome_columns: Sequence[str],
    seed: int = 260826,
) -> pd.DataFrame:
    frame = test.copy()
    frame["joint_outcome_pattern"] = frame[list(outcome_columns)].astype(str).agg("".join, axis=1)
    strata = [column for column in ["landmark_round_type", "joint_outcome_pattern"] if column in frame.columns]
    sample = proportional_stratified_sample(frame, n=n, strata=strata, seed=seed)
    stratum_counts = frame.groupby(strata, dropna=False).size().rename("population_n")
    sample_counts = sample.groupby(strata, dropna=False).size().rename("sample_n")
    weights = pd.concat([stratum_counts, sample_counts], axis=1).reset_index()
    if weights["sample_n"].isna().any() or weights["sample_n"].le(0).any():
        raise RuntimeError("Audit sampling omitted a frozen stratum; design weights are undefined")
    weights["design_weight"] = weights["population_n"] / weights["sample_n"]
    sample = sample.merge(weights[strata + ["design_weight"]], on=strata, how="left")
    sample["audit_inclusion_probability"] = 1 / sample["design_weight"]
    sample["audit_stratum"] = sample[strata].astype("string").fillna("__MISSING__").agg("|".join, axis=1)
    return sample.sort_values("case_id").reset_index(drop=True)


def approximation_repeat_diagnostics(attributions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column
        for column in ["case_id", "outcome", "family", "config_id", "refit_id", "background_id", "feature_group"]
        if column in attributions.columns
    ]
    return (
        attributions.groupby(keys, dropna=False)["phi_log_odds"]
        .agg(approximation_mean="mean", approximation_sd="std", approximation_range=lambda s: s.max() - s.min(), replicates="size")
        .reset_index()
    )
