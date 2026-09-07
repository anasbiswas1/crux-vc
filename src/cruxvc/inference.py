"""Paired explanation estimands, hierarchical bootstrap, and multiplicity helpers."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from .metrics import attribution_distance_bundle, normalized_absolute, sqrt_jsd_base2


def attribution_vectors(
    long: pd.DataFrame,
    *,
    value_column: str = "phi_log_odds",
    index_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if index_columns is None:
        index_columns = [
            column
            for column in [
                "case_id",
                "outcome",
                "family",
                "config_id",
                "model_id",
                "refit_id",
                "background_id",
                "approximation_seed",
                "explanation_method",
                "landmark_round_type",
                "joint_outcome_pattern",
                "audit_stratum",
                "design_weight",
                "audit_inclusion_probability",
            ]
            if column in long.columns
        ]
    wide = long.pivot_table(
        index=list(index_columns), columns="feature_group", values=value_column, aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide


def _feature_columns(wide: pd.DataFrame) -> list[str]:
    metadata = {
        "case_id",
        "outcome",
        "family",
        "config_id",
        "model_id",
        "refit_id",
        "background_id",
        "approximation_seed",
        "explanation_method",
        "seed",
        "predicted_class",
        "probability",
        "landmark_round_type",
        "joint_outcome_pattern",
        "audit_stratum",
        "design_weight",
        "audit_inclusion_probability",
    }
    return [column for column in wide.columns if column not in metadata]


def matched_cross_spec_distances(
    wide: pd.DataFrame,
    outcome_a: str,
    outcome_b: str,
) -> pd.DataFrame:
    feature_columns = _feature_columns(wide)
    match_columns = [
        column
        for column in [
            "case_id",
            "family",
            "config_id",
            "refit_id",
            "background_id",
            "approximation_seed",
            "explanation_method",
            "landmark_round_type",
            "joint_outcome_pattern",
            "audit_stratum",
            "design_weight",
            "audit_inclusion_probability",
        ]
        if column in wide.columns
    ]
    left = wide[wide["outcome"].eq(outcome_a)].copy()
    right = wide[wide["outcome"].eq(outcome_b)].copy()
    merged = left.merge(right, on=match_columns, suffixes=("_a", "_b"), validate="one_to_one")
    rows = []
    for row in merged.itertuples(index=False):
        mapping = row._asdict()
        vector_a = np.array([mapping[f"{column}_a"] for column in feature_columns], dtype=float)
        vector_b = np.array([mapping[f"{column}_b"] for column in feature_columns], dtype=float)
        bundle = attribution_distance_bundle(vector_a, vector_b, feature_columns)
        rows.append({**{column: mapping[column] for column in match_columns}, **bundle})
    return pd.DataFrame(rows)


def within_spec_refit_distances(wide: pd.DataFrame, outcome: str) -> pd.DataFrame:
    feature_columns = _feature_columns(wide)
    frame = wide[wide["outcome"].eq(outcome)].copy()
    group_columns = [
        column
        for column in [
            "case_id",
            "family",
            "config_id",
            "background_id",
            "approximation_seed",
            "explanation_method",
            "landmark_round_type",
            "joint_outcome_pattern",
            "audit_stratum",
            "design_weight",
            "audit_inclusion_probability",
        ]
        if column in frame.columns
    ]
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False):
        refits = sorted(group["refit_id"].astype(str).unique()) if "refit_id" in group else ["single"]
        pairs = list(itertools.combinations(refits, 2))
        if not pairs:
            continue
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_columns, key_values))
        for refit_a, refit_b in pairs:
            left = group[group["refit_id"].astype(str).eq(refit_a)].iloc[0]
            right = group[group["refit_id"].astype(str).eq(refit_b)].iloc[0]
            vector_a = left[feature_columns].to_numpy(dtype=float)
            vector_b = right[feature_columns].to_numpy(dtype=float)
            bundle = attribution_distance_bundle(vector_a, vector_b, feature_columns)
            rows.append({**key_dict, "refit_a": refit_a, "refit_b": refit_b, **bundle})
    return pd.DataFrame(rows)


def rq1_delta_spec(
    wide: pd.DataFrame,
    outcome_a: str,
    outcome_b: str,
    *,
    distance_column: str = "sqrt_jsd",
    nacf_epsilon: float = 0.05,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    cross = matched_cross_spec_distances(wide, outcome_a, outcome_b)
    within_a = within_spec_refit_distances(wide, outcome_a)
    within_b = within_spec_refit_distances(wide, outcome_b)
    def case_summary(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        metadata = [
            column
            for column in ["case_id", "audit_stratum", "design_weight"]
            if column in frame.columns
        ]
        result = frame.groupby(metadata, dropna=False, as_index=False)[value].mean()
        if "design_weight" not in result:
            result["design_weight"] = 1.0
        if "audit_stratum" not in result:
            result["audit_stratum"] = "all_cases"
        return result

    def weighted_mean(frame: pd.DataFrame, value: str) -> float:
        valid = frame[value].notna() & frame["design_weight"].notna() & frame["design_weight"].gt(0)
        if not valid.any():
            return np.nan
        return float(np.average(frame.loc[valid, value], weights=frame.loc[valid, "design_weight"]))

    cross_case = case_summary(cross, distance_column)
    within_a_case = case_summary(within_a, distance_column) if not within_a.empty else pd.DataFrame()
    within_b_case = case_summary(within_b, distance_column) if not within_b.empty else pd.DataFrame()
    d_ab = weighted_mean(cross_case, distance_column)
    d_wa = weighted_mean(within_a_case, distance_column) if not within_a_case.empty else np.nan
    d_wb = weighted_mean(within_b_case, distance_column) if not within_b_case.empty else np.nan
    d_w = float(np.nanmean([d_wa, d_wb]))
    delta = d_ab - d_w
    nacf = delta / (1 - d_w) if np.isfinite(d_w) and d_w <= 1 - nacf_epsilon else np.nan
    summary = {
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "D_ab": d_ab,
        "D_W_a": d_wa,
        "D_W_b": d_wb,
        "D_W": d_w,
        "delta_spec": delta,
        "NACF": nacf,
        "n_cases_cross": int(cross["case_id"].nunique()),
    }
    within = pd.concat(
        [within_a.assign(outcome=outcome_a), within_b.assign(outcome=outcome_b)], ignore_index=True
    )
    return summary, cross, within


def hierarchical_bootstrap_rq1(
    cross: pd.DataFrame,
    within: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = 260826,
    distance_column: str = "sqrt_jsd",
) -> np.ndarray:
    """Stratified two-level bootstrap for the weighted paired RQ1 estimand.

    Cases are resampled with replacement inside the frozen audit strata. For
    every sampled case, computational cross-specification and within-outcome
    replicate rows are independently resampled before their case means are
    formed. The final draw is an inverse-inclusion-weighted population mean.
    """
    rng = np.random.default_rng(seed)
    case_metadata_columns = [
        column for column in ["case_id", "audit_stratum", "design_weight"] if column in cross.columns
    ]
    case_metadata = cross[case_metadata_columns].drop_duplicates("case_id").copy()
    if "audit_stratum" not in case_metadata:
        case_metadata["audit_stratum"] = "all_cases"
    if "design_weight" not in case_metadata:
        case_metadata["design_weight"] = 1.0
    cross_arrays = {
        int(case_id): group[distance_column].dropna().to_numpy(dtype=float)
        for case_id, group in cross.groupby("case_id")
    }
    within_arrays = {
        (int(case_id), str(outcome)): group[distance_column].dropna().to_numpy(dtype=float)
        for (case_id, outcome), group in within.groupby(["case_id", "outcome"])
    }
    outcomes = sorted(within["outcome"].dropna().astype(str).unique())
    values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled_cases: list[int] = []
        for _, stratum in case_metadata.groupby("audit_stratum", dropna=False, sort=True):
            ids = stratum["case_id"].astype(int).to_numpy()
            sampled_cases.extend(rng.choice(ids, len(ids), replace=True).astype(int).tolist())
        cross_values = []
        within_values = []
        weights = []
        for case_id in sampled_cases:
            c = cross_arrays.get(int(case_id), np.array([], dtype=float))
            if not len(c):
                continue
            c_mean = float(np.mean(rng.choice(c, len(c), replace=True)))
            outcome_means = []
            for outcome in outcomes:
                w = within_arrays.get((int(case_id), outcome), np.array([], dtype=float))
                if len(w):
                    outcome_means.append(float(np.mean(rng.choice(w, len(w), replace=True))))
            if not outcome_means:
                continue
            cross_values.append(c_mean)
            within_values.append(float(np.mean(outcome_means)))
            weights.append(
                float(case_metadata.loc[case_metadata["case_id"].eq(case_id), "design_weight"].iloc[0])
            )
        values[repetition] = (
            float(np.average(cross_values, weights=weights) - np.average(within_values, weights=weights))
            if within_values
            else np.nan
        )
    return values


def bootstrap_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    alpha = 1 - confidence
    return float(np.quantile(array, alpha / 2)), float(np.quantile(array, 1 - alpha / 2))


def three_way_interpretation(
    estimate: float,
    interval: tuple[float, float],
    *,
    meaningful_effect: float,
    equivalence_half_width: float,
) -> str:
    lower, upper = interval
    if lower > meaningful_effect:
        return "material_sensitivity"
    if lower > -equivalence_half_width and upper < equivalence_half_width:
        return "practical_equivalence"
    return "inconclusive"


def model_pair_distances(
    wide: pd.DataFrame,
    *,
    prediction_table: pd.DataFrame | None = None,
    probability_tolerance: float = 0.05,
) -> pd.DataFrame:
    feature_columns = _feature_columns(wide)
    rows = []
    group_columns = [
        "outcome",
        "case_id",
        "background_id",
        "approximation_seed",
        "audit_stratum",
        "design_weight",
        "audit_inclusion_probability",
    ]
    group_columns = [column for column in group_columns if column in wide.columns]
    for key, frame in wide.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_columns, key_values))
        for left_index, right_index in itertools.combinations(range(len(frame)), 2):
            left = frame.iloc[left_index]
            right = frame.iloc[right_index]
            if left.get("model_id") == right.get("model_id"):
                continue
            vector_left = left[feature_columns].to_numpy(dtype=float)
            vector_right = right[feature_columns].to_numpy(dtype=float)
            bundle = attribution_distance_bundle(vector_left, vector_right, feature_columns)
            record = {
                **key_dict,
                "left_model_id": left.get("model_id"),
                "right_model_id": right.get("model_id"),
                "left_family": left.get("family"),
                "right_family": right.get("family"),
                "pair_type": "intra_family" if left.get("family") == right.get("family") else "inter_family",
                **bundle,
            }
            rows.append(record)
    result = pd.DataFrame(rows)
    if prediction_table is not None and not result.empty:
        prediction_key = ["case_id", "outcome", "model_id"]
        left_predictions = prediction_table[prediction_key + ["probability", "predicted_class"]].rename(
            columns={"model_id": "left_model_id", "probability": "left_probability", "predicted_class": "left_class"}
        )
        right_predictions = prediction_table[prediction_key + ["probability", "predicted_class"]].rename(
            columns={"model_id": "right_model_id", "probability": "right_probability", "predicted_class": "right_class"}
        )
        result = result.merge(left_predictions, on=["case_id", "outcome", "left_model_id"], how="left")
        result = result.merge(right_predictions, on=["case_id", "outcome", "right_model_id"], how="left")
        result["prediction_equivalent"] = (
            result["left_class"].eq(result["right_class"])
            & (result["left_probability"] - result["right_probability"]).abs().le(probability_tolerance)
        )
    return result


def rq2_delta_family(pair_distances: pd.DataFrame, distance_column: str = "sqrt_jsd") -> pd.DataFrame:
    def weighted_case_mean(frame: pd.DataFrame, group_columns: list[str]) -> pd.Series:
        case_columns = [*group_columns, "case_id"]
        aggregation = {distance_column: "mean"}
        if "design_weight" in frame.columns:
            aggregation["design_weight"] = "first"
        case_level = frame.groupby(case_columns, dropna=False, as_index=False).agg(aggregation)
        if "design_weight" not in case_level:
            case_level["design_weight"] = 1.0
        values: dict[Any, float] = {}
        for key, part in case_level.groupby(group_columns, dropna=False):
            values[key] = float(np.average(part[distance_column], weights=part["design_weight"]))
        return pd.Series(values, dtype=float)

    rows = []
    for outcome, frame in pair_distances.groupby("outcome"):
        inter = frame[frame["pair_type"].eq("inter_family")]
        intra = frame[frame["pair_type"].eq("intra_family")]
        # Equal family/family-pair weighting, then cases.
        inter = inter.assign(
            family_pair=inter.apply(
                lambda r: "|".join(sorted([str(r.left_family), str(r.right_family)])), axis=1
            )
        )
        inter_group = weighted_case_mean(inter, ["family_pair"]) if not inter.empty else pd.Series(dtype=float)
        intra_group = weighted_case_mean(intra, ["left_family"]) if not intra.empty else pd.Series(dtype=float)
        rows.append(
            {
                "outcome": outcome,
                "D_inter_family": float(inter_group.mean()) if len(inter_group) else np.nan,
                "D_intra_family": float(intra_group.mean()) if len(intra_group) else np.nan,
                "delta_family": float(inter_group.mean() - intra_group.mean())
                if len(inter_group) and len(intra_group)
                else np.nan,
                "n_inter_pairs": int(len(inter)),
                "n_intra_pairs": int(len(intra)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_rq2_delta_family(
    pair_distances: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = 260826,
    distance_column: str = "sqrt_jsd",
) -> pd.DataFrame:
    """Paired, audit-stratified case-and-pair bootstrap for RQ2.

    The bootstrap keeps the finite model-family design fixed, resamples cases
    within the frozen audit strata, and resamples the available pair records
    inside each sampled case. Family pairs and families are then weighted
    equally, matching the point estimand.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for outcome, frame in pair_distances.groupby("outcome"):
        metadata_columns = [
            column for column in ["case_id", "audit_stratum", "design_weight"] if column in frame.columns
        ]
        metadata = frame[metadata_columns].drop_duplicates("case_id").copy()
        if "audit_stratum" not in metadata:
            metadata["audit_stratum"] = "all_cases"
        if "design_weight" not in metadata:
            metadata["design_weight"] = 1.0
        prepared = frame.copy()
        prepared["family_pair"] = prepared.apply(
            lambda r: "|".join(sorted([str(r.left_family), str(r.right_family)])), axis=1
        )
        inter_arrays = {
            (str(group), int(case_id)): part[distance_column].dropna().to_numpy(dtype=float)
            for (group, case_id), part in prepared[prepared["pair_type"].eq("inter_family")].groupby(
                ["family_pair", "case_id"]
            )
        }
        intra_arrays = {
            (str(group), int(case_id)): part[distance_column].dropna().to_numpy(dtype=float)
            for (group, case_id), part in prepared[prepared["pair_type"].eq("intra_family")].groupby(
                ["left_family", "case_id"]
            )
        }
        inter_groups = sorted({key[0] for key in inter_arrays})
        intra_groups = sorted({key[0] for key in intra_arrays})
        weight_lookup = metadata.set_index("case_id")["design_weight"].astype(float).to_dict()
        for repetition in range(repetitions):
            sampled_cases: list[int] = []
            for _, stratum in metadata.groupby("audit_stratum", dropna=False, sort=True):
                ids = stratum["case_id"].astype(int).to_numpy()
                sampled_cases.extend(rng.choice(ids, len(ids), replace=True).astype(int).tolist())

            def design_group_mean(arrays, groups):
                group_means = []
                for group in groups:
                    values, weights = [], []
                    for case_id in sampled_cases:
                        candidates = arrays.get((group, int(case_id)), np.array([], dtype=float))
                        if len(candidates):
                            values.append(float(np.mean(rng.choice(candidates, len(candidates), replace=True))))
                            weights.append(float(weight_lookup[int(case_id)]))
                    if values:
                        group_means.append(float(np.average(values, weights=weights)))
                return float(np.mean(group_means)) if group_means else np.nan

            inter_mean = design_group_mean(inter_arrays, inter_groups)
            intra_mean = design_group_mean(intra_arrays, intra_groups)
            rows.append(
                {
                    "outcome": outcome,
                    "repetition": repetition,
                    "D_inter_family": inter_mean,
                    "D_intra_family": intra_mean,
                    "delta_family": inter_mean - intra_mean
                    if np.isfinite(inter_mean) and np.isfinite(intra_mean)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def adjust_pvalues(pvalues: Sequence[float], method: str) -> np.ndarray:
    mapping = {"holm": "holm", "benjamini_hochberg": "fdr_bh", "bh": "fdr_bh"}
    return multipletests(np.asarray(pvalues, dtype=float), method=mapping.get(method, method))[1]


def construct_driver_taxonomy(
    attributions: pd.DataFrame,
    faithfulness: pd.DataFrame | None = None,
    *,
    top_k: int = 5,
    inclusion_threshold: float = 0.70,
    direction_threshold: float = 0.70,
) -> pd.DataFrame:
    frame = attributions.copy()
    group_key = [
        column
        for column in ["case_id", "outcome", "family", "model_id", "refit_id", "background_id", "approximation_seed"]
        if column in frame.columns
    ]
    frame["rank"] = frame.groupby(group_key)["absolute_phi"].rank(method="first", ascending=False)
    frame["top_k"] = frame["rank"].le(top_k)
    summary = (
        frame.groupby(["feature_group", "outcome", "family"], dropna=False)
        .agg(
            top_k_inclusion=("top_k", "mean"),
            median_phi=("phi_log_odds", "median"),
            positive_fraction=("phi_log_odds", lambda s: float((s > 0).mean())),
            negative_fraction=("phi_log_odds", lambda s: float((s < 0).mean())),
            attribution_iqr=("phi_log_odds", lambda s: float(s.quantile(0.75) - s.quantile(0.25))),
        )
        .reset_index()
    )
    records = []
    for feature_group, group in summary.groupby("feature_group"):
        inclusion_ok = group["top_k_inclusion"].min() >= inclusion_threshold
        direction_stability = np.maximum(group["positive_fraction"], group["negative_fraction"])
        direction_ok = direction_stability.min() >= direction_threshold
        outcome_range = group.groupby("outcome")["median_phi"].median().max() - group.groupby("outcome")["median_phi"].median().min()
        family_range = group.groupby("family")["median_phi"].median().max() - group.groupby("family")["median_phi"].median().min()
        # Driver states are withheld until mandatory explanation controls exist.
        # A feature-level control table takes precedence; otherwise a global
        # control summary gates every feature together.
        faithfulness_ok = False
        if faithfulness is not None and "passes_control" in faithfulness.columns:
            if "feature_group" in faithfulness.columns:
                matched = faithfulness[faithfulness["feature_group"].eq(feature_group)]
                faithfulness_ok = bool(not matched.empty and matched["passes_control"].all())
            else:
                faithfulness_ok = bool(not faithfulness.empty and faithfulness["passes_control"].all())
        if not faithfulness_ok or not direction_ok:
            state = "unresolved_non_faithful"
        elif inclusion_ok and outcome_range <= family_range:
            state = "construct_robust"
        elif outcome_range > family_range:
            state = "outcome_family_specific"
        else:
            state = "model_contingent"
        records.append(
            {
                "feature_group": feature_group,
                "state": state,
                "minimum_top_k_inclusion": float(group["top_k_inclusion"].min()),
                "minimum_direction_stability": float(direction_stability.min()),
                "outcome_median_range": float(outcome_range),
                "family_median_range": float(family_range),
                "faithfulness_pass": faithfulness_ok,
            }
        )
    return pd.DataFrame(records)


def fit_crossed_attribution_models(attributions: pd.DataFrame) -> pd.DataFrame:
    """Fit feature-group-specific crossed mixed models as a descriptive decomposition.

    Outcome and family are fixed; startup is the grouping factor; configuration,
    refit, background, and approximation are variance components. Failed fits are
    retained with an explicit status rather than dropped.
    """
    import warnings
    import statsmodels.formula.api as smf

    frame = attributions.copy()
    required = {"feature_group", "phi_log_odds", "case_id", "outcome", "family"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Crossed decomposition missing columns: {sorted(missing)}")
    rows = []
    for feature_group, group in frame.groupby("feature_group"):
        group = group.copy()
        scale = float(np.median(np.abs(group["phi_log_odds"] - group["phi_log_odds"].median())))
        scale = scale if scale > 1e-8 else float(group["phi_log_odds"].std() or 1.0)
        group["phi_scaled"] = group["phi_log_odds"] / scale
        vc_formula = {}
        for name, column in [
            ("configuration", "config_id"),
            ("refit", "refit_id"),
            ("background", "background_id"),
            ("approximation", "approximation_seed"),
        ]:
            if column in group.columns and group[column].nunique() > 1:
                vc_formula[name] = f"0 + C({column})"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "phi_scaled ~ C(outcome) * C(family)",
                    data=group,
                    groups=group["case_id"],
                    vc_formula=vc_formula or None,
                    re_formula="1",
                )
                fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
            record = {
                "feature_group": feature_group,
                "status": "converged" if bool(getattr(fit, "converged", True)) else "not_converged",
                "robust_scale": scale,
                "startup_variance": float(np.asarray(fit.cov_re)[0, 0]) if np.asarray(fit.cov_re).size else np.nan,
                "residual_variance": float(fit.scale),
                "aic": float(fit.aic) if np.isfinite(fit.aic) else np.nan,
                "bic": float(fit.bic) if np.isfinite(fit.bic) else np.nan,
            }
            variance_components = np.asarray(getattr(fit, "vcomp", []), dtype=float)
            for index, name in enumerate(vc_formula):
                record[f"{name}_variance"] = float(variance_components[index]) if index < len(variance_components) else np.nan
            fixed = fit.fe_params.to_dict()
            record["fixed_effects_json"] = __import__("json").dumps({k: float(v) for k, v in fixed.items()}, sort_keys=True)
            rows.append(record)
        except Exception as exc:
            rows.append(
                {
                    "feature_group": feature_group,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "robust_scale": scale,
                }
            )
    return pd.DataFrame(rows)
