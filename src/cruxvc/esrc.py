"""Operational explanation-loss panels for the conditional ESRC extension.

The functions in this module intentionally separate *loss construction* from
risk calibration.  Every model, background, seed, reference panel and
perturbation budget must be frozen before the independent risk-calibration
block is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .explanations import GowerKNNDonorSampler, explain_dataset
from .faithfulness import faithfulness_audit, normalized_faithfulness_loss
from .metrics import normalized_absolute, sqrt_jsd_base2


@dataclass
class OperationalExplanationPanel:
    """Outputs from a frozen operational explanation-loss panel."""

    losses: pd.DataFrame
    attributions: pd.DataFrame
    faithfulness: pd.DataFrame
    audit: dict[str, Any]


def _group_order(*frames: pd.DataFrame) -> list[str]:
    names: set[str] = set()
    for frame in frames:
        if not frame.empty and "feature_group" in frame:
            names.update(frame["feature_group"].dropna().astype(str))
    return sorted(names)


def _mean_case_vectors(
    attributions: pd.DataFrame,
    *,
    group_order: Sequence[str],
    extra_index: Sequence[str] = (),
) -> pd.DataFrame:
    """Return one signed attribution vector per case and requested index cell."""
    required = {"case_id", "feature_group", "phi_log_odds", *extra_index}
    missing = required - set(attributions.columns)
    if missing:
        raise ValueError(f"Attribution frame lacks columns: {sorted(missing)}")
    index = ["case_id", *extra_index]
    wide = attributions.pivot_table(
        index=index,
        columns="feature_group",
        values="phi_log_odds",
        aggfunc="mean",
        fill_value=0.0,
    )
    return wide.reindex(columns=list(group_order), fill_value=0.0).sort_index()


def case_reference_panel_loss(
    deployed_attributions: pd.DataFrame,
    panel_attributions: pd.DataFrame,
    *,
    panel_id_column: str = "panel_id",
    group_order: Sequence[str] | None = None,
    expected_panel_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute finite-panel same-outcome instability for every case.

    Missing reference explanations are scored as distance one rather than being
    silently dropped.  The estimand is conditional on the finite, frozen panel.
    """
    groups = list(group_order or _group_order(deployed_attributions, panel_attributions))
    deployed = _mean_case_vectors(deployed_attributions, group_order=groups)
    if panel_id_column not in panel_attributions.columns:
        raise ValueError(f"Panel attribution frame lacks {panel_id_column!r}")
    panel = _mean_case_vectors(
        panel_attributions,
        group_order=groups,
        extra_index=[panel_id_column],
    )
    expected_panel_ids = (
        [str(value) for value in expected_panel_ids]
        if expected_panel_ids is not None
        else sorted(panel_attributions[panel_id_column].dropna().astype(str).unique())
    )
    rows: list[dict[str, Any]] = []
    for case_id, deployed_row in deployed.iterrows():
        distances: list[float] = []
        available = 0
        for panel_id in expected_panel_ids:
            key = (case_id, panel_id)
            if key in panel.index:
                distances.append(
                    sqrt_jsd_base2(
                        normalized_absolute(deployed_row.to_numpy(dtype=float)),
                        normalized_absolute(panel.loc[key].to_numpy(dtype=float)),
                    )
                )
                available += 1
            else:
                distances.append(1.0)
        rows.append(
            {
                "case_id": int(case_id),
                "L_S": float(np.mean(distances)) if distances else 1.0,
                "reference_panel_expected": int(len(expected_panel_ids)),
                "reference_panel_available": int(available),
            }
        )
    return pd.DataFrame(rows)


def case_construct_fragility_loss(
    deployment_attributions: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str]] = (("F18", "F36"), ("F36", "B+36")),
    group_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the mean matched confirmatory construct-distance per case."""
    if "outcome" not in deployment_attributions.columns:
        raise ValueError("Deployment attributions require an outcome column")
    groups = list(group_order or _group_order(deployment_attributions))
    wide = _mean_case_vectors(
        deployment_attributions,
        group_order=groups,
        extra_index=["outcome"],
    )
    case_ids = sorted(deployment_attributions["case_id"].dropna().astype(int).unique())
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        distances: list[float] = []
        available = 0
        for left, right in contrasts:
            left_key = (case_id, left)
            right_key = (case_id, right)
            if left_key in wide.index and right_key in wide.index:
                distances.append(
                    sqrt_jsd_base2(
                        normalized_absolute(wide.loc[left_key].to_numpy(dtype=float)),
                        normalized_absolute(wide.loc[right_key].to_numpy(dtype=float)),
                    )
                )
                available += 1
            else:
                distances.append(1.0)
        rows.append(
            {
                "case_id": int(case_id),
                "L_C": float(np.mean(distances)) if distances else 1.0,
                "construct_contrasts_expected": int(len(contrasts)),
                "construct_contrasts_available": int(available),
            }
        )
    return pd.DataFrame(rows)


def _safe_explain(
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
) -> tuple[pd.DataFrame, str | None]:
    try:
        frame = explain_dataset(
            model,
            cases,
            background,
            groups,
            feature_columns,
            model_metadata=model_metadata,
            background_id=background_id,
            approximation_seed=approximation_seed,
            n_orderings=n_orderings,
        )
        return frame, None
    except Exception as exc:  # the loss protocol assigns failures the worst loss
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def build_operational_explanation_panel(
    *,
    deployment_models: Mapping[str, Any],
    reference_models: Sequence[tuple[str, Any]],
    cases: pd.DataFrame,
    background: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
    sampler: GowerKNNDonorSampler,
    faithfulness_scale: float,
    approximation_seed: int,
    n_orderings: int,
    faithfulness_steps: Sequence[int],
    faithfulness_random_repetitions: int,
    seed: int,
    prediction_outcome: str = "F36",
) -> OperationalExplanationPanel:
    """Build the frozen 2009 operational loss panel used by ESRC.

    The three confirmatory deployment explanations define construct fragility;
    the finite same-outcome refit panel defines instability; conditional
    top-versus-random deletion defines the bounded faithfulness-proxy loss; and
    the frozen deployment classifier defines prediction error.
    """
    required_outcomes = {"F18", "F36", "B+36"}
    missing_models = required_outcomes - set(deployment_models)
    if missing_models:
        raise ValueError(f"Missing matched deployment models: {sorted(missing_models)}")
    if prediction_outcome not in cases.columns:
        raise ValueError(f"Cases lack prediction outcome {prediction_outcome!r}")
    if faithfulness_scale <= 0:
        raise ValueError("faithfulness_scale must be positive and development-frozen")

    deployment_parts: list[pd.DataFrame] = []
    panel_parts: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for outcome in sorted(required_outcomes):
        frame, error = _safe_explain(
            deployment_models[outcome],
            cases,
            background,
            groups,
            feature_columns,
            model_metadata={
                "outcome": outcome,
                "model_id": f"esrc_deployment_{outcome}",
                "refit_id": "deployment",
                "panel_id": "deployment",
                "analysis_role": "esrc_matched_deployment",
            },
            background_id="esrc_frozen_background",
            approximation_seed=approximation_seed,
            n_orderings=n_orderings,
        )
        if error:
            failures.append({"component": f"deployment_{outcome}", "error": error})
        else:
            deployment_parts.append(frame)

    for panel_id, model in reference_models:
        frame, error = _safe_explain(
            model,
            cases,
            background,
            groups,
            feature_columns,
            model_metadata={
                "outcome": prediction_outcome,
                "model_id": str(panel_id),
                "refit_id": str(panel_id),
                "panel_id": str(panel_id),
                "analysis_role": "esrc_same_outcome_reference_panel",
            },
            background_id="esrc_frozen_background",
            approximation_seed=approximation_seed,
            n_orderings=n_orderings,
        )
        if error:
            failures.append({"component": f"reference_{panel_id}", "error": error})
        else:
            panel_parts.append(frame)

    deployment_attr = (
        pd.concat(deployment_parts, ignore_index=True) if deployment_parts else pd.DataFrame()
    )
    panel_attr = pd.concat(panel_parts, ignore_index=True) if panel_parts else pd.DataFrame()
    group_order = sorted(groups)
    all_case_ids = pd.Index(cases["case_id"].astype(int), name="case_id")

    if not deployment_attr.empty and not panel_attr.empty:
        instability = case_reference_panel_loss(
            deployment_attr[deployment_attr["outcome"].eq(prediction_outcome)],
            panel_attr,
            group_order=group_order,
            expected_panel_ids=[panel_id for panel_id, _ in reference_models],
        ).set_index("case_id")
    else:
        instability = pd.DataFrame(
            {
                "L_S": 1.0,
                "reference_panel_expected": len(reference_models),
                "reference_panel_available": 0,
            },
            index=all_case_ids,
        )
    instability = instability.reindex(all_case_ids).fillna(
        {"L_S": 1.0, "reference_panel_expected": len(reference_models), "reference_panel_available": 0}
    )

    if not deployment_attr.empty:
        construct = case_construct_fragility_loss(
            deployment_attr,
            group_order=group_order,
        ).set_index("case_id")
    else:
        construct = pd.DataFrame(
            {"L_C": 1.0, "construct_contrasts_expected": 2, "construct_contrasts_available": 0},
            index=all_case_ids,
        )
    construct = construct.reindex(all_case_ids).fillna(
        {"L_C": 1.0, "construct_contrasts_expected": 2, "construct_contrasts_available": 0}
    )

    f36_attr = deployment_attr[deployment_attr.get("outcome", pd.Series(dtype=str)).eq(prediction_outcome)].copy()
    faithfulness = pd.DataFrame()
    if not f36_attr.empty:
        # One vector per case/group is required by the ranking audit.
        f36_mean = (
            f36_attr.groupby(["case_id", "feature_group"], as_index=False)
            .agg(phi_log_odds=("phi_log_odds", "mean"))
        )
        try:
            _, faithfulness = faithfulness_audit(
                deployment_models[prediction_outcome],
                cases,
                f36_mean,
                groups,
                sampler,
                feature_columns,
                steps=tuple(int(value) for value in faithfulness_steps),
                random_repetitions=int(faithfulness_random_repetitions),
                seed=int(seed),
            )
            faithfulness["L_F"] = normalized_faithfulness_loss(
                faithfulness["top_aopc"],
                faithfulness["random_aopc_mean"],
                faithfulness_scale,
            )
        except Exception as exc:
            failures.append({"component": "faithfulness", "error": f"{type(exc).__name__}: {exc}"})
            faithfulness = pd.DataFrame({"case_id": all_case_ids, "L_F": 1.0})
    else:
        faithfulness = pd.DataFrame({"case_id": all_case_ids, "L_F": 1.0})
    faithfulness = faithfulness.set_index("case_id").reindex(all_case_ids)
    faithfulness["L_F"] = faithfulness["L_F"].fillna(1.0).clip(0, 1)

    try:
        probabilities = deployment_models[prediction_outcome].predict_proba(cases)[:, 1]
        prediction_error = (
            (np.asarray(probabilities, dtype=float) >= 0.5).astype(int)
            != cases[prediction_outcome].to_numpy(dtype=int)
        ).astype(float)
    except Exception as exc:
        failures.append({"component": "prediction", "error": f"{type(exc).__name__}: {exc}"})
        prediction_error = np.ones(len(cases), dtype=float)

    losses = pd.DataFrame(index=all_case_ids)
    losses["L_S"] = instability["L_S"].astype(float).clip(0, 1)
    losses["L_C"] = construct["L_C"].astype(float).clip(0, 1)
    losses["L_F"] = faithfulness["L_F"].astype(float).clip(0, 1)
    losses["L_Y"] = prediction_error
    losses["reference_panel_expected"] = instability["reference_panel_expected"].astype(int)
    losses["reference_panel_available"] = instability["reference_panel_available"].astype(int)
    losses["construct_contrasts_expected"] = construct["construct_contrasts_expected"].astype(int)
    losses["construct_contrasts_available"] = construct["construct_contrasts_available"].astype(int)
    losses = losses.reset_index()

    combined_attributions = pd.concat(
        [frame for frame in [deployment_attr, panel_attr] if not frame.empty],
        ignore_index=True,
    ) if (not deployment_attr.empty or not panel_attr.empty) else pd.DataFrame()
    audit = {
        "n_cases": int(len(cases)),
        "n_feature_groups": int(len(groups)),
        "reference_panel_expected": int(len(reference_models)),
        "deployment_outcomes_expected": sorted(required_outcomes),
        "faithfulness_scale": float(faithfulness_scale),
        "approximation_seed": int(approximation_seed),
        "permutation_orderings": int(n_orderings),
        "faithfulness_random_repetitions": int(faithfulness_random_repetitions),
        "failed_components": failures,
        "complete_loss_rows": int(losses[["L_S", "L_C", "L_F", "L_Y"]].notna().all(axis=1).sum()),
    }
    return OperationalExplanationPanel(
        losses=losses,
        attributions=combined_attributions,
        faithfulness=faithfulness.reset_index(),
        audit=audit,
    )
