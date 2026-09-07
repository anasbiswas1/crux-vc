"""Distribution-aware conditional perturbation, AOPC, and explanation controls."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .explanations import GowerKNNDonorSampler, model_log_odds


def _predicted_class_probability(model: Any, frame: pd.DataFrame, predicted_class: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Faithfulness audit requires a binary probabilistic classifier")
    return probabilities[:, int(predicted_class)]


def ranked_group_names(attribution_frame: pd.DataFrame, descending: bool = True) -> list[str]:
    frame = attribution_frame.groupby("feature_group", as_index=False)["phi_log_odds"].mean()
    return frame.assign(abs_phi=frame["phi_log_odds"].abs()).sort_values(
        "abs_phi", ascending=not descending, kind="mergesort"
    )["feature_group"].tolist()


def conditional_deletion_curve_one(
    model: Any,
    x: pd.Series,
    group_order: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    sampler: GowerKNNDonorSampler,
    feature_columns: Sequence[str],
    *,
    steps: Sequence[int],
    seed: int,
    target: str = "predicted_class_probability",
) -> pd.DataFrame:
    original = pd.DataFrame([x[list(feature_columns)]])
    original_probability = np.asarray(model.predict_proba(original), dtype=float)[0, 1]
    predicted_class = int(original_probability >= 0.5)
    baseline_log_odds = float(model_log_odds(model, original)[0])
    if target == "predicted_class_probability":
        baseline_target = float(_predicted_class_probability(model, original, predicted_class)[0])
    elif target == "log_odds":
        baseline_target = baseline_log_odds
    else:
        raise KeyError(f"Unknown faithfulness target {target!r}")
    rows = [{
        "removed_groups": 0,
        "log_odds": baseline_log_odds,
        "target_value": baseline_target,
        "predicted_class": predicted_class,
        "target": target,
        "absolute_change": 0.0,
    }]
    for step in sorted(set(int(value) for value in steps if value > 0)):
        removed = list(group_order[: min(step, len(group_order))])
        perturbed = sampler.replace_groups(x, removed, groups, seed=seed + step)
        perturbed_frame = pd.DataFrame([perturbed[list(feature_columns)]])
        value_log_odds = float(model_log_odds(model, perturbed_frame)[0])
        value_target = (
            float(_predicted_class_probability(model, perturbed_frame, predicted_class)[0])
            if target == "predicted_class_probability"
            else value_log_odds
        )
        rows.append(
            {
                "removed_groups": len(removed),
                "log_odds": value_log_odds,
                "target_value": value_target,
                "predicted_class": predicted_class,
                "target": target,
                "absolute_change": abs(baseline_target - value_target),
            }
        )
    return pd.DataFrame(rows)


def aopc(curve: pd.DataFrame, value_column: str = "absolute_change") -> float:
    nonzero = curve[curve["removed_groups"].gt(0)]
    return float(nonzero[value_column].mean()) if not nonzero.empty else 0.0


def faithfulness_audit(
    model: Any,
    cases: pd.DataFrame,
    attributions: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    sampler: GowerKNNDonorSampler,
    feature_columns: Sequence[str],
    *,
    steps: Sequence[int] = (1, 2, 3, 5),
    random_repetitions: int = 20,
    seed: int = 260826,
    target: str = "predicted_class_probability",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    curve_records = []
    summary_records = []
    for row in cases.itertuples(index=False):
        x = pd.Series(row._asdict())
        case_attr = attributions[attributions["case_id"].eq(int(x["case_id"]))]
        top_order = ranked_group_names(case_attr, descending=True)
        bottom_order = list(reversed(top_order))
        top_curve = conditional_deletion_curve_one(
            model, x, top_order, groups, sampler, feature_columns,
            steps=steps, seed=seed + int(x["case_id"]), target=target,
        )
        bottom_curve = conditional_deletion_curve_one(
            model,
            x,
            bottom_order,
            groups,
            sampler,
            feature_columns,
            steps=steps,
            seed=seed + int(x["case_id"]) + 100_000,
            target=target,
        )
        random_aopcs = []
        for repetition in range(random_repetitions):
            random_order = list(rng.permutation(top_order))
            random_curve = conditional_deletion_curve_one(
                model,
                x,
                random_order,
                groups,
                sampler,
                feature_columns,
                steps=steps,
                seed=seed + int(x["case_id"]) + 200_000 + repetition,
                target=target,
            )
            random_curve["ordering"] = "random"
            random_curve["random_repetition"] = repetition
            random_aopcs.append(aopc(random_curve))
            curve_records.append(random_curve.assign(case_id=int(x["case_id"])))
        top_curve["ordering"] = "top"
        top_curve["random_repetition"] = -1
        bottom_curve["ordering"] = "bottom"
        bottom_curve["random_repetition"] = -1
        curve_records.extend(
            [top_curve.assign(case_id=int(x["case_id"])), bottom_curve.assign(case_id=int(x["case_id"]))]
        )
        top_aopc = aopc(top_curve)
        bottom_aopc = aopc(bottom_curve)
        random_mean = float(np.mean(random_aopcs))
        scale = max(random_mean, 1e-8)
        summary_records.append(
            {
                "case_id": int(x["case_id"]),
                "top_aopc": top_aopc,
                "bottom_aopc": bottom_aopc,
                "random_aopc_mean": random_mean,
                "top_minus_random_aopc": top_aopc - random_mean,
                "top_minus_bottom_aopc": top_aopc - bottom_aopc,
                "normalized_top_minus_random": (top_aopc - random_mean) / scale,
                "faithfulness_target": target,
            }
        )
    return pd.concat(curve_records, ignore_index=True), pd.DataFrame(summary_records)


def normalized_faithfulness_loss(
    top_aopc: Sequence[float], random_aopc: Sequence[float], scale: float
) -> np.ndarray:
    difference = np.asarray(top_aopc, dtype=float) - np.asarray(random_aopc, dtype=float)
    score = np.clip(difference / max(float(scale), 1e-12), 0, 1)
    return 1 - score


def sanity_control_summary(
    real_attributions: pd.DataFrame,
    randomized_attributions: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["case_id", "feature_group"]
    real = real_attributions.groupby(keys)["phi_log_odds"].mean().rename("real_phi")
    random = randomized_attributions.groupby(keys)["phi_log_odds"].mean().rename("randomized_phi")
    merged = pd.concat([real, random], axis=1).dropna().reset_index()
    rows = []
    for feature_group, frame in merged.groupby("feature_group"):
        correlation = frame[["real_phi", "randomized_phi"]].corr().iloc[0, 1] if len(frame) > 2 else np.nan
        rows.append(
            {
                "feature_group": feature_group,
                "real_randomized_correlation": float(correlation),
                "mean_absolute_change": float((frame["real_phi"] - frame["randomized_phi"]).abs().mean()),
                "passes_control": bool(np.isnan(correlation) or correlation < 0.8),
            }
        )
    return pd.DataFrame(rows)
