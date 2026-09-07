"""Conformal baselines, matched-coverage gates, and finite-family risk control."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta, binom
from sklearn.model_selection import StratifiedKFold

from .metrics import partial_area_risk_coverage


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.size == 0:
        raise ValueError("Calibration scores are empty")
    rank = int(math.ceil((values.size + 1) * (1 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def split_conformal_binary_sets(
    calibration_probability: Sequence[float],
    calibration_y: Sequence[int],
    test_probability: Sequence[float],
    *,
    alpha: float = 0.10,
) -> pd.DataFrame:
    p_cal = np.asarray(calibration_probability, dtype=float)
    y_cal = np.asarray(calibration_y, dtype=int)
    true_probability = np.where(y_cal == 1, p_cal, 1 - p_cal)
    scores = 1 - true_probability
    threshold = finite_sample_quantile(scores, alpha)
    p_test = np.asarray(test_probability, dtype=float)
    include_zero = (1 - (1 - p_test)) <= threshold  # nonconformity for class 0 equals p_test
    include_one = (1 - p_test) <= threshold
    set_size = include_zero.astype(int) + include_one.astype(int)
    predicted = (p_test >= 0.5).astype(int)
    singleton_label = np.where(set_size == 1, np.where(include_one, 1, 0), -1)
    return pd.DataFrame(
        {
            "include_0": include_zero,
            "include_1": include_one,
            "set_size": set_size,
            "is_singleton": set_size == 1,
            "singleton_label": singleton_label,
            "predicted_class": predicted,
            "conformal_threshold": threshold,
            "alpha": alpha,
        }
    )


def mondrian_binary_sets(
    calibration_probability: Sequence[float],
    calibration_y: Sequence[int],
    test_probability: Sequence[float],
    *,
    alpha: float = 0.10,
    minimum_class_count: int = 20,
) -> pd.DataFrame:
    p_cal = np.asarray(calibration_probability, dtype=float)
    y_cal = np.asarray(calibration_y, dtype=int)
    thresholds = {}
    for label in (0, 1):
        class_probability = p_cal[y_cal == label] if label == 1 else 1 - p_cal[y_cal == label]
        if len(class_probability) < minimum_class_count:
            thresholds[label] = np.nan
        else:
            thresholds[label] = finite_sample_quantile(1 - class_probability, alpha)
    if not np.isfinite(thresholds[0]) or not np.isfinite(thresholds[1]):
        raise RuntimeError(
            f"Mondrian calibration counts are inadequate: n0={(y_cal == 0).sum()}, n1={(y_cal == 1).sum()}"
        )
    p_test = np.asarray(test_probability, dtype=float)
    include_zero = p_test <= thresholds[0]
    include_one = (1 - p_test) <= thresholds[1]
    set_size = include_zero.astype(int) + include_one.astype(int)
    return pd.DataFrame(
        {
            "include_0": include_zero,
            "include_1": include_one,
            "set_size": set_size,
            "is_singleton": set_size == 1,
            "singleton_label": np.where(set_size == 1, np.where(include_one, 1, 0), -1),
            "threshold_0": thresholds[0],
            "threshold_1": thresholds[1],
            "alpha": alpha,
        }
    )


def cross_fitted_split_conformal_binary_sets(
    probability: Sequence[float],
    y: Sequence[int],
    *,
    alpha: float = 0.10,
    folds: int = 5,
    seed: int = 260826,
) -> pd.DataFrame:
    """Out-of-fold conformal sets for calibration-block gate scores.

    Each case's set is produced from a threshold fitted on the other folds. This
    avoids using a calibration case's own label when its set size is later used to
    calibrate a selective gate threshold.
    """
    p = np.asarray(probability, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(p) != len(labels):
        raise ValueError("probability and y lengths differ")
    output: list[pd.DataFrame] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (fit_index, validation_index) in enumerate(splitter.split(p, labels)):
        frame = split_conformal_binary_sets(
            p[fit_index], labels[fit_index], p[validation_index], alpha=alpha
        )
        frame["_row_index"] = validation_index
        frame["crossfit_fold"] = fold_id
        output.append(frame)
    result = pd.concat(output, ignore_index=True).sort_values("_row_index").reset_index(drop=True)
    if not np.array_equal(result["_row_index"].to_numpy(dtype=int), np.arange(len(p))):
        raise RuntimeError("Cross-fitted conformal outputs do not align with input rows")
    return result.drop(columns="_row_index")


def cross_fitted_mondrian_binary_sets(
    probability: Sequence[float],
    y: Sequence[int],
    *,
    alpha: float = 0.10,
    folds: int = 5,
    seed: int = 260826,
    minimum_class_count: int = 20,
) -> pd.DataFrame:
    """Out-of-fold Mondrian sets for calibration-block gate scores."""
    p = np.asarray(probability, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(p) != len(labels):
        raise ValueError("probability and y lengths differ")
    output: list[pd.DataFrame] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (fit_index, validation_index) in enumerate(splitter.split(p, labels)):
        frame = mondrian_binary_sets(
            p[fit_index],
            labels[fit_index],
            p[validation_index],
            alpha=alpha,
            minimum_class_count=minimum_class_count,
        )
        frame["_row_index"] = validation_index
        frame["crossfit_fold"] = fold_id
        output.append(frame)
    result = pd.concat(output, ignore_index=True).sort_values("_row_index").reset_index(drop=True)
    if not np.array_equal(result["_row_index"].to_numpy(dtype=int), np.arange(len(p))):
        raise RuntimeError("Cross-fitted Mondrian outputs do not align with input rows")
    return result.drop(columns="_row_index")


def entropy(probability: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def standard_gate_scores(
    probability: Sequence[float],
    *,
    epistemic_variance: Sequence[float] | None = None,
    model_disagreement: Sequence[float] | None = None,
    density_score: Sequence[float] | None = None,
    predicted_explanation_loss: Sequence[float] | None = None,
) -> pd.DataFrame:
    p = np.asarray(probability, dtype=float)
    frame = pd.DataFrame(
        {
            "confidence_loss": 1 - np.maximum(p, 1 - p),
            "margin_loss": 1 - np.abs(2 * p - 1),
            "entropy": entropy(p),
        }
    )
    optional = {
        "epistemic_variance": epistemic_variance,
        "model_disagreement": model_disagreement,
        "density_ood": density_score,
        "predicted_explanation_loss": predicted_explanation_loss,
    }
    for name, values in optional.items():
        if values is not None:
            frame[name] = np.asarray(values, dtype=float)
    return frame


def calibrate_gate_threshold(
    score: Sequence[float], target_acceptance: float, *, lower_is_better: bool = True
) -> float:
    values = np.asarray(score, dtype=float)
    if not 0 < target_acceptance <= 1:
        raise ValueError("target_acceptance must be in (0, 1]")
    quantile = target_acceptance if lower_is_better else 1 - target_acceptance
    return float(np.quantile(values, quantile, method="higher" if lower_is_better else "lower"))


def apply_gate(score: Sequence[float], threshold: float, *, lower_is_better: bool = True) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    return values <= threshold if lower_is_better else values >= threshold


def accepted_risk(loss: Sequence[float], accepted: Sequence[bool]) -> float:
    values = np.asarray(loss, dtype=float)
    mask = np.asarray(accepted, dtype=bool)
    if mask.sum() == 0:
        return np.nan
    return float(values[mask].mean())


def deployable_gate_evaluation(
    calibration_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    test_losses: pd.DataFrame,
    coverages: Iterable[float],
    *,
    lower_is_better: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    lower_is_better = dict(lower_is_better or {})
    rows = []
    for gate in calibration_scores.columns:
        if gate not in test_scores.columns:
            continue
        orientation = lower_is_better.get(gate, True)
        for target in coverages:
            threshold = calibrate_gate_threshold(calibration_scores[gate], target, lower_is_better=orientation)
            accepted = apply_gate(test_scores[gate], threshold, lower_is_better=orientation)
            for loss_name in test_losses.columns:
                rows.append(
                    {
                        "gate": gate,
                        "target_calibration_acceptance": float(target),
                        "threshold": threshold,
                        "realized_test_coverage": float(accepted.mean()),
                        "accepted_n": int(accepted.sum()),
                        "reviewed_n": int((~accepted).sum()),
                        "loss": loss_name,
                        "accepted_risk": accepted_risk(test_losses[loss_name], accepted),
                    }
                )
    return pd.DataFrame(rows)


def retrospective_matched_coverage_evaluation(
    test_scores: pd.DataFrame,
    test_losses: pd.DataFrame,
    coverages: Iterable[float],
    *,
    lower_is_better: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    lower_is_better = dict(lower_is_better or {})
    rows = []
    n = len(test_scores)
    for gate in test_scores.columns:
        orientation = lower_is_better.get(gate, True)
        order = np.argsort(test_scores[gate].to_numpy()) if orientation else np.argsort(-test_scores[gate].to_numpy())
        for target in coverages:
            k = max(1, int(math.ceil(n * float(target))))
            accepted = np.zeros(n, dtype=bool)
            accepted[order[:k]] = True
            for loss_name in test_losses.columns:
                rows.append(
                    {
                        "gate": gate,
                        "target_test_coverage": float(target),
                        "realized_test_coverage": float(accepted.mean()),
                        "accepted_n": int(accepted.sum()),
                        "reviewed_n": int((~accepted).sum()),
                        "loss": loss_name,
                        "accepted_risk": accepted_risk(test_losses[loss_name], accepted),
                        "evaluation": "retrospective_identical_batch_coverage",
                    }
                )
    return pd.DataFrame(rows)


def summarize_risk_coverage(curves: pd.DataFrame, lower: float = 0.50, upper: float = 0.80) -> pd.DataFrame:
    rows = []
    coverage_column = (
        "realized_test_coverage" if "realized_test_coverage" in curves.columns else "target_test_coverage"
    )
    for (gate, loss), frame in curves.groupby(["gate", "loss"]):
        observed_lower = float(frame[coverage_column].min())
        observed_upper = float(frame[coverage_column].max())
        rows.append(
            {
                "gate": gate,
                "loss": loss,
                "normalized_partial_aurc": partial_area_risk_coverage(
                    frame[coverage_column], frame["accepted_risk"], lower, upper
                ),
                "coverage_lower": lower,
                "coverage_upper": upper,
                "observed_coverage_min": observed_lower,
                "observed_coverage_max": observed_upper,
                "full_interval_supported": bool(observed_lower <= lower and observed_upper >= upper),
            }
        )
    return pd.DataFrame(rows)


def clopper_pearson_lower(successes: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0:
        return np.nan
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def bounded_mean_lower_tail_pvalue(values: Sequence[float], rho: float) -> float:
    """Conservative Hoeffding–Bentkus-style p-value for H0: mean >= rho.

    The implemented p-value is the minimum of a one-sided Hoeffding bound and
    ``e * BinomialCDF(floor(n * mean); n, rho)``. It is used only for a frozen,
    finite candidate family; the design lock requires independent checking before
    a theorem claim is made.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.any((x < 0) | (x > 1)):
        return 1.0
    mean = float(x.mean())
    if mean >= rho:
        return 1.0
    n = len(x)
    hoeffding = math.exp(-2 * n * (rho - mean) ** 2)
    bentkus = math.e * float(binom.cdf(math.floor(n * mean), n, rho))
    return float(min(1.0, hoeffding, bentkus))


def coverage_lower_tail_pvalue(accepted: Sequence[bool], q_min: float) -> float:
    mask = np.asarray(accepted, dtype=bool)
    successes = int(mask.sum())
    # H0 q <= q_min versus H1 q > q_min: upper binomial tail.
    return float(binom.sf(successes - 1, len(mask), q_min))


def holm_rejections(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    rejected = np.zeros(len(p), dtype=bool)
    still_rejecting = True
    m = len(p)
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if still_rejecting and p[index] <= threshold:
            rejected[index] = True
        else:
            still_rejecting = False
    return rejected


def calibrate_finite_policy_family(
    calibration: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    loss_budgets: Mapping[str, float],
    q_min: float,
    familywise_delta: float,
    minimum_accepted: int = 100,
    utility_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Evaluate frozen policies whose rows provide score, threshold, and orientation.

    ``calibration`` must contain all score columns and bounded loss columns named in
    ``loss_budgets``. Rejected observations are excluded from conditional-risk
    means; coverage is tested separately over all observations.
    """
    rows = []
    for candidate in candidates.itertuples(index=False):
        score_name = str(candidate.score_name)
        if score_name not in score_columns or score_name not in calibration.columns:
            raise KeyError(f"Unknown candidate score {score_name}")
        lower = bool(candidate.lower_is_better)
        accepted = apply_gate(calibration[score_name], float(candidate.threshold), lower_is_better=lower)
        accepted_n = int(accepted.sum())
        component_p = {}
        empirical_risk = {}
        for loss_name, budget in loss_budgets.items():
            values = calibration.loc[accepted, loss_name].to_numpy(dtype=float)
            empirical_risk[loss_name] = float(values.mean()) if values.size else np.nan
            component_p[loss_name] = (
                bounded_mean_lower_tail_pvalue(values, float(budget))
                if accepted_n >= minimum_accepted
                else 1.0
            )
        component_p["coverage"] = coverage_lower_tail_pvalue(accepted, q_min)
        intersection_union_p = max(component_p.values())
        utility = (
            float(calibration.loc[accepted, utility_column].mean())
            if utility_column and accepted_n and utility_column in calibration.columns
            else float(accepted.mean())
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "score_name": score_name,
                "threshold": float(candidate.threshold),
                "lower_is_better": lower,
                "accepted_n": accepted_n,
                "coverage": float(accepted.mean()),
                "utility": utility,
                "intersection_union_p": intersection_union_p,
                **{f"risk_{name}": value for name, value in empirical_risk.items()},
                **{f"p_{name}": value for name, value in component_p.items()},
            }
        )
    result = pd.DataFrame(rows)
    result["certified"] = holm_rejections(result["intersection_union_p"], familywise_delta)
    eligible = result[result["certified"]].copy()
    selected = None
    if not eligible.empty:
        selected = eligible.sort_values(
            ["utility", "coverage", "score_name", "threshold"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        result["selected"] = result["candidate_id"].eq(selected["candidate_id"])
    else:
        result["selected"] = False
    return result, selected


def build_frozen_candidate_family(
    development_scores: pd.DataFrame,
    acceptance_grid: Sequence[float],
    *,
    orientations: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    orientations = dict(orientations or {})
    rows = []
    for score_name in development_scores.columns:
        lower = orientations.get(score_name, True)
        for acceptance in acceptance_grid:
            threshold = calibrate_gate_threshold(
                development_scores[score_name], acceptance, lower_is_better=lower
            )
            rows.append(
                {
                    "candidate_id": f"{score_name}__q{acceptance:.2f}",
                    "score_name": score_name,
                    "target_development_acceptance": float(acceptance),
                    "threshold": threshold,
                    "lower_is_better": lower,
                }
            )
    return pd.DataFrame(rows)
