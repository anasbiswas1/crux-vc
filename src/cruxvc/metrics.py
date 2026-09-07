"""Prediction, calibration, ranking, and explanation-distance metrics."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def clip_probabilities(probabilities: np.ndarray | Sequence[float], eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)


def logit(probabilities: np.ndarray | Sequence[float], eps: float = 1e-6) -> np.ndarray:
    p = clip_probabilities(probabilities, eps)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y_true: Sequence[int], probabilities: Sequence[float]) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        return np.nan, np.nan
    x = logit(probabilities).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def binary_prediction_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probabilities)
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, p)) if np.unique(y).size > 1 else np.nan
    auroc = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else np.nan
    brier = float(brier_score_loss(y, p))
    baseline_brier = prevalence * (1 - prevalence)
    brier_skill = 1 - brier / baseline_brier if baseline_brier > 0 else np.nan
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": float(len(y)),
        "positives": float(y.sum()),
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_lift": ap / prevalence if prevalence > 0 and np.isfinite(ap) else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": brier,
        "brier_skill": float(brier_skill),
        "auroc": auroc,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def top_budget_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], budgets: Iterable[float]
) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": np.asarray(probabilities, dtype=float)})
    frame = frame.sort_values(["p"], ascending=False, kind="mergesort").reset_index(drop=True)
    prevalence = frame["y"].mean()
    rows = []
    for budget in budgets:
        k = max(1, int(math.ceil(len(frame) * float(budget))))
        selected = frame.iloc[:k]
        precision = float(selected["y"].mean())
        recall = float(selected["y"].sum() / max(frame["y"].sum(), 1))
        rows.append(
            {
                "budget": float(budget),
                "k": int(k),
                "precision": precision,
                "recall": recall,
                "lift": precision / prevalence if prevalence > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def normalized_absolute(vector: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    values = np.abs(np.asarray(vector, dtype=float))
    total = values.sum()
    if total <= eps:
        return np.zeros_like(values)
    return values / (total + eps)


def sqrt_jsd_base2(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-12) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    p_mass = float(np.abs(p).sum())
    q_mass = float(np.abs(q).sum())
    if p_mass <= tolerance and q_mass <= tolerance:
        return 0.0
    if (p_mass <= tolerance) != (q_mass <= tolerance):
        return 1.0
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2.0))


def rank_biased_overlap(left: Sequence[str], right: Sequence[str], p: float = 0.9) -> float:
    if not 0 < p < 1:
        raise ValueError("RBO persistence p must be in (0, 1)")
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    seen_left: set[str] = set()
    seen_right: set[str] = set()
    score = 0.0
    for d in range(1, depth + 1):
        if d <= len(left):
            seen_left.add(str(left[d - 1]))
        if d <= len(right):
            seen_right.add(str(right[d - 1]))
        agreement = len(seen_left & seen_right) / d
        score += (1 - p) * (p ** (d - 1)) * agreement
    score += (p**depth) * (len(seen_left & seen_right) / depth)
    return float(score)


def ranked_groups(vector: Sequence[float], group_names: Sequence[str]) -> list[str]:
    order = np.argsort(-np.abs(np.asarray(vector, dtype=float)), kind="stable")
    return [str(group_names[index]) for index in order]


def attribution_distance_bundle(
    left: Sequence[float], right: Sequence[float], group_names: Sequence[str], top_k: int = 5
) -> dict[str, float]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_norm = normalized_absolute(left_array)
    right_norm = normalized_absolute(right_array)
    left_rank = ranked_groups(left_array, group_names)
    right_rank = ranked_groups(right_array, group_names)
    left_top = set(left_rank[:top_k])
    right_top = set(right_rank[:top_k])
    union = left_top | right_top
    jaccard = len(left_top & right_top) / len(union) if union else 1.0
    tau = kendalltau(
        [left_rank.index(name) for name in group_names],
        [right_rank.index(name) for name in group_names],
        variant="b",
    ).statistic
    cosine_distance = 0.0 if np.allclose(left_norm, right_norm) else float(cosine(left_norm, right_norm))
    nonzero = (np.abs(left_array) > 1e-12) & (np.abs(right_array) > 1e-12)
    sign_agreement = float(np.mean(np.sign(left_array[nonzero]) == np.sign(right_array[nonzero]))) if nonzero.any() else np.nan
    return {
        "sqrt_jsd": sqrt_jsd_base2(left_norm, right_norm),
        "rbo": rank_biased_overlap(left_rank, right_rank),
        "top_k_jaccard": float(jaccard),
        "kendall_tau_b": float(tau) if tau is not None else np.nan,
        "cosine_distance": cosine_distance,
        "sign_agreement": sign_agreement,
    }


def partial_area_risk_coverage(
    coverage: Sequence[float], risk: Sequence[float], lower: float = 0.50, upper: float = 0.80
) -> float:
    c = np.asarray(coverage, dtype=float)
    r = np.asarray(risk, dtype=float)
    mask = np.isfinite(c) & np.isfinite(r)
    c, r = c[mask], r[mask]
    if c.size < 2:
        return np.nan
    # A partial AURC is undefined when the observed curve does not span the
    # complete preregistered interval. Never silently extrapolate a deployment
    # curve beyond its realized acceptance support.
    if float(np.min(c)) > lower or float(np.max(c)) < upper:
        return np.nan
    collapsed = pd.DataFrame({"coverage": c, "risk": r}).groupby("coverage", as_index=False)["risk"].mean()
    c = collapsed["coverage"].to_numpy(dtype=float)
    r = collapsed["risk"].to_numpy(dtype=float)
    order = np.argsort(c)
    c, r = c[order], r[order]
    grid = np.unique(np.concatenate(([lower, upper], c[(c > lower) & (c < upper)])))
    interp = np.interp(grid, c, r)
    return float(trapezoid(interp, grid) / (upper - lower))
