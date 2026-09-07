"""Top-K portfolio, utility, calibration, and selection-composition audits."""
from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import rank_biased_overlap


def ranked_case_ids(predictions: pd.DataFrame) -> list[int]:
    return (
        predictions.sort_values(["probability", "case_id"], ascending=[False, True], kind="mergesort")["case_id"]
        .astype(int)
        .tolist()
    )


def top_k_set(predictions: pd.DataFrame, budget: float) -> set[int]:
    k = max(1, int(np.ceil(len(predictions) * budget)))
    return set(ranked_case_ids(predictions)[:k])


def portfolio_overlap_audit(
    predictions: pd.DataFrame,
    *,
    budgets: Iterable[float],
    comparison_column: str = "outcome",
) -> pd.DataFrame:
    rows = []
    levels = sorted(predictions[comparison_column].unique())
    for left, right in itertools.combinations(levels, 2):
        left_frame = predictions[predictions[comparison_column].eq(left)]
        right_frame = predictions[predictions[comparison_column].eq(right)]
        common = sorted(set(left_frame["case_id"]) & set(right_frame["case_id"]))
        left_frame = left_frame[left_frame["case_id"].isin(common)]
        right_frame = right_frame[right_frame["case_id"].isin(common)]
        left_rank = ranked_case_ids(left_frame)
        right_rank = ranked_case_ids(right_frame)
        right_position = {case_id: position for position, case_id in enumerate(right_rank)}
        rank_reversal_mean = float(
            np.mean([abs(position - right_position[case_id]) for position, case_id in enumerate(left_rank)])
        )
        for budget in budgets:
            left_set = top_k_set(left_frame, budget)
            right_set = top_k_set(right_frame, budget)
            union = left_set | right_set
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "budget": float(budget),
                    "k": max(1, int(np.ceil(len(common) * budget))),
                    "jaccard": len(left_set & right_set) / len(union) if union else 1.0,
                    "rbo": rank_biased_overlap([str(x) for x in left_rank], [str(x) for x in right_rank]),
                    "threshold_crossings": len(left_set.symmetric_difference(right_set)),
                    "mean_absolute_rank_reversal": rank_reversal_mean,
                }
            )
    return pd.DataFrame(rows)


def selection_performance(
    predictions: pd.DataFrame,
    *,
    budgets: Iterable[float],
) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in ["outcome", "model_id"] if column in predictions.columns]
    for key, frame in predictions.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(group_columns, key_values))
        total_positive = max(int(frame["y_true"].sum()), 1)
        prevalence = float(frame["y_true"].mean())
        for budget in budgets:
            selected_ids = top_k_set(frame, budget)
            selected = frame[frame["case_id"].isin(selected_ids)]
            precision = float(selected["y_true"].mean())
            rows.append(
                {
                    **metadata,
                    "budget": float(budget),
                    "selected_n": int(len(selected)),
                    "precision": precision,
                    "recall": float(selected["y_true"].sum() / total_positive),
                    "lift": precision / prevalence if prevalence else np.nan,
                    "review_workload": int(len(frame) - len(selected)),
                }
            )
    return pd.DataFrame(rows)


def expected_utility(
    predictions: pd.DataFrame,
    *,
    budgets: Iterable[float],
    false_positive_costs: Sequence[float],
    false_negative_costs: Sequence[float],
    true_positive_value: float = 1.0,
) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in ["outcome", "model_id"] if column in predictions.columns]
    for key, frame in predictions.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(group_columns, key_values))
        for budget in budgets:
            selected_ids = top_k_set(frame, budget)
            selected = frame["case_id"].isin(selected_ids)
            y = frame["y_true"].astype(int)
            tp = int((selected & y.eq(1)).sum())
            fp = int((selected & y.eq(0)).sum())
            fn = int((~selected & y.eq(1)).sum())
            for fp_cost in false_positive_costs:
                for fn_cost in false_negative_costs:
                    utility = true_positive_value * tp - float(fp_cost) * fp - float(fn_cost) * fn
                    rows.append(
                        {
                            **metadata,
                            "budget": float(budget),
                            "false_positive_cost": float(fp_cost),
                            "false_negative_cost": float(fn_cost),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "utility": float(utility),
                            "utility_per_case": float(utility / len(frame)),
                        }
                    )
    return pd.DataFrame(rows)


def subgroup_selection_composition(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    subgroup_columns: Sequence[str],
    budgets: Iterable[float],
    minimum_cell: int = 10,
) -> pd.DataFrame:
    merged = predictions.merge(metadata[["case_id", *subgroup_columns]], on="case_id", how="left")
    rows = []
    group_columns = [column for column in ["outcome", "model_id"] if column in merged.columns]
    for key, frame in merged.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, key_values))
        for budget in budgets:
            selected_ids = top_k_set(frame, budget)
            selected = frame[frame["case_id"].isin(selected_ids)]
            for subgroup in subgroup_columns:
                total_counts = frame[subgroup].fillna("__MISSING__").value_counts()
                selected_counts = selected[subgroup].fillna("__MISSING__").value_counts()
                for level, total_n in total_counts.items():
                    selected_n = int(selected_counts.get(level, 0))
                    if total_n < minimum_cell:
                        level_reported = "__SMALL_CELL_COLLAPSED__"
                    else:
                        level_reported = str(level)
                    rows.append(
                        {
                            **base,
                            "budget": float(budget),
                            "subgroup": subgroup,
                            "level": level_reported,
                            "population_n": int(total_n),
                            "selected_n": selected_n,
                            "selection_rate": selected_n / total_n,
                        }
                    )
    result = pd.DataFrame(rows)
    collapsed = result.groupby(
        [*group_columns, "budget", "subgroup", "level"], dropna=False, as_index=False
    ).agg(population_n=("population_n", "sum"), selected_n=("selected_n", "sum"))
    collapsed["selection_rate"] = collapsed["selected_n"] / collapsed["population_n"]
    return collapsed
