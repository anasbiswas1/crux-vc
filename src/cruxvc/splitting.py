"""Frozen chronological blocks and expanding-window development folds."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BLOCK_ORDER = ["development", "probability_calibration", "risk_calibration", "final_test"]


def assign_time_blocks(cohort: pd.DataFrame, block_config: dict[str, list[str]]) -> pd.DataFrame:
    result = cohort[["case_id", "company_permalink", "t0"]].copy()
    result["time_block"] = pd.NA
    for name in BLOCK_ORDER:
        start, end = map(pd.Timestamp, block_config[name])
        mask = result["t0"].between(start, end, inclusive="both")
        if result.loc[mask, "time_block"].notna().any():
            raise RuntimeError(f"Time block overlap detected for {name}")
        result.loc[mask, "time_block"] = name
    if result["time_block"].isna().any():
        missing = result.loc[result["time_block"].isna(), ["company_permalink", "t0"]]
        raise RuntimeError(f"Unassigned cohort rows: {missing.head().to_dict('records')}")
    result["time_block"] = pd.Categorical(result["time_block"], BLOCK_ORDER, ordered=True)
    return result.sort_values(["time_block", "t0", "company_permalink"]).reset_index(drop=True)


def expanding_window_folds(cohort: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.to_datetime(cohort["t0"])
    definitions = [
        ("fold_1", "2005-01-01", "2005-12-31", "2006-01-01", "2006-06-30"),
        ("fold_2", "2005-01-01", "2006-06-30", "2006-07-01", "2006-12-31"),
        ("fold_3", "2005-01-01", "2006-12-31", "2007-01-01", "2007-06-30"),
        ("fold_4", "2005-01-01", "2007-06-30", "2007-07-01", "2007-12-31"),
    ]
    rows: list[pd.DataFrame] = []
    for fold, train_start, train_end, val_start, val_end in definitions:
        train_mask = t0.between(pd.Timestamp(train_start), pd.Timestamp(train_end), inclusive="both")
        val_mask = t0.between(pd.Timestamp(val_start), pd.Timestamp(val_end), inclusive="both")
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            raise RuntimeError(f"Empty expanding-window split {fold}")
        train = cohort.loc[train_mask, ["case_id", "company_permalink", "t0"]].copy()
        train["fold_id"] = fold
        train["role"] = "train"
        validation = cohort.loc[val_mask, ["case_id", "company_permalink", "t0"]].copy()
        validation["fold_id"] = fold
        validation["role"] = "validation"
        rows.extend([train, validation])
    result = pd.concat(rows, ignore_index=True)
    return result[["fold_id", "role", "case_id", "company_permalink", "t0"]]


def block_count_table(cohort: pd.DataFrame, split_ids: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    merged = cohort.merge(split_ids[["case_id", "time_block"]], on="case_id", how="left")
    rows: list[dict[str, Any]] = []
    for block in BLOCK_ORDER:
        subset = merged[merged["time_block"].astype(str).eq(block)]
        row: dict[str, Any] = {"block": block, "n": int(len(subset))}
        row.update({outcome: int(subset[outcome].sum()) for outcome in outcomes})
        rows.append(row)
    total: dict[str, Any] = {"block": "total", "n": int(len(merged))}
    total.update({outcome: int(merged[outcome].sum()) for outcome in outcomes})
    rows.append(total)
    return pd.DataFrame(rows)


def structural_endpoint_gates(
    count_table: pd.DataFrame,
    outcomes: list[str],
    gate_config: dict[str, Any],
) -> pd.DataFrame:
    lookup = count_table.set_index("block")
    rows = []
    for outcome in outcomes:
        checks = {
            "development_positives": int(lookup.loc["development", outcome]),
            "probability_calibration_positives": int(lookup.loc["probability_calibration", outcome]),
            "risk_calibration_positives": int(lookup.loc["risk_calibration", outcome]),
            "final_test_positives": int(lookup.loc["final_test", outcome]),
        }
        passes = (
            checks["development_positives"] >= int(gate_config["development_positives_min"])
            and checks["probability_calibration_positives"]
            >= int(gate_config["probability_calibration_positives_min"])
            and checks["risk_calibration_positives"] >= int(gate_config["risk_calibration_positives_min"])
            and checks["final_test_positives"] >= int(gate_config["final_test_positives_min"])
        )
        rows.append({"outcome": outcome, **checks, "structural_pass": bool(passes)})
    return pd.DataFrame(rows)
