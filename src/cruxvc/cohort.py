"""Landmark cohort and horizon-aligned outcome construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from .constants import (
    B_PLUS_TYPES,
    C_PLUS_TYPES,
    LANDMARK_TYPES,
    QUALIFYING_FINANCING_TYPES,
)
from .events import earliest_date_type_audit


@dataclass
class CohortBuildResult:
    cohort: pd.DataFrame
    flow: pd.DataFrame
    overlap: pd.DataFrame
    multistate: pd.DataFrame
    audit: dict[str, Any]


def _within_months(date: pd.Series, t0: pd.Series, months: int) -> pd.Series:
    end = t0.map(lambda x: x + DateOffset(months=months))
    return date.gt(t0) & date.le(end)


def _event_label(
    cohort_base: pd.DataFrame,
    events: pd.DataFrame,
    *,
    months: int,
    allowed_types: set[str] | frozenset[str],
) -> tuple[pd.Series, pd.Series]:
    subset = events[events["round_type"].isin(allowed_types)][
        ["company_permalink", "funded_at", "round_type"]
    ]
    joined = cohort_base[["company_permalink", "t0"]].merge(subset, on="company_permalink", how="left")
    joined = joined[_within_months(joined["funded_at"], joined["t0"], months)]
    first = joined.groupby("company_permalink")["funded_at"].min()
    labels = cohort_base["company_permalink"].map(first).notna().astype("int8")
    event_date = cohort_base["company_permalink"].map(first)
    return labels, event_date


def build_landmark_cohort(
    funding_events: pd.DataFrame,
    acquisitions: pd.DataFrame,
    *,
    landmark_start: str = "2005-01-01",
    landmark_end: str = "2010-10-01",
    administrative_cutoff: str = "2013-10-01",
) -> CohortBuildResult:
    start = pd.Timestamp(landmark_start)
    end = pd.Timestamp(landmark_end)
    cutoff = pd.Timestamp(administrative_cutoff)
    flow_rows: list[dict[str, Any]] = []

    earliest = earliest_date_type_audit(funding_events)
    flow_rows.append({"step": "companies_with_dated_funding", "n": int(len(earliest))})
    earliest = earliest[earliest["earliest_unambiguous"]].copy()
    flow_rows.append({"step": "unambiguous_earliest_date_type", "n": int(len(earliest))})
    earliest["landmark_round_type"] = earliest["earliest_types"]
    earliest = earliest[earliest["landmark_round_type"].isin(LANDMARK_TYPES)].copy()
    flow_rows.append({"step": "earliest_type_angel_or_series_a", "n": int(len(earliest))})
    earliest = earliest[earliest["earliest_date"].between(start, end, inclusive="both")].copy()
    flow_rows.append({"step": "landmark_in_window", "n": int(len(earliest))})
    earliest["t0"] = earliest["earliest_date"]
    earliest["followup_end_36"] = earliest["t0"].map(lambda x: x + DateOffset(months=36))
    earliest = earliest[earliest["followup_end_36"].le(cutoff)].copy()
    flow_rows.append({"step": "complete_36_month_followup", "n": int(len(earliest))})

    first_acquisition = acquisitions.groupby("company_permalink")["acquired_at"].min()
    earliest["first_acquisition_at"] = earliest["company_permalink"].map(first_acquisition)
    earliest["acquired_before_t0"] = earliest["first_acquisition_at"].lt(earliest["t0"])
    earliest = earliest[~earliest["acquired_before_t0"].fillna(False)].copy()
    flow_rows.append({"step": "not_acquired_before_landmark", "n": int(len(earliest))})

    landmark_event = funding_events.rename(
        columns={
            "funded_at": "t0",
            "round_type": "landmark_round_type",
            "amount_max_usd": "landmark_amount_usd",
            "amount_sum_usd": "landmark_amount_sum_sensitivity_usd",
            "amount_conflict": "landmark_amount_conflict",
            "amount_missing": "landmark_amount_missing",
            "source_row_count": "landmark_source_row_count",
            "event_key": "landmark_event_key",
        }
    )
    keep = [
        "company_permalink",
        "t0",
        "landmark_round_type",
        "landmark_amount_usd",
        "landmark_amount_sum_sensitivity_usd",
        "landmark_amount_conflict",
        "landmark_amount_missing",
        "landmark_source_row_count",
        "landmark_event_key",
    ]
    cohort = earliest.merge(landmark_event[keep], on=["company_permalink", "t0", "landmark_round_type"], how="left")
    if cohort["landmark_event_key"].isna().any():
        raise RuntimeError("Landmark lineage failure: an eligible company lacks its exact aggregated event")
    if cohort.duplicated("company_permalink").any():
        raise RuntimeError("Cohort contains duplicate companies after deterministic landmark construction")

    cohort = cohort.sort_values(["t0", "company_permalink"]).reset_index(drop=True)
    cohort["case_id"] = np.arange(1, len(cohort) + 1, dtype=np.int64)

    cohort["F18"], cohort["F18_event_at"] = _event_label(
        cohort, funding_events, months=18, allowed_types=QUALIFYING_FINANCING_TYPES
    )
    cohort["F36"], cohort["F36_event_at"] = _event_label(
        cohort, funding_events, months=36, allowed_types=QUALIFYING_FINANCING_TYPES
    )
    cohort["B+36"], cohort["B+36_event_at"] = _event_label(
        cohort, funding_events, months=36, allowed_types=B_PLUS_TYPES
    )
    cohort["C+36"], cohort["C+36_event_at"] = _event_label(
        cohort, funding_events, months=36, allowed_types=C_PLUS_TYPES
    )
    acquisition_join = cohort[["company_permalink", "t0"]].merge(
        acquisitions[["company_permalink", "acquired_at"]], on="company_permalink", how="left"
    )
    acquisition_join = acquisition_join[
        _within_months(acquisition_join["acquired_at"], acquisition_join["t0"], 36)
    ]
    a_first = acquisition_join.groupby("company_permalink")["acquired_at"].min()
    cohort["A36_event_at"] = cohort["company_permalink"].map(a_first)
    cohort["A36"] = cohort["A36_event_at"].notna().astype("int8")
    cohort["Broad36"] = (cohort["F36"].eq(1) | cohort["A36"].eq(1)).astype("int8")

    cohort["multistate_36"] = np.select(
        [
            cohort["F36"].eq(0) & cohort["A36"].eq(0),
            cohort["F36"].eq(1) & cohort["A36"].eq(0),
            cohort["F36"].eq(0) & cohort["A36"].eq(1),
        ],
        ["no_recorded_event", "financing_only", "acquisition_only"],
        default="both",
    )
    cohort["acquired_before_financing"] = (
        cohort["A36_event_at"].notna()
        & cohort["F36_event_at"].notna()
        & cohort["A36_event_at"].lt(cohort["F36_event_at"])
    )

    labels = ["F18", "F36", "B+36", "C+36", "A36", "Broad36"]
    overlap = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for left in labels:
        for right in labels:
            overlap.loc[left, right] = int((cohort[left].eq(1) & cohort[right].eq(1)).sum())
    overlap.index.name = "outcome"
    overlap = overlap.reset_index()
    multistate = cohort["multistate_36"].value_counts(dropna=False).rename_axis("state").reset_index(name="n")
    flow = pd.DataFrame(flow_rows)
    audit = {
        "cohort_n": int(len(cohort)),
        "outcome_counts": {label: int(cohort[label].sum()) for label in labels},
        "nested_violation_Bplus_not_F36": int((cohort["B+36"].eq(1) & cohort["F36"].eq(0)).sum()),
        "nested_violation_Cplus_not_Bplus": int((cohort["C+36"].eq(1) & cohort["B+36"].eq(0)).sum()),
        "acquisition_before_financing_n": int(cohort["acquired_before_financing"].sum()),
    }
    if audit["nested_violation_Bplus_not_F36"] or audit["nested_violation_Cplus_not_Bplus"]:
        raise RuntimeError(f"Outcome nesting invariant failed: {audit}")
    return CohortBuildResult(cohort, flow, overlap, multistate, audit)
