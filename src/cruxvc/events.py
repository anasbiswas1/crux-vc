"""Deterministic funding, investment, and acquisition event construction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schema import normalize_source_tables, parse_date_series


ROUND_TYPE_MAP = {
    "angel": "angel",
    "seed": "seed",
    "series a": "series-a",
    "series-a": "series-a",
    "a": "series-a",
    "series b": "series-b",
    "series-b": "series-b",
    "b": "series-b",
    "series c": "series-c-plus",
    "series-c": "series-c-plus",
    "c": "series-c-plus",
    "series d": "series-c-plus",
    "series-d": "series-c-plus",
    "series e": "series-c-plus",
    "series-e": "series-c-plus",
    "series f": "series-c-plus",
    "series-f": "series-c-plus",
    "series g": "series-c-plus",
    "series-g": "series-c-plus",
    "venture": "venture",
    "venture round": "venture",
    "private equity": "private-equity",
    "private-equity": "private-equity",
    "crowdfunding": "crowdfunding",
    "crowd": "crowdfunding",
    "other": "other",
    "grant": "grant",
    "debt financing": "debt-financing",
    "debt_financing": "debt-financing",
    "post ipo debt": "post-ipo",
    "post ipo equity": "post-ipo",
    "post_ipo_debt": "post-ipo",
    "post_ipo_equity": "post-ipo",
}


def normalize_round_type(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if text in ROUND_TYPE_MAP:
        return ROUND_TYPE_MAP[text]
    if re.match(r"^(series\s*)?[c-z](\+| round)?$", text):
        return "series-c-plus"
    if "post" in text and "ipo" in text:
        return "post-ipo"
    if text.startswith("series"):
        letter = re.sub(r"[^a-z]", "", text.replace("series", ""))[:1]
        if letter == "a":
            return "series-a"
        if letter == "b":
            return "series-b"
        if letter and letter >= "c":
            return "series-c-plus"
    return text.replace(" ", "-") or "unknown"


def _clean_identifier(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq("") | result.str.lower().isin(["nan", "none", "null"]))


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


@dataclass
class EventBuildResult:
    companies: pd.DataFrame
    funding_events: pd.DataFrame
    investment_edges: pd.DataFrame
    acquisitions: pd.DataFrame
    audit: dict[str, Any]


def build_event_tables(
    companies: pd.DataFrame,
    rounds: pd.DataFrame,
    investments: pd.DataFrame,
    acquisitions: pd.DataFrame,
    *,
    administrative_cutoff: str = "2013-10-01",
) -> EventBuildResult:
    companies, rounds, investments, acquisitions = normalize_source_tables(
        companies, rounds, investments, acquisitions
    )
    cutoff = pd.Timestamp(administrative_cutoff)

    # Company table: one deterministic record per permalink, preferring the row with
    # the largest number of nonmissing fields and then a stable row hash.
    companies = companies.drop_duplicates().copy()
    companies["company_permalink"] = _clean_identifier(companies["company_permalink"])
    companies = companies[companies["company_permalink"].notna()].copy()
    companies["_nonnull"] = companies.notna().sum(axis=1)
    companies["_row_repr"] = companies.astype("string").fillna("").agg("|".join, axis=1)
    companies = (
        companies.sort_values(["company_permalink", "_nonnull", "_row_repr"], ascending=[True, False, True])
        .drop_duplicates("company_permalink", keep="first")
        .drop(columns=["_nonnull", "_row_repr"])
        .reset_index(drop=True)
    )

    raw_round_rows = len(rounds)
    exact_round_duplicates = int(rounds.duplicated().sum())
    rounds = rounds.drop_duplicates().copy()
    rounds["company_permalink"] = _clean_identifier(rounds["company_permalink"])
    rounds["funded_at"] = parse_date_series(rounds["funded_at"])
    rounds["round_type"] = rounds["funding_round_type"].map(normalize_round_type)
    rounds["raised_amount_usd"] = _to_numeric(rounds["raised_amount_usd"])
    rounds = rounds[
        rounds["company_permalink"].notna()
        & rounds["funded_at"].notna()
        & rounds["funded_at"].le(cutoff)
    ].copy()
    rounds["event_key"] = (
        rounds["company_permalink"].astype(str)
        + "|"
        + rounds["funded_at"].dt.strftime("%Y-%m-%d")
        + "|"
        + rounds["round_type"].astype(str)
    )
    funding_events = (
        rounds.groupby(["company_permalink", "funded_at", "round_type", "event_key"], dropna=False)
        .agg(
            amount_max_usd=("raised_amount_usd", "max"),
            amount_sum_usd=("raised_amount_usd", "sum"),
            source_row_count=("raised_amount_usd", "size"),
            nonmissing_amount_count=("raised_amount_usd", "count"),
            distinct_amount_count=("raised_amount_usd", lambda s: int(s.dropna().nunique())),
        )
        .reset_index()
    )
    funding_events["amount_conflict"] = funding_events["distinct_amount_count"].gt(1)
    funding_events["amount_missing"] = funding_events["amount_max_usd"].isna()
    funding_events = funding_events.sort_values(
        ["company_permalink", "funded_at", "round_type", "event_key"]
    ).reset_index(drop=True)

    raw_investment_rows = len(investments)
    exact_investment_duplicates = int(investments.duplicated().sum())
    investments = investments.drop_duplicates().copy()
    investments["company_permalink"] = _clean_identifier(investments["company_permalink"])
    investments["investor_permalink"] = _clean_identifier(investments["investor_permalink"])
    investments["funded_at"] = parse_date_series(investments["funded_at"])
    investments["round_type"] = investments["funding_round_type"].map(normalize_round_type)
    malformed_investments = int(investments["company_permalink"].isna().sum())
    investments = investments[
        investments["company_permalink"].notna()
        & investments["investor_permalink"].notna()
        & investments["funded_at"].notna()
        & investments["funded_at"].le(cutoff)
    ].copy()
    investment_key = ["company_permalink", "investor_permalink", "funded_at", "round_type"]
    duplicate_investment_keys = int(investments.duplicated(investment_key).sum())
    investments = investments.sort_values(investment_key).drop_duplicates(investment_key, keep="first")
    investments["event_key"] = (
        investments["company_permalink"].astype(str)
        + "|"
        + investments["funded_at"].dt.strftime("%Y-%m-%d")
        + "|"
        + investments["round_type"].astype(str)
    )
    investment_edges = investments.reset_index(drop=True)

    raw_acquisition_rows = len(acquisitions)
    exact_acquisition_duplicates = int(acquisitions.duplicated().sum())
    acquisitions = acquisitions.drop_duplicates().copy()
    acquisitions["company_permalink"] = _clean_identifier(acquisitions["company_permalink"])
    acquisitions["acquirer_permalink"] = _clean_identifier(acquisitions["acquirer_permalink"])
    acquisitions["acquired_at"] = parse_date_series(acquisitions["acquired_at"])
    acquisitions = acquisitions[
        acquisitions["company_permalink"].notna()
        & acquisitions["acquired_at"].notna()
        & acquisitions["acquired_at"].le(cutoff)
    ].copy()
    acquisition_key = ["company_permalink", "acquired_at", "acquirer_permalink"]
    acquisitions = acquisitions.sort_values(acquisition_key).drop_duplicates(acquisition_key, keep="first")
    acquisitions = acquisitions.reset_index(drop=True)

    audit = {
        "raw_round_rows": int(raw_round_rows),
        "exact_round_duplicates": exact_round_duplicates,
        "valid_round_rows_after_exact_dedup": int(len(rounds)),
        "collapsed_funding_events": int(len(funding_events)),
        "raw_investment_rows": int(raw_investment_rows),
        "exact_investment_duplicates": exact_investment_duplicates,
        "malformed_investment_rows_missing_company": malformed_investments,
        "duplicate_investment_keys": duplicate_investment_keys,
        "valid_unique_investment_edges": int(len(investment_edges)),
        "raw_acquisition_rows": int(raw_acquisition_rows),
        "exact_acquisition_duplicates": exact_acquisition_duplicates,
        "valid_unique_acquisitions": int(len(acquisitions)),
        "latest_funding_event": funding_events["funded_at"].max(),
        "latest_acquisition_event": acquisitions["acquired_at"].max(),
    }
    return EventBuildResult(companies, funding_events, investment_edges, acquisitions, audit)


def earliest_date_type_audit(funding_events: pd.DataFrame) -> pd.DataFrame:
    earliest = funding_events.groupby("company_permalink")["funded_at"].min().rename("earliest_date")
    joined = funding_events.merge(earliest, on="company_permalink", how="inner")
    joined = joined[joined["funded_at"].eq(joined["earliest_date"])]
    audit = (
        joined.groupby("company_permalink")
        .agg(
            earliest_date=("earliest_date", "first"),
            earliest_type_count=("round_type", "nunique"),
            earliest_types=("round_type", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    audit["earliest_unambiguous"] = audit["earliest_type_count"].eq(1)
    return audit
