"""Leakage-aware feature construction and feature-time ledger."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .constants import PROHIBITED_FEATURES


@dataclass
class FeatureBuildResult:
    strict: pd.DataFrame
    extended: pd.DataFrame
    ledger: pd.DataFrame
    audit: dict[str, Any]


def _safe_numeric_summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


def _metadata_columns(companies: pd.DataFrame) -> pd.DataFrame:
    columns = ["company_permalink"]
    aliases = {
        "founded_at": ["founded_at", "founded_date"],
        "category_code": ["category_code", "company_category_code"],
        "country_code": ["country_code", "company_country_code"],
        "region": ["region", "company_region", "state_code"],
    }
    result = companies[["company_permalink"]].copy()
    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in companies.columns), None)
        result[target] = companies[source] if source else pd.NA
    result["founded_at"] = pd.to_datetime(result["founded_at"], errors="coerce").dt.normalize()
    for column in ["category_code", "country_code", "region"]:
        result[column] = result[column].astype("string").str.strip().replace({"": pd.NA})
    return result


def build_investor_history_features(
    cohort: pd.DataFrame, investment_edges: pd.DataFrame
) -> pd.DataFrame:
    edges = investment_edges[
        ["company_permalink", "investor_permalink", "funded_at", "round_type", "event_key"]
    ].copy()
    edges = edges.sort_values(["funded_at", "event_key", "investor_permalink"])
    landmark_edges = edges.merge(
        cohort[["case_id", "company_permalink", "t0", "landmark_round_type", "landmark_event_key"]],
        left_on=["company_permalink", "funded_at", "round_type", "event_key"],
        right_on=["company_permalink", "t0", "landmark_round_type", "landmark_event_key"],
        how="inner",
    )
    landmark_map = landmark_edges.groupby("case_id")["investor_permalink"].apply(list).to_dict()

    dates = sorted(set(edges["funded_at"].dropna()) | set(cohort["t0"].dropna()))
    cohort_by_date = {date: frame for date, frame in cohort.groupby("t0")}
    edges_by_date = {date: frame for date, frame in edges.groupby("funded_at")}
    event_count: defaultdict[str, int] = defaultdict(int)
    companies_seen: defaultdict[str, set[str]] = defaultdict(set)
    graph = nx.Graph()
    records: list[dict[str, Any]] = []

    for date in dates:
        queries = cohort_by_date.get(date)
        if queries is not None:
            for row in queries.itertuples(index=False):
                investors = sorted(set(landmark_map.get(row.case_id, [])))
                counts = [event_count[i] for i in investors]
                company_counts = [len(companies_seen[i]) for i in investors]
                degrees = [int(graph.degree(i)) if graph.has_node(i) else 0 for i in investors]
                weighted = [
                    float(graph.degree(i, weight="weight")) if graph.has_node(i) else 0.0
                    for i in investors
                ]
                count_mean, count_max = _safe_numeric_summary(counts)
                company_mean, company_max = _safe_numeric_summary(company_counts)
                degree_mean, degree_max = _safe_numeric_summary(degrees)
                weighted_mean, weighted_max = _safe_numeric_summary(weighted)
                records.append(
                    {
                        "case_id": row.case_id,
                        "landmark_investor_count": len(investors),
                        "landmark_investor_edge_missing": int(len(investors) == 0),
                        "investor_prior_event_count_mean": count_mean,
                        "investor_prior_event_count_max": count_max,
                        "investor_prior_company_count_mean": company_mean,
                        "investor_prior_company_count_max": company_max,
                        "investor_prior_degree_mean": degree_mean,
                        "investor_prior_degree_max": degree_max,
                        "investor_prior_weighted_degree_mean": weighted_mean,
                        "investor_prior_weighted_degree_max": weighted_max,
                    }
                )

        date_edges = edges_by_date.get(date)
        if date_edges is None:
            continue
        for _, event in date_edges.groupby("event_key", sort=True):
            investors = sorted(set(event["investor_permalink"].dropna().astype(str)))
            company = str(event["company_permalink"].iloc[0])
            for investor in investors:
                event_count[investor] += 1
                companies_seen[investor].add(company)
                graph.add_node(investor)
            for left_index, left in enumerate(investors):
                for right in investors[left_index + 1 :]:
                    if graph.has_edge(left, right):
                        graph[left][right]["weight"] += 1.0
                    else:
                        graph.add_edge(left, right, weight=1.0)

    result = pd.DataFrame(records)
    if len(result) != len(cohort):
        missing = sorted(set(cohort["case_id"]) - set(result["case_id"]))
        raise RuntimeError(f"Investor-history feature builder missed cohort cases: {missing[:10]}")
    return result


def _window_summary(
    dates: np.ndarray,
    amounts: np.ndarray,
    t0: pd.Timestamp,
    days: int,
) -> tuple[int, float]:
    upper = np.searchsorted(dates, np.datetime64(t0), side="left")
    lower_date = np.datetime64(t0 - pd.Timedelta(days=days))
    lower = np.searchsorted(dates, lower_date, side="left")
    window = amounts[lower:upper]
    finite = window[np.isfinite(window)]
    return int(upper - lower), float(np.median(finite)) if finite.size else np.nan


def build_market_features(
    cohort: pd.DataFrame, funding_events: pd.DataFrame, companies: pd.DataFrame
) -> pd.DataFrame:
    events = funding_events.sort_values("funded_at").copy()
    dates = events["funded_at"].to_numpy(dtype="datetime64[ns]")
    amounts = events["amount_max_usd"].to_numpy(dtype=float)
    metadata = _metadata_columns(companies)
    events = events.merge(metadata[["company_permalink", "category_code", "region"]], on="company_permalink", how="left")
    category_groups = {
        key: (
            frame["funded_at"].sort_values().to_numpy(dtype="datetime64[ns]"),
            frame.sort_values("funded_at")["amount_max_usd"].to_numpy(dtype=float),
        )
        for key, frame in events[events["category_code"].notna()].groupby("category_code")
    }
    region_groups = {
        key: (
            frame["funded_at"].sort_values().to_numpy(dtype="datetime64[ns]"),
            frame.sort_values("funded_at")["amount_max_usd"].to_numpy(dtype=float),
        )
        for key, frame in events[events["region"].notna()].groupby("region")
    }
    cohort_meta = cohort[["case_id", "company_permalink", "t0"]].merge(metadata, on="company_permalink", how="left")
    rows = []
    for row in cohort_meta.itertuples(index=False):
        c90, _ = _window_summary(dates, amounts, row.t0, 90)
        c365, med365 = _window_summary(dates, amounts, row.t0, 365)
        ccat, mcat = 0, np.nan
        if pd.notna(row.category_code) and row.category_code in category_groups:
            ccat, mcat = _window_summary(*category_groups[row.category_code], row.t0, 365)
        creg, mreg = 0, np.nan
        if pd.notna(row.region) and row.region in region_groups:
            creg, mreg = _window_summary(*region_groups[row.region], row.t0, 365)
        rows.append(
            {
                "case_id": row.case_id,
                "market_deal_count_90d": c90,
                "market_deal_count_365d": c365,
                "market_median_amount_365d": med365,
                "category_deal_count_365d": ccat,
                "category_median_amount_365d": mcat,
                "region_deal_count_365d": creg,
                "region_median_amount_365d": mreg,
            }
        )
    return pd.DataFrame(rows)


def _investor_metadata_features(
    cohort: pd.DataFrame, investment_edges: pd.DataFrame
) -> pd.DataFrame:
    country_col = next(
        (c for c in ["investor_country_code", "investor_country", "country_code"] if c in investment_edges.columns),
        None,
    )
    edges = investment_edges.copy()
    if country_col is None:
        edges["_investor_country"] = pd.NA
    else:
        edges["_investor_country"] = edges[country_col].astype("string").str.strip().replace({"": pd.NA})
    landmark = edges.merge(
        cohort[["case_id", "company_permalink", "t0", "landmark_round_type", "landmark_event_key"]],
        left_on=["company_permalink", "funded_at", "round_type", "event_key"],
        right_on=["company_permalink", "t0", "landmark_round_type", "landmark_event_key"],
        how="inner",
    )
    result = (
        landmark.groupby("case_id")
        .agg(
            landmark_investor_country_count=("_investor_country", lambda s: int(s.dropna().nunique())),
            landmark_investor_metadata_missing_rate=("_investor_country", lambda s: float(s.isna().mean())),
        )
        .reset_index()
    )
    result = cohort[["case_id"]].merge(result, on="case_id", how="left")
    result["landmark_investor_country_count"] = result["landmark_investor_country_count"].fillna(0).astype(int)
    result["landmark_investor_metadata_missing_rate"] = result[
        "landmark_investor_metadata_missing_rate"
    ].fillna(1.0)
    return result


def feature_time_ledger() -> pd.DataFrame:
    entries = [
        ("landmark_amount_usd", "event-timed", "aggregated exact landmark funding event", "strict", True),
        ("log1p_landmark_amount_usd", "event-timed", "landmark amount", "strict", True),
        ("landmark_amount_missing", "event-timed", "landmark event", "strict", True),
        ("landmark_amount_conflict", "event-timed", "same date/type source rows", "strict", True),
        ("landmark_round_type", "event-timed", "exact landmark event", "strict", True),
        ("landmark_investor_count", "event-timed", "matched landmark investment edges", "strict", True),
        ("landmark_investor_edge_missing", "event-timed", "matched landmark investment edges", "strict", True),
        ("investor_prior_event_count_mean", "event-timed", "investment edges strictly before t0", "strict", True),
        ("investor_prior_event_count_max", "event-timed", "investment edges strictly before t0", "strict", True),
        ("investor_prior_company_count_mean", "event-timed", "investment edges strictly before t0", "strict", True),
        ("investor_prior_company_count_max", "event-timed", "investment edges strictly before t0", "strict", True),
        ("investor_prior_degree_mean", "event-timed", "co-investment graph strictly before t0", "strict", True),
        ("investor_prior_degree_max", "event-timed", "co-investment graph strictly before t0", "strict", True),
        ("investor_prior_weighted_degree_mean", "event-timed", "co-investment graph strictly before t0", "strict", True),
        ("investor_prior_weighted_degree_max", "event-timed", "co-investment graph strictly before t0", "strict", True),
        ("landmark_year", "event-timed", "t0", "strict", True),
        ("landmark_quarter", "event-timed", "t0", "strict", True),
        ("market_deal_count_90d", "event-timed", "events strictly before t0", "strict", True),
        ("market_deal_count_365d", "event-timed", "events strictly before t0", "strict", True),
        ("market_median_amount_365d", "event-timed", "events strictly before t0", "strict", True),
        ("event_source_conflict_count", "event-timed", "landmark event quality", "strict", True),
        ("founding_age_days", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("founding_date_missing", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("founding_date_invalid", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("category_code", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("country_code", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("region", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
        ("landmark_investor_country_count", "snapshot-static with uncertain historical availability", "investor snapshot", "extended", True),
        ("landmark_investor_metadata_missing_rate", "snapshot-static with uncertain historical availability", "investor snapshot", "extended", True),
        ("category_deal_count_365d", "snapshot-static with uncertain historical availability", "category tags plus prior events", "extended", True),
        ("category_median_amount_365d", "snapshot-static with uncertain historical availability", "category tags plus prior events", "extended", True),
        ("region_deal_count_365d", "snapshot-static with uncertain historical availability", "region tags plus prior events", "extended", True),
        ("region_median_amount_365d", "snapshot-static with uncertain historical availability", "region tags plus prior events", "extended", True),
        ("company_metadata_missing_count", "snapshot-static with uncertain historical availability", "company snapshot", "extended", True),
    ]
    prohibited = [
        (feature, "prohibited outcome-accumulating", "snapshot or post-landmark", "prohibited", False)
        for feature in sorted(PROHIBITED_FEATURES)
    ]
    return pd.DataFrame(
        entries + prohibited,
        columns=["feature", "availability_class", "source_time_rule", "feature_set", "approved"],
    )


def build_features(
    cohort: pd.DataFrame,
    companies: pd.DataFrame,
    funding_events: pd.DataFrame,
    investment_edges: pd.DataFrame,
) -> FeatureBuildResult:
    metadata = _metadata_columns(companies)
    base = cohort[
        [
            "case_id",
            "company_permalink",
            "t0",
            "landmark_round_type",
            "landmark_amount_usd",
            "landmark_amount_conflict",
            "landmark_amount_missing",
            "landmark_source_row_count",
        ]
    ].copy()
    base["log1p_landmark_amount_usd"] = np.log1p(base["landmark_amount_usd"].clip(lower=0))
    base["landmark_year"] = base["t0"].dt.year.astype(int)
    base["landmark_quarter"] = base["t0"].dt.quarter.astype(int)
    base["event_source_conflict_count"] = (
        base["landmark_amount_conflict"].astype(int) + base["landmark_source_row_count"].gt(1).astype(int)
    )

    investor = build_investor_history_features(cohort, investment_edges)
    market = build_market_features(cohort, funding_events, companies)
    strict = base.merge(investor, on="case_id", how="left").merge(market, on="case_id", how="left")

    extended = strict.merge(metadata, on="company_permalink", how="left")
    extended["founding_date_missing"] = extended["founded_at"].isna().astype(int)
    extended["founding_date_invalid"] = extended["founded_at"].gt(extended["t0"]).fillna(False).astype(int)
    age = (extended["t0"] - extended["founded_at"]).dt.days
    extended["founding_age_days"] = age.mask(age.lt(0))
    extended = extended.drop(columns=["founded_at"])
    investor_meta = _investor_metadata_features(cohort, investment_edges)
    extended = extended.merge(investor_meta, on="case_id", how="left")
    metadata_fields = ["category_code", "country_code", "region"]
    extended["company_metadata_missing_count"] = extended[metadata_fields].isna().sum(axis=1).astype(int)

    strict_columns = [
        "case_id",
        "company_permalink",
        "t0",
        "landmark_amount_usd",
        "log1p_landmark_amount_usd",
        "landmark_amount_missing",
        "landmark_amount_conflict",
        "landmark_round_type",
        "landmark_investor_count",
        "landmark_investor_edge_missing",
        "investor_prior_event_count_mean",
        "investor_prior_event_count_max",
        "investor_prior_company_count_mean",
        "investor_prior_company_count_max",
        "investor_prior_degree_mean",
        "investor_prior_degree_max",
        "investor_prior_weighted_degree_mean",
        "investor_prior_weighted_degree_max",
        "landmark_year",
        "landmark_quarter",
        "market_deal_count_90d",
        "market_deal_count_365d",
        "market_median_amount_365d",
        "event_source_conflict_count",
    ]
    extended_additional = [
        "founding_age_days",
        "founding_date_missing",
        "founding_date_invalid",
        "category_code",
        "country_code",
        "region",
        "landmark_investor_country_count",
        "landmark_investor_metadata_missing_rate",
        "category_deal_count_365d",
        "category_median_amount_365d",
        "region_deal_count_365d",
        "region_median_amount_365d",
        "company_metadata_missing_count",
    ]
    strict = strict[strict_columns].sort_values("case_id").reset_index(drop=True)
    extended = extended[strict_columns + extended_additional].sort_values("case_id").reset_index(drop=True)
    ledger = feature_time_ledger()
    audit = {
        "strict_rows": int(len(strict)),
        "extended_rows": int(len(extended)),
        "landmark_amount_missing": int(strict["landmark_amount_missing"].sum()),
        "founding_date_missing": int(extended["founding_date_missing"].sum()),
        "founding_date_invalid": int(extended["founding_date_invalid"].sum()),
        "landmark_investor_edge_missing": int(strict["landmark_investor_edge_missing"].sum()),
    }
    validate_feature_matrix(strict, ledger, feature_set="strict")
    validate_feature_matrix(extended, ledger, feature_set="extended")
    return FeatureBuildResult(strict, extended, ledger, audit)


def validate_feature_matrix(df: pd.DataFrame, ledger: pd.DataFrame, *, feature_set: str) -> None:
    identifiers = {"case_id", "company_permalink", "t0"}
    model_features = set(df.columns) - identifiers
    prohibited = model_features & PROHIBITED_FEATURES
    if prohibited:
        raise RuntimeError(f"Prohibited features entered {feature_set} matrix: {sorted(prohibited)}")
    approved = set(ledger.loc[ledger["approved"], "feature"])
    unregistered = model_features - approved
    if unregistered:
        raise RuntimeError(f"Features lack an approved availability class: {sorted(unregistered)}")
    duplicates = df["case_id"].duplicated().sum()
    if duplicates:
        raise RuntimeError(f"Feature matrix has {duplicates} duplicate case IDs")
