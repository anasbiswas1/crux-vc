"""Source schema canonicalization and low-level data-quality audits."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def snake_case(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")
    return value.lower()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [snake_case(c) for c in result.columns]
    return result


def coalesce_alias(df: pd.DataFrame, target: str, aliases: Iterable[str], required: bool = False) -> pd.DataFrame:
    aliases = [snake_case(a) for a in aliases]
    candidates = [c for c in aliases if c in df.columns]
    if target in df.columns:
        return df
    if not candidates:
        if required:
            raise ValueError(f"No column found for {target}; aliases={aliases}")
        df[target] = pd.NA
        return df
    result = df.copy()
    value = result[candidates[0]]
    for column in candidates[1:]:
        value = value.combine_first(result[column])
    result[target] = value
    return result


def normalize_source_tables(
    companies: pd.DataFrame,
    rounds: pd.DataFrame,
    investments: pd.DataFrame,
    acquisitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    companies = canonicalize_columns(companies)
    rounds = canonicalize_columns(rounds)
    investments = canonicalize_columns(investments)
    acquisitions = canonicalize_columns(acquisitions)

    company_aliases = ["company_permalink", "permalink", "company_url", "company_id"]
    rounds = coalesce_alias(rounds, "company_permalink", company_aliases, required=True)
    investments = coalesce_alias(investments, "company_permalink", company_aliases, required=True)
    acquisitions = coalesce_alias(acquisitions, "company_permalink", company_aliases, required=True)
    companies = coalesce_alias(companies, "company_permalink", company_aliases, required=True)

    rounds = coalesce_alias(rounds, "funded_at", ["funded_at", "funding_date", "date"], required=True)
    rounds = coalesce_alias(
        rounds, "funding_round_type", ["funding_round_type", "round_type", "funding_type"], required=True
    )
    rounds = coalesce_alias(
        rounds, "raised_amount_usd", ["raised_amount_usd", "raised_amount", "amount_usd", "amount"]
    )

    investments = coalesce_alias(
        investments, "investor_permalink", ["investor_permalink", "investor_id", "investor_url"], required=True
    )
    investments = coalesce_alias(investments, "funded_at", ["funded_at", "funding_date", "date"], required=True)
    investments = coalesce_alias(
        investments,
        "funding_round_type",
        ["funding_round_type", "round_type", "funding_type"],
        required=True,
    )
    acquisitions = coalesce_alias(
        acquisitions, "acquired_at", ["acquired_at", "acquisition_date", "date"], required=True
    )
    acquisitions = coalesce_alias(
        acquisitions, "acquirer_permalink", ["acquirer_permalink", "acquirer_id", "acquirer_url"]
    )
    return companies, rounds, investments, acquisitions


def parse_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False).dt.normalize()


def schema_audit(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "table": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }


def read_source_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")
