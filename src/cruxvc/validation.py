"""Fail-fast scientific invariants and release checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import PROHIBITED_FEATURES
from .hashing import sha256_file


def assert_expected_counts(
    observed: pd.DataFrame,
    expected: Mapping[str, Mapping[str, int]],
    *,
    strict: bool,
) -> list[str]:
    messages = []
    lookup = observed.set_index("block")
    for block, values in expected.items():
        if block not in lookup.index:
            messages.append(f"Missing block {block}")
            continue
        for column, expected_value in values.items():
            observed_value = int(lookup.loc[block, column])
            if observed_value != int(expected_value):
                messages.append(
                    f"{block}.{column}: expected {expected_value}, observed {observed_value}"
                )
    if messages and strict:
        raise RuntimeError("Expected-count reproduction failed:\n" + "\n".join(messages))
    return messages


def assert_no_split_leakage(split_ids: pd.DataFrame) -> None:
    if split_ids["case_id"].duplicated().any():
        raise RuntimeError("A case appears in multiple frozen time blocks")
    required = {"development", "probability_calibration", "risk_calibration", "final_test"}
    observed = set(split_ids["time_block"].astype(str))
    if not required.issubset(observed):
        raise RuntimeError(f"Missing time blocks: {sorted(required - observed)}")


def assert_no_prohibited_features(feature_columns: Iterable[str]) -> None:
    prohibited = set(feature_columns) & PROHIBITED_FEATURES
    if prohibited:
        raise RuntimeError(f"Prohibited features detected: {sorted(prohibited)}")


def assert_graph_edges_precede_landmark(
    feature_lineage: pd.DataFrame,
    *,
    edge_date_column: str = "max_edge_date_used",
    landmark_column: str = "t0",
) -> None:
    if edge_date_column not in feature_lineage.columns:
        return
    invalid = feature_lineage[edge_date_column].notna() & feature_lineage[edge_date_column].ge(
        feature_lineage[landmark_column]
    )
    if invalid.any():
        raise RuntimeError("At least one graph feature used an edge at or after the landmark")


def assert_development_control_isolation(
    approved_matrix_ids: Iterable[str], control_matrix_ids: Iterable[str]
) -> None:
    overlap = set(approved_matrix_ids) & set(control_matrix_ids)
    if overlap:
        raise RuntimeError(f"Development control matrices share approved IDs: {sorted(overlap)}")


def verify_output_hashes(manifest: Mapping[str, Any], root: str | Path) -> None:
    root = Path(root)
    for relative, expected in manifest.get("output_hashes", {}).items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Output hash mismatch: {relative}")


def anonymize_case_ids(frame: pd.DataFrame, salt: str, column: str = "case_id") -> pd.DataFrame:
    import hashlib

    result = frame.copy()
    result[column] = result[column].map(
        lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()[:16]
    )
    return result


def release_scan(root: str | Path) -> list[str]:
    """Scan released *result data* for direct startup identifiers.

    Method documentation and source code may legitimately describe identifier
    field names, so they are not treated as leaked records. The scan inspects
    released CSV/Parquet tables and JSON result payloads, flags direct identifier
    columns/keys, and looks for Crunchbase organization URLs in values.
    """
    import re

    root = Path(root)
    problems: list[str] = []
    exact_identifier_fields = {
        "company_permalink",
        "company_name",
        "homepage_url",
        "founder_name",
        "organization_name",
        "startup_name",
    }
    identifier_fragments = ("permalink", "homepage", "founder_name", "company_name", "startup_name")
    organization_url = re.compile(r"(?:crunchbase\.com/(?:organization|company)/|/company/)[^\s,;]+", re.I)

    def field_is_identifier(value: Any) -> bool:
        name = str(value).strip().lower()
        return name in exact_identifier_fields or any(fragment in name for fragment in identifier_fragments)

    def scan_values(path: Path, frame: pd.DataFrame) -> None:
        bad_columns = [str(column) for column in frame.columns if field_is_identifier(column)]
        if bad_columns:
            problems.append(f"{path}: direct identifier columns {sorted(bad_columns)}")
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            sample = frame[column].dropna().astype(str).head(5000)
            if sample.str.contains(organization_url, regex=True).any():
                problems.append(f"{path}: Crunchbase organization URL detected in column {column}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = {part.lower() for part in path.relative_to(root).parts}
        if not ({"results", "tables", "figures"} & relative_parts):
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                scan_values(path, pd.read_csv(path, low_memory=False))
            elif suffix == ".parquet":
                scan_values(path, pd.read_parquet(path))
            elif suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                stack = [payload]
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if field_is_identifier(key):
                                problems.append(f"{path}: direct identifier key {key}")
                            stack.append(item)
                    elif isinstance(value, list):
                        stack.extend(value)
                    elif isinstance(value, str) and organization_url.search(value):
                        problems.append(f"{path}: Crunchbase organization URL detected")
        except Exception as exc:
            problems.append(f"{path}: release scan could not inspect file ({type(exc).__name__}: {exc})")
    return sorted(set(problems))
