"""Fail-graceful optional external benchmark loaders."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def locate_phbench_data(explicit_path: str | Path | None = None) -> dict[str, Any]:
    candidate = explicit_path or os.environ.get("PHBENCH_DATA_PATH")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists():
            return {"available": True, "path": str(path.resolve()), "source": "explicit_or_environment"}
        return {"available": False, "reason": f"PHBENCH_DATA_PATH does not exist: {path}"}
    return {
        "available": False,
        "reason": (
            "PHBench is optional and may require gated Hugging Face access. Set PHBENCH_DATA_PATH "
            "to an authorized local snapshot. The workflow will not scrape or bypass access controls."
        ),
    }


def read_phbench_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(path)
    files = list(root.rglob("*.parquet")) + list(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found under {root}")
    tables: dict[str, pd.DataFrame] = {}
    for file in files:
        key = file.stem
        tables[key] = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file, low_memory=False)
    return tables


def openml_benchmark_specs() -> list[dict[str, Any]]:
    return [
        {"name": "adult", "data_id": 1590},
        {"name": "bank-marketing", "data_id": 1461},
        {"name": "credit-g", "data_id": 31},
        {"name": "phoneme", "data_id": 1489},
    ]


def fetch_openml_benchmark(data_id: int):
    from sklearn.datasets import fetch_openml

    return fetch_openml(data_id=data_id, as_frame=True, parser="auto")


def locate_kaggle_snapshot_data(explicit_path: str | Path | None = None) -> dict[str, Any]:
    """Locate an authorized local startup-snapshot file without downloading it.

    The snapshot is an optional, deliberately leaky negative control. Users must
    obtain it under its own licence and expose it through
    ``KAGGLE_STARTUP_SNAPSHOT_PATH`` or an explicit path.
    """
    candidate = explicit_path or os.environ.get("KAGGLE_STARTUP_SNAPSHOT_PATH")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists():
            return {"available": True, "path": str(path.resolve()), "source": "explicit_or_environment"}
        return {"available": False, "reason": f"KAGGLE_STARTUP_SNAPSHOT_PATH does not exist: {path}"}
    return {
        "available": False,
        "reason": (
            "The Kaggle-derived startup snapshot is optional and is used only as a leaky-snapshot "
            "negative control. Set KAGGLE_STARTUP_SNAPSHOT_PATH to an authorized local CSV or "
            "Parquet file; the workflow will not download or bypass access controls."
        ),
    }


def read_snapshot_table(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read one CSV/Parquet snapshot, choosing the largest table in a directory."""
    root = Path(path)
    if root.is_file():
        selected = root
    else:
        candidates = list(root.rglob("*.parquet")) + list(root.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV/Parquet files found under {root}")
        selected = max(candidates, key=lambda item: item.stat().st_size)
    if selected.suffix.lower() == ".parquet":
        frame = pd.read_parquet(selected)
    else:
        frame = pd.read_csv(selected, low_memory=False)
    return frame, str(selected.resolve())
