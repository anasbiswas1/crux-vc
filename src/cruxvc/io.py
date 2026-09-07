"""Safe tabular and JSON I/O with schema sidecars."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .hashing import hash_jsonable, sha256_file


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def write_json(data: Any, path: str | Path) -> Path:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    return atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataframe_schema(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "nullable": bool(df[column].isna().any()),
            }
            for column in df.columns
        ],
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "schema_hash": hash_jsonable([(str(c), str(df[c].dtype)) for c in df.columns]),
    }


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    compression: str = "zstd",
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if suffix == ".parquet":
        df.to_parquet(tmp, index=index, compression=compression)
    elif suffix == ".csv":
        df.to_csv(tmp, index=index)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(tmp, index=index)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")
    os.replace(tmp, path)
    sidecar = dataframe_schema(df)
    sidecar["file_sha256"] = sha256_file(path)
    sidecar["metadata"] = metadata or {}
    write_json(sidecar, path.with_suffix(path.suffix + ".schema.json"))
    return path


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported table extension: {suffix}")


def require_columns(df: pd.DataFrame, required: list[str] | tuple[str, ...] | set[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
