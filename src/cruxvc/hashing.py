"""Canonical file/object hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def hash_jsonable(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_paths(paths: Iterable[str | Path], root: str | Path | None = None) -> dict[str, str]:
    root_path = Path(root).resolve() if root is not None else None
    result: dict[str, str] = {}
    for item in sorted({Path(p).resolve() for p in paths}):
        key = str(item.relative_to(root_path)) if root_path and item.is_relative_to(root_path) else str(item)
        result[key] = sha256_file(item)
    return result
