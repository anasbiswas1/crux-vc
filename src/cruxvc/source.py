"""Acquire and verify the public Crunchbase mirror without trusting a hard-coded commit."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .io import write_json


def ensure_git_source(
    repository_url: str,
    destination: str | Path,
    *,
    branch: str = "master",
    update: bool = False,
) -> dict[str, Any]:
    destination = Path(destination)
    if not (destination / ".git").exists():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"Destination exists but is not a Git checkout: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", branch, "--single-branch", repository_url, str(destination)],
            check=True,
        )
    elif update:
        status = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain"], text=True
        ).strip()
        if status:
            raise RuntimeError(f"Refusing to update a dirty source checkout: {destination}")
        subprocess.run(["git", "-C", str(destination), "fetch", "origin", branch], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", branch], check=True)
        subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)

    commit = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
    ).strip()
    return {"path": str(destination), "commit": commit, "remote": remote, "branch": branch}


def locate_source_csvs(source_root: str | Path, filenames: list[str]) -> dict[str, Path]:
    source_root = Path(source_root)
    located: dict[str, Path] = {}
    for filename in filenames:
        candidates = list(source_root.rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"Could not locate {filename} beneath {source_root}")
        if len(candidates) > 1:
            candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
        located[filename] = candidates[0]
    return located


def verify_source_files(
    files: dict[str, Path], expected_hashes: dict[str, str], *, strict: bool = True
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for filename, path in files.items():
        actual = sha256_file(path)
        expected = expected_hashes.get(filename)
        match = expected is None or actual == expected
        report[filename] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": expected,
            "match": match,
            "bytes": path.stat().st_size,
        }
        if strict and expected and not match:
            raise RuntimeError(
                f"Source hash mismatch for {filename}: expected {expected}, found {actual}. "
                "Do not continue until provenance is resolved."
            )
    return report


def copy_source_files(files: dict[str, Path], destination: str | Path) -> dict[str, Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for filename, source in files.items():
        target = destination / filename
        if target.exists() and sha256_file(target) == sha256_file(source):
            copied[filename] = target
            continue
        shutil.copy2(source, target)
        copied[filename] = target
    return copied


def write_resolved_manifest(
    path: str | Path,
    checkout: dict[str, Any],
    verification: dict[str, dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "source_checkout": checkout,
        "files": verification,
        "all_expected_hashes_match": all(item["match"] for item in verification.values()),
        "extra": extra or {},
    }
    return write_json(payload, path)
