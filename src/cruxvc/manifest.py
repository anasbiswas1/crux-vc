"""Completion manifests, protocol locks, and final-test access logging."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import STAGE_DEPENDENCIES, STAGE_NAMES
from .hashing import canonical_json_bytes, hash_jsonable, hash_paths, sha256_file
from .io import read_json, write_json
from .paths import ProjectPaths


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    from importlib import metadata

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def manifest_path(paths: ProjectPaths, stage_id: str, suffix: str | None = None) -> Path:
    tail = f"_{suffix}" if suffix else ""
    return paths.manifests / f"{stage_id}_{STAGE_NAMES[stage_id]}{tail}.json"


def require_dependencies(paths: ProjectPaths, stage_id: str) -> None:
    for dependency in STAGE_DEPENDENCIES.get(stage_id, []):
        candidate = manifest_path(paths, dependency)
        if not candidate.exists():
            raise RuntimeError(
                f"Stage {stage_id} requires completed stage {dependency}; missing {candidate}. "
                "Run notebooks in numerical order."
            )
        payload = read_json(candidate)
        if payload.get("status") != "completed":
            raise RuntimeError(f"Dependency {dependency} is not completed: {candidate}")


@dataclass
class StageRecorder:
    paths: ProjectPaths
    stage_id: str
    inputs: list[Path] = field(default_factory=list)
    permitted_blocks: list[str] = field(default_factory=list)
    config_payload: Mapping[str, Any] | None = None
    suffix: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.stage_id not in STAGE_NAMES:
            raise KeyError(f"Unknown stage id {self.stage_id}")
        require_dependencies(self.paths, self.stage_id)

    @property
    def path(self) -> Path:
        return manifest_path(self.paths, self.stage_id, self.suffix)

    def complete(self, outputs: Iterable[str | Path], extra: Mapping[str, Any] | None = None) -> Path:
        primary_output_paths = [Path(p) for p in outputs]
        missing = [str(p) for p in primary_output_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot complete stage with missing outputs: {missing}")
        # Table writers emit schema sidecars. Hash them automatically so a stage
        # cannot be considered complete if its declared output schema is altered.
        schema_sidecars = [
            path.with_suffix(path.suffix + ".schema.json")
            for path in primary_output_paths
            if path.with_suffix(path.suffix + ".schema.json").exists()
        ]
        output_paths = list(dict.fromkeys([*primary_output_paths, *schema_sidecars]))
        if self.path.exists():
            existing = read_json(self.path)
            old_outputs = existing.get("output_hashes", {})
            new_outputs = hash_paths(output_paths, self.paths.root)
            if old_outputs != new_outputs:
                raise RuntimeError(
                    f"Stage manifest already exists with different output hashes: {self.path}. "
                    "Archive the prior run rather than silently overwriting it."
                )
            return self.path
        payload = {
            "schema_version": 1,
            "stage_id": self.stage_id,
            "stage_name": STAGE_NAMES[self.stage_id],
            "status": "completed",
            "started_at_utc": self.started_at,
            "completed_at_utc": utc_now(),
            "permitted_blocks": self.permitted_blocks,
            "input_hashes": hash_paths([p for p in self.inputs if p.exists()], self.paths.root),
            "output_hashes": hash_paths(output_paths, self.paths.root),
            "config_hash": hash_jsonable(self.config_payload) if self.config_payload is not None else None,
            "git_commit": git_commit(self.paths.root),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(
                    ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "interpret", "shap"]
                ),
            },
            "notes": self.notes,
            "extra": dict(extra or {}),
        }
        return write_json(payload, self.path)


def _unsigned_lock_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in {"content_sha256", "hmac_sha256"}}


def create_protocol_lock(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    author_email: str,
    signing_key: str | None = None,
) -> Path:
    body = dict(payload)
    body["author_email"] = author_email
    body["created_at_utc"] = utc_now()
    body["signature_scheme"] = "HMAC-SHA256" if signing_key else "SHA256-ATTESTATION-ONLY"
    unsigned = _unsigned_lock_payload(body)
    body["content_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if signing_key:
        body["hmac_sha256"] = hmac.new(
            signing_key.encode("utf-8"), canonical_json_bytes(unsigned), hashlib.sha256
        ).hexdigest()
    else:
        body["hmac_sha256"] = None
    return write_json(body, path)


def verify_protocol_lock(
    path: str | Path,
    *,
    signing_key: str | None = None,
    require_hmac: bool = False,
) -> dict[str, Any]:
    payload = read_json(path)
    unsigned = _unsigned_lock_payload(payload)
    expected_hash = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if not hmac.compare_digest(expected_hash, str(payload.get("content_sha256", ""))):
        raise RuntimeError(f"Protocol lock content hash is invalid: {path}")
    signature = payload.get("hmac_sha256")
    if require_hmac and not signature:
        raise RuntimeError(f"Final execution requires an HMAC-signed protocol lock: {path}")
    if signature:
        if not signing_key:
            raise RuntimeError(f"Signing key required to verify protocol lock: {path}")
        expected_signature = hmac.new(
            signing_key.encode("utf-8"), canonical_json_bytes(unsigned), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, str(signature)):
            raise RuntimeError(f"Protocol lock HMAC is invalid: {path}")
    return payload


def signing_key_from_environment() -> str | None:
    value = os.environ.get("CRUX_PROTOCOL_SIGNING_KEY")
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("CRUX_PROTOCOL_SIGNING_KEY")
    except Exception:
        return None


def verify_test_access_log(path: str | Path) -> list[dict[str, Any]]:
    """Verify every entry hash and predecessor link in a final-test access log."""
    path = Path(path)
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous_hash = "0" * 64
    for index, entry in enumerate(entries):
        if entry.get("previous_hash") != previous_hash:
            raise RuntimeError(f"Test-access log predecessor mismatch at entry {index}: {path}")
        stored_hash = str(entry.get("entry_hash", ""))
        unsigned = {key: value for key, value in entry.items() if key != "entry_hash"}
        expected_hash = hash_jsonable(unsigned)
        if not hmac.compare_digest(stored_hash, expected_hash):
            raise RuntimeError(f"Test-access log entry hash mismatch at entry {index}: {path}")
        previous_hash = stored_hash
    return entries


def append_test_access_log(
    paths: ProjectPaths,
    *,
    stage_id: str,
    purpose: str,
    resources: Iterable[str | Path],
) -> Path:
    log_path = paths.protocol / "test_access_log.jsonl"
    entries = verify_test_access_log(log_path)
    previous_hash = entries[-1]["entry_hash"] if entries else "0" * 64
    entry = {
        "timestamp_utc": utc_now(),
        "stage_id": stage_id,
        "purpose": purpose,
        "resources": [paths.relative(Path(p)) if Path(p).is_absolute() else str(p) for p in resources],
        "git_commit": git_commit(paths.root),
        "previous_hash": previous_hash,
    }
    entry["entry_hash"] = hash_jsonable(entry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return log_path


def assert_file_hash(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")
