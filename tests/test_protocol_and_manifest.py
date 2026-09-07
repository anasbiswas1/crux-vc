from __future__ import annotations

import json
from pathlib import Path

import pytest

from cruxvc.manifest import append_test_access_log, create_protocol_lock, verify_protocol_lock
from cruxvc.paths import ProjectPaths


def test_signed_protocol_lock_detects_tampering(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".cruxvc-root").write_text("CRUX-VC\n", encoding="utf-8")
    paths = ProjectPaths(root).ensure()
    lock = create_protocol_lock(
        {"lock_type": "design", "final_grade": True, "budget": 100},
        paths.locks / "design_lock.json",
        author_email="up2082724@myport.ac.uk",
        signing_key="test-secret",
    )
    verified = verify_protocol_lock(lock, signing_key="test-secret", require_hmac=True)
    assert verified["final_grade"] is True

    tampered = json.loads(lock.read_text(encoding="utf-8"))
    tampered["budget"] = 99
    lock.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash"):
        verify_protocol_lock(lock, signing_key="test-secret", require_hmac=True)


def test_test_access_log_is_hash_chained(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".cruxvc-root").write_text("CRUX-VC\n", encoding="utf-8")
    paths = ProjectPaths(root).ensure()
    resource = root / "resource.txt"
    resource.write_text("x", encoding="utf-8")
    log = append_test_access_log(paths, stage_id="11", purpose="first", resources=[resource])
    append_test_access_log(paths, stage_id="12", purpose="second", resources=[resource])
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["previous_hash"] == "0" * 64
    assert entries[1]["previous_hash"] == entries[0]["entry_hash"]


def test_test_access_log_detects_tampering(tmp_path: Path):
    from cruxvc.manifest import verify_test_access_log

    root = tmp_path / "repo"
    root.mkdir()
    (root / ".cruxvc-root").write_text("CRUX-VC\n", encoding="utf-8")
    paths = ProjectPaths(root).ensure()
    resource = root / "resource.txt"
    resource.write_text("x", encoding="utf-8")
    log = append_test_access_log(paths, stage_id="11", purpose="first", resources=[resource])
    append_test_access_log(paths, stage_id="12", purpose="second", resources=[resource])
    lines = log.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["purpose"] = "tampered"
    lines[1] = json.dumps(second, sort_keys=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="entry hash"):
        verify_test_access_log(log)
