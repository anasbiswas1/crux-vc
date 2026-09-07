"""Configuration loading, merging, and deterministic hashing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import hash_jsonable


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_project_config(root: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    config = load_yaml(root / "config" / "project.yaml")
    if override_path:
        config = deep_merge(config, load_yaml(override_path))
    return config


def active_compute_profile(config: Mapping[str, Any], profile: str | None = None) -> dict[str, Any]:
    import os

    selected = profile or os.environ.get("CRUX_PROFILE") or config["execution"]["compute_profile"]
    profiles = config.get("compute_profiles", {})
    if selected not in profiles:
        raise KeyError(f"Unknown CRUX compute profile {selected!r}; choose from {sorted(profiles)}")
    payload = copy.deepcopy(dict(profiles[selected]))
    payload["name"] = selected
    payload["is_final"] = selected == config["execution"].get("final_profile_name", "full")
    return payload


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_jsonable(config)


def dump_canonical_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
