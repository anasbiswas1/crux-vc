"""Notebook bootstrap and deterministic runtime metadata."""
from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import active_compute_profile, load_project_config
from .constants import STAGE_PERMITTED_BLOCKS
from .manifest import StageRecorder
from .paths import ProjectPaths


@dataclass
class NotebookContext:
    stage_id: str
    paths: ProjectPaths
    config: dict[str, Any]
    profile: dict[str, Any]
    recorder: StageRecorder


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_editable_install(root: Path) -> None:
    try:
        import cruxvc  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root)], check=True)


def bootstrap_notebook(
    stage_id: str,
    *,
    inputs: list[str | Path] | None = None,
    permitted_blocks: list[str] | None = None,
    suffix: str | None = None,
    notes: dict[str, Any] | None = None,
) -> NotebookContext:
    paths = ProjectPaths.discover().ensure()
    os.chdir(paths.root)
    src = str(paths.root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    config = load_project_config(paths.root, os.environ.get("CRUX_CONFIG_OVERRIDE"))
    profile = active_compute_profile(config)
    seed = int(config["execution"]["random_seed"])
    set_global_seed(seed)
    contract_path = paths.config / "notebook_contracts.yaml"
    contract = {}
    if contract_path.exists():
        import yaml

        contracts = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        contract = dict(contracts.get(stage_id, {}))
    declared_inputs: list[Path] = []
    for pattern in contract.get("inputs", []):
        matches = sorted(paths.root.glob(str(pattern))) if any(token in str(pattern) for token in "*?[") else [paths.root / str(pattern)]
        declared_inputs.extend(path for path in matches if path.exists() and path.is_file())
    explicit_inputs = [Path(p) for p in (inputs or [])]
    merged_inputs = list(dict.fromkeys([*declared_inputs, *explicit_inputs]))
    recorder = StageRecorder(
        paths=paths,
        stage_id=stage_id,
        inputs=merged_inputs,
        permitted_blocks=permitted_blocks or list(contract.get("permitted_blocks", STAGE_PERMITTED_BLOCKS.get(stage_id, ()))),
        config_payload={"project": config, "profile": profile, "notebook_contract": contract},
        suffix=suffix,
        notes={"contract": contract, **(notes or {})},
    )
    print(f"CRUX-VC stage {stage_id} | root={paths.root} | profile={profile['name']} | seed={seed}")
    return NotebookContext(stage_id, paths, config, profile, recorder)
