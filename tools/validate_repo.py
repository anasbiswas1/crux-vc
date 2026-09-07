#!/usr/bin/env python
"""Run static repository validation without accessing research data."""
from __future__ import annotations

import compileall
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "NOTEBOOKS.md",
    "VALIDATION_REPORT.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "CITATION.cff",
    "config/project.yaml",
    "config/model_grids.yaml",
    "config/feature_groups.yaml",
    "config/statistical_analysis.yaml",
    "config/notebook_contracts.yaml",
    "protocol/source_manifest.expected.json",
    "protocol/preregistration_template.yaml",
    "docs/EXECUTION_GUIDE.md",
    "docs/METHODS_IMPLEMENTATION_NOTES.md",
    "docs/DATA_DICTIONARY.md",
]


def main() -> None:
    if not (ROOT / ".cruxvc-root").exists():
        raise SystemExit("Repository marker missing")
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"Missing required file: {relative}")
        if path.suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    for directory in [ROOT / "src", ROOT / "tools", ROOT / "config", ROOT / "tests"]:
        if not compileall.compile_dir(directory, quiet=1):
            raise SystemExit(f"Python source compilation failed: {directory.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_notebooks.py")], check=True)
    print("CRUX-VC repository validation passed.")


if __name__ == "__main__":
    main()
