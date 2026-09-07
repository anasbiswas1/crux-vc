#!/usr/bin/env python
"""Print the standard Colab bootstrap cell for a CRUX-VC notebook."""
from __future__ import annotations

import argparse


def bootstrap_source(stage_id: str, suffix_expression: str = "None") -> str:
    return f'''# Standard CRUX-VC Colab bootstrap. Git dotfiles restore from the Drive project root; tokens never appear in cells.
import os, subprocess, sys
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    pass

import shutil
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/CRUX_Research")
for _dotfile in (".gitconfig", ".git-credentials"):
    if (DRIVE_PROJECT_ROOT / _dotfile).exists():
        shutil.copy(DRIVE_PROJECT_ROOT / _dotfile, Path.home() / _dotfile)
if (Path.home() / ".git-credentials").exists():
    os.chmod(Path.home() / ".git-credentials", 0o600)

REPO_URL = "https://github.com/anasbiswas1/crux-vc"
REPO_ROOT = Path(os.environ.get("CRUX_REPO_ROOT", "/content/drive/MyDrive/CRUX_Research/crux-vc"))
if not (REPO_ROOT / ".cruxvc-root").exists():
    if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
        raise RuntimeError(f"{{REPO_ROOT}} exists but is not a CRUX-VC checkout")
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)], check=True)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import yaml, pandas, sklearn, pyarrow  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")], check=True)

from cruxvc.runtime import bootstrap_notebook
CTX = bootstrap_notebook("{stage_id}", suffix={suffix_expression})
P, CFG, PROFILE = CTX.paths, CTX.config, CTX.profile
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_id")
    parser.add_argument("--suffix-expression", default="None")
    args = parser.parse_args()
    print(bootstrap_source(args.stage_id, args.suffix_expression))


if __name__ == "__main__":
    main()
