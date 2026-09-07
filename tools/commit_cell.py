#!/usr/bin/env python
"""Strip notebook outputs, commit, and push with the canonical identity."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/CRUX_Research")
REPO_ROOT = Path(os.environ.get("CRUX_REPO_ROOT", str(DRIVE_PROJECT_ROOT / "crux-vc")))
GIT_NAME = "Md Anas Biswas"
GIT_EMAIL = "anasbiswas@gmail.com"


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit('usage: python tools/commit_cell.py "commit message"')
    message = sys.argv[1].strip()

    for dotfile in (".gitconfig", ".git-credentials"):
        source = DRIVE_PROJECT_ROOT / dotfile
        if source.exists():
            shutil.copy(source, Path.home() / dotfile)
    credentials = Path.home() / ".git-credentials"
    if credentials.exists():
        os.chmod(credentials, 0o600)

    os.chdir(REPO_ROOT)
    subprocess.run(["git", "config", "user.name", GIT_NAME], check=True)
    subprocess.run(["git", "config", "user.email", GIT_EMAIL], check=True)

    for notebook in sorted(glob(str(REPO_ROOT / "notebooks" / "*.ipynb"))):
        subprocess.run(
            ["jupyter", "nbconvert", "--ClearOutputPreprocessor.enabled=True", "--inplace", notebook],
            check=True,
        )

    subprocess.run(["git", "add", "-A"], check=True)
    commit = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
    print(commit.stdout or commit.stderr)
    subprocess.run(["git", "push", "origin", "main"], check=True)


if __name__ == "__main__":
    main()
