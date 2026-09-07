"""Colab/Drive defaults. Runtime discovery in :mod:`cruxvc.paths` overrides safely."""
from pathlib import Path
import os

REPO_URL = "https://github.com/anasbiswas1/crux-vc"
AUTHOR_EMAIL = "up2082724@myport.ac.uk"
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/CRUX_Research")
DEFAULT_REPO_ROOT = Path(
    os.environ.get("CRUX_REPO_ROOT", str(DRIVE_PROJECT_ROOT / "crux-vc"))
)
