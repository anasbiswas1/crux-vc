# === crux-vc session bootstrap: run first on every fresh runtime ===
import os, shutil, subprocess, importlib
from google.colab import drive

DRIVE_ROOT = "/content/drive/MyDrive/CRUX_Research"
REPO       = os.path.join(DRIVE_ROOT, "crux-vc")

drive.mount("/content/drive", force_remount=False)
for f in (".gitconfig", ".git-credentials"):
    shutil.copy(os.path.join(DRIVE_ROOT, f), os.path.join("/root", f))
os.chmod("/root/.git-credentials", 0o600)

os.chdir(REPO)
subprocess.run(["git", "pull", "origin", "main"], check=True)

for mod in ("interpret", "shap", "xgboost", "lightgbm", "openpyxl"):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run(["pip", "install", "-q", mod], check=True)

os.environ["PYTHONPATH"] = REPO + ":" + os.environ.get("PYTHONPATH", "")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())
