from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_PATH = str(PROJECT_ROOT / "tmp")
REINVENT_JOB_TMP_PATH = str(Path(TMP_PATH) / "reinvent")
POOL_PATH = str(Path(TMP_PATH) / "pool")
REINVENT_PATH = os.environ.get("REINVENT_ROOT", str(PROJECT_ROOT / "REINVENT4"))
REINVENT_PYTHON = os.environ.get(
    "REINVENT_PYTHON",
    str(Path(REINVENT_PATH) / ".venv" / "bin" / "python"),
)
