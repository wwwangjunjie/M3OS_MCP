from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOLTZ_OUTPUT_DIR = str(PROJECT_ROOT / "boltz_output")
BOLTZ_CACHE_DIR = str(PROJECT_ROOT / "boltz_data")
BOLTZ_PYTHON = os.environ.get(
    "BOLTZ_PYTHON",
    str(PROJECT_ROOT / ".venv" / "bin" / "python"),
)
BOLTZ_CLI = os.environ.get(
    "BOLTZ_CLI",
    str(PROJECT_ROOT / ".venv" / "bin" / "boltz"),
)

PLIP_SOURCE = str(PROJECT_ROOT / "plip_source")
PLIP_PYTHON = os.environ.get(
    "PLIP_PYTHON",
    str(Path(PLIP_SOURCE) / ".venv" / "bin" / "python"),
)
CAL_INTERACTION_SCRIPT = str(PROJECT_ROOT / "cal_interaction.py")

MCP_SERVER_PATH = str(PROJECT_ROOT / "mcp_server_improved.py")
