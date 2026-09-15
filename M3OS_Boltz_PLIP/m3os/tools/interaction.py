from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

from configs.tool_globals import (
    BOLTZ_OUTPUT_DIR,
    CAL_INTERACTION_SCRIPT,
    PLIP_PYTHON,
    PROJECT_ROOT,
)


SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


def analyze_complex_interactions(
    structure_path: str,
    subprocess_timeout_seconds: float | None = None,
):
    """Analyze an existing protein-ligand structure with PLIP.

    Args:
        structure_path: Local PDB, CIF, or mmCIF file containing a
            protein-ligand complex.

    Returns:
        A dictionary containing the source structure path, the saved PLIP JSON
        path, and the interaction categories extracted by PLIP.
    """
    if not isinstance(structure_path, str) or not structure_path.strip():
        return "Error: 'structure_path' must be a non-empty string."

    source = Path(structure_path).expanduser().resolve()
    if not source.is_file():
        return f"Error: Structure file not found: {source}"
    if source.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_STRUCTURE_SUFFIXES))
        return (
            f"Error: Unsupported structure format '{source.suffix}'. "
            f"Expected one of: {supported}."
        )

    output_dir = Path(BOLTZ_OUTPUT_DIR) / "plip_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = secrets.token_hex(5)
    output_json = output_dir / f"{source.stem}_{unique_id}_plip.json"

    plip_command = [
        PLIP_PYTHON,
        CAL_INTERACTION_SCRIPT,
        "-i",
        str(source),
        "-o",
        str(output_json),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            plip_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=subprocess_timeout_seconds,
        )
        if not output_json.is_file():
            return (
                "PLIP finished but the expected interaction JSON file was "
                f"not found: {output_json}"
            )

        with output_json.open("r", encoding="utf-8") as handle:
            interaction_data = json.load(handle)

        return {
            "status": "success",
            "structure_path": str(source),
            "interaction_json_path": str(output_json.resolve()),
            "interactions": interaction_data,
        }
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if exc.stderr else "No stderr"
        return f"PLIP command failed: {exc}. Stderr: {stderr}"
    except subprocess.TimeoutExpired as exc:
        return f"PLIP command timed out after {exc.timeout} seconds."
    except FileNotFoundError as exc:
        return f"Required executable not found: {exc.filename}"
    except json.JSONDecodeError as exc:
        return f"PLIP produced invalid JSON at {output_json}: {exc}"
