from __future__ import annotations

import json
import os
import re
import secrets
import string
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from configs.tool_globals import (
    BOLTZ_CACHE_DIR,
    BOLTZ_CLI,
    BOLTZ_OUTPUT_DIR,
    BOLTZ_PYTHON,
    PROJECT_ROOT,
)


def _parse_complex_input(data: str | dict[str, Any]) -> dict[str, Any] | str:
    if isinstance(data, dict):
        return data

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return (
            "Error: Invalid input format. Expected JSON with 'sequence' "
            "and 'smiles'."
        )

    if not isinstance(parsed, dict):
        return "Error: Invalid input format. Expected a JSON dictionary."

    return parsed


def _safe_output_name(protein_seq: str, ligand_smiles: str) -> str:
    safe_protein = re.sub(r"[^a-zA-Z0-9]", "", protein_seq[:10]) or "protein"
    safe_smiles = re.sub(r"[^a-zA-Z0-9]", "", ligand_smiles[:10]) or "ligand"
    rand_str = "".join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    return f"{safe_protein}_{safe_smiles}_{rand_str}"


def generate_complex_structure(
    data: str | dict[str, Any],
    subprocess_timeout_seconds: float | None = None,
):
    """
    Create a protein-ligand complex structure with Boltz.

    Args:
        data: JSON/dict with "sequence" and "smiles" keys.

    Returns:
        A dictionary containing the generated PDB path and Boltz affinity
        metadata. PLIP is intentionally not run here; interaction analysis is
        exposed as a separate MCP tool.
    """
    parsed = _parse_complex_input(data)
    if isinstance(parsed, str):
        return parsed

    protein_seq = parsed.get("sequence")
    ligand_smiles = parsed.get("smiles")
    if not protein_seq or not ligand_smiles:
        return "Error: Input must include non-empty 'sequence' and 'smiles'."

    output_dir = Path(BOLTZ_OUTPUT_DIR)
    cache_dir = Path(BOLTZ_CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_name = _safe_output_name(protein_seq, ligand_smiles)
    yaml_filename = output_dir / f"{output_name}.yaml"
    counter = 1
    while yaml_filename.exists():
        yaml_filename = output_dir / f"{output_name}_{counter}.yaml"
        counter += 1
    output_name = yaml_filename.stem

    config_data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": protein_seq}},
            {"ligand": {"id": "B", "smiles": ligand_smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    with yaml_filename.open("w") as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

    boltz_command = [
        BOLTZ_PYTHON,
        BOLTZ_CLI,
        "predict",
        str(yaml_filename),
        "--out_dir",
        str(output_dir),
        "--cache",
        str(cache_dir),
        "--use_msa_server",
        "--accelerator",
        os.environ.get("BOLTZ_ACCELERATOR", "gpu"),
        "--num_workers",
        os.environ.get("BOLTZ_NUM_WORKERS", "20"),
        "--output_format",
        "pdb",
    ]
    disable_kernels = os.environ.get("BOLTZ_DISABLE_KERNELS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    use_kernels = os.environ.get("BOLTZ_USE_KERNELS", "").lower()
    if disable_kernels or use_kernels in {"0", "false", "no"}:
        boltz_command.append("--no_kernels")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            boltz_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=subprocess_timeout_seconds,
        )
        print("Boltz structure prediction completed.")
        print(f"Output: {result.stdout}")

        boltz_prediction_dir = (
            output_dir / f"boltz_results_{output_name}" / "predictions" / output_name
        )
        boltz_affinity = boltz_prediction_dir / f"affinity_{output_name}.json"
        boltz_model = boltz_prediction_dir / f"{output_name}_model_0.pdb"

        if not boltz_model.is_file():
            return (
                "Boltz prediction finished but the expected structure file "
                f"was not found: {boltz_model}"
            )
        if not boltz_affinity.is_file():
            return (
                "Boltz prediction finished but the expected affinity file "
                f"was not found: {boltz_affinity}"
            )

        with boltz_affinity.open("r") as f1:
            raw_affinity_data = json.load(f1)
        pred_pIC50 = (6 - raw_affinity_data["affinity_pred_value"]) * 1.364

        return {
            "status": "success",
            "structure_path": str(boltz_model.resolve()),
            "affinity_json_path": str(boltz_affinity.resolve()),
            "prediction_dir": str(boltz_prediction_dir.resolve()),
            "input_yaml_path": str(yaml_filename.resolve()),
            "pred_pIC50": pred_pIC50,
        }

    except subprocess.CalledProcessError as e:
        stderr = e.stderr if e.stderr else "No stderr"
        return (
            f"Config file created: {yaml_filename}. Boltz command failed: {e}. "
            f"Stderr: {stderr}"
        )
    except subprocess.TimeoutExpired as e:
        return (
            f"Config file created: {yaml_filename}. Boltz command timed out "
            f"after {e.timeout} seconds."
        )
    except FileNotFoundError as e:
        return (
            f"Config file created: {yaml_filename}. Required executable not "
            f"found: {e.filename}"
        )
