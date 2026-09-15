#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REINVENT MCP Server
Standalone FastMCP server for molecular generation
Supports:
- Reinvent
- Mol2Mol
- LibInvent
- LinkInvent
"""

import os
import sys
import json
import asyncio
import csv
import signal
import logging
import secrets
import string
import ast
import tomllib
from pathlib import Path
from typing import Any, Literal

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# add current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server import FastMCP

# =========================
# Import REINVENT functions
# =========================

from m3os.tools.generation import (
    update_reinvent_config,
    run_reinvent,
    update_libinvent_config,
    save_smi_for_rgroup,
)
from configs.tool_globals import POOL_PATH, REINVENT_PATH

# =========================
# Logging
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mcp_port(default_port: int) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))

# =========================
# FastMCP App
# =========================

app = FastMCP(
    "reinvent4",
    host="0.0.0.0",
    port=get_mcp_port(8041),
)

# =============================================================================
# BASIC GENERATION
# =============================================================================
import traceback
@app.tool()
async def setup_generation_Mol2Mol_LinkInvent(
    model_type: Literal[
        # "Reinvent",
        "Mol2Mol",
        "LinkInvent"
    ] = "Reinvent",
    num_samples: int = 200,
    device: str = "cuda:0",
) -> list:
    """
    Setup REINVENT generation config (just for Mol2Mol and LinkInvent).

    Args:
        model_type:
            Mol2Mol (Similar molecule generation) / LinkInvent (Fragment linking)
        num_samples:
            number of generated molecules
        device:
            cuda device

    Returns:
        [config_file, rand_str]

        A list containing the configuration file path and a random string used for unique file naming in subsequent steps. 
        The random string ensures that each generation task has a unique identifier, preventing file naming conflicts when multiple tasks are run concurrently.
    """

    rand_str = ''.join(
        secrets.choice(string.ascii_lowercase)
        for _ in range(10)
    )

    try:
        result = update_reinvent_config(
            model_type=model_type,
            device=device,
            num_samples=num_samples,
            rand_str=rand_str,
        )

        return [str(result), rand_str]

    except Exception as e:
        logger.exception("setup_generation failed")

        return {
            "status": "error",
            "model_type": model_type,
            "rand_str": rand_str,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

# =============================================================================
# input smi
# =============================================================================

@app.tool()
async def prepare_smi_input(
    smiles: tuple,
    file_path: str
) -> str:
    """
    Save SMILES to .smi.

    Args:
        smiles:
            tuple of smiles

        file_path:
            output smi path

    Returns:
        save result
    """

    parent_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w") as f:

        for smi in smiles:
            f.write(f"{smi}\n")

    return f"SMILES saved to {file_path}"


# =============================================================================
# RUN GENERATION
# =============================================================================

def _generation_output_path(config_file: str) -> Path:
    config_path = Path(config_file).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output_value = str(config.get("parameters", {}).get("output_file", "")).strip()
    if not output_value:
        raise ValueError(f"REINVENT config has no parameters.output_file: {config_path}")
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        output_path = Path(REINVENT_PATH) / output_path
    return output_path.resolve()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _generation_id(config_file: str, output_path: Path) -> str:
    if output_path.stem.startswith("sampling_") and "_" in output_path.stem:
        return output_path.stem.rsplit("_", 1)[-1]
    config_parent = Path(config_file).expanduser().resolve().parent.name
    if config_parent:
        return config_parent
    return ""


MOL2MOL_MODELS = (
    "high_similarity",
    "medium_similarity",
    "similarity",
    "mmp",
    "scaffold",
    "scaffold_generic",
)
MOL2MOL_OUTPUT_FIELDS = (
    "SMILES",
    "Input_SMILES",
    "SMDD_Tanimoto",
    "Tanimoto",
    "NLL",
    "Mol2Mol_model",
    "Mol2Mol_sampling_strategy",
    "Mol2Mol_round",
)


def _canonical_molecule(smiles: str) -> tuple[str, Any]:
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid seed SMILES: {smiles}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return canonical, molecule


def _collect_similarity_candidates(
    generated_csv: Path,
    *,
    seed_fingerprint: Any,
    seed_canonical: str,
    similarity_threshold: float,
    seen: set[str],
    model_variant: str,
    sampling_strategy: str,
    round_index: int,
    exclude_seed: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    )
    accepted: list[dict[str, Any]] = []
    counts = {
        "raw": 0,
        "valid": 0,
        "threshold_passes": 0,
        "duplicates": 0,
        "seed_matches": 0,
    }
    with generated_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["raw"] += 1
            molecule = Chem.MolFromSmiles(str(row.get("SMILES", "")).strip())
            if molecule is None:
                continue
            counts["valid"] += 1
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    seed_fingerprint,
                    fingerprint_generator.GetFingerprint(molecule),
                )
            )
            if similarity < similarity_threshold:
                continue
            counts["threshold_passes"] += 1
            if exclude_seed and canonical == seed_canonical:
                counts["seed_matches"] += 1
                continue
            if canonical in seen:
                counts["duplicates"] += 1
                continue
            seen.add(canonical)
            accepted.append(
                {
                    "SMILES": canonical,
                    "Input_SMILES": seed_canonical,
                    "SMDD_Tanimoto": similarity,
                    "Tanimoto": row.get("Tanimoto", ""),
                    "NLL": row.get("NLL", ""),
                    "Mol2Mol_model": model_variant,
                    "Mol2Mol_sampling_strategy": sampling_strategy,
                    "Mol2Mol_round": round_index,
                }
            )
    return accepted, counts


def _write_mol2mol_candidates(
    output_path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOL2MOL_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)


def _generate_similarity_constrained_mol2mol_sync(
    seed_smiles: str,
    similarity_threshold: float,
    num_candidates: int,
    samples_per_round: int,
    max_rounds: int,
    device: str,
    model_variant: str,
    temperature: float,
    random_seed: int,
    exclude_seed: bool,
    output_csv_path: str,
) -> str:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if not 1 <= num_candidates <= 10000:
        raise ValueError("num_candidates must be between 1 and 10000")
    if not 1 <= samples_per_round <= 10000:
        raise ValueError("samples_per_round must be between 1 and 10000")
    if not 1 <= max_rounds <= 100:
        raise ValueError("max_rounds must be between 1 and 100")
    if model_variant not in MOL2MOL_MODELS:
        raise ValueError(
            f"model_variant must be one of: {', '.join(MOL2MOL_MODELS)}"
        )
    if not 0.01 <= temperature <= 5.0:
        raise ValueError("temperature must be between 0.01 and 5")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")

    seed_canonical, seed_molecule = _canonical_molecule(seed_smiles)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    )
    seed_fingerprint = fingerprint_generator.GetFingerprint(seed_molecule)
    generation_id = "".join(
        secrets.choice(string.ascii_lowercase) for _ in range(10)
    )
    output_path = (
        Path(output_csv_path).expanduser().resolve()
        if str(output_csv_path or "").strip()
        else Path(POOL_PATH).resolve()
        / f"sampling_Mol2Mol_similarity_{generation_id}.csv"
    )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_csv_paths: list[str] = []
    totals = {
        "raw": 0,
        "valid": 0,
        "threshold_passes": 0,
        "duplicates": 0,
        "seed_matches": 0,
    }
    rounds_used = 0
    for round_index in range(1, max_rounds + 1):
        rounds_used = round_index
        round_id = f"{generation_id}r{round_index:02d}"
        config_file = update_reinvent_config(
            model_type="Mol2Mol",
            device=device,
            num_samples=samples_per_round,
            rand_str=round_id,
            mol2mol_model=model_variant,
            sample_strategy="multinomial",
            temperature=temperature,
            random_seed=random_seed + round_index - 1,
        )
        config_path = Path(config_file).resolve()
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        input_path = Path(config["parameters"]["smiles_file"]).resolve()
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(f"{seed_canonical}\n", encoding="utf-8")

        result = run_reinvent(str(config_path))
        if result != "REINVENT execution completed successfully.":
            raise RuntimeError(result)
        generated_csv = _generation_output_path(str(config_path))
        if not generated_csv.is_file():
            raise RuntimeError(
                f"Mol2Mol completed without creating its CSV: {generated_csv}"
            )
        source_csv_paths.append(str(generated_csv))
        new_candidates, counts = _collect_similarity_candidates(
            generated_csv,
            seed_fingerprint=seed_fingerprint,
            seed_canonical=seed_canonical,
            similarity_threshold=similarity_threshold,
            seen=seen,
            model_variant=model_variant,
            sampling_strategy="multinomial",
            round_index=round_index,
            exclude_seed=exclude_seed,
        )
        candidates.extend(new_candidates)
        for key, value in counts.items():
            totals[key] += value
        if len(candidates) >= num_candidates:
            break

    selected = candidates[:num_candidates]
    _write_mol2mol_candidates(output_path, selected)
    payload = {
        "status": "success" if len(selected) >= num_candidates else "partial",
        "stage": "mol2mol_similarity_constrained_generation",
        "rand_str": generation_id,
        "seed_smiles": seed_canonical,
        "similarity_threshold": similarity_threshold,
        "similarity_definition": {
            "fingerprint": "Morgan bit vector",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
            "metric": "Tanimoto",
        },
        "model_variant": model_variant,
        "sampling_strategy": "multinomial",
        "temperature": temperature,
        "requested_count": num_candidates,
        "output_count": len(selected),
        "rounds_used": rounds_used,
        "raw_sample_count": totals["raw"],
        "valid_sample_count": totals["valid"],
        "threshold_pass_count": totals["threshold_passes"],
        "duplicate_count": totals["duplicates"],
        "excluded_seed_count": totals["seed_matches"],
        "output_csv_path": str(output_path),
        "source_csv_paths": source_csv_paths,
        "candidates": selected,
    }
    return json.dumps(payload, ensure_ascii=False)

@app.tool()
async def run_generation(config_file: str) -> str:
    """
    Run REINVENT generation.

    Args:
        config_file: Path to the TOML configuration file
        
    Returns:
        JSON string containing status, rand_str, generated_csv_path, and
        molecule_count.
    """
    output_path = _generation_output_path(config_file)
    result = await asyncio.to_thread(run_reinvent, config_file)
    status = (
        "success"
        if result == "REINVENT execution completed successfully."
        else "error"
    )
    payload = {
        "status": status,
        "message": str(result),
        "rand_str": _generation_id(config_file, output_path),
        "generated_csv_path": str(output_path),
        "molecule_count": (
            _csv_row_count(output_path)
            if status == "success" and output_path.is_file()
            else 0
        ),
    }
    if status == "success" and not output_path.is_file():
        payload["status"] = "error"
        payload["message"] = (
            "REINVENT completed without creating the configured CSV: "
            f"{output_path}"
        )
    return json.dumps(payload, ensure_ascii=False)


@app.tool()
async def generate_similarity_constrained_mol2mol(
    seed_smiles: str,
    similarity_threshold: float = 0.7,
    num_candidates: int = 10000,
    samples_per_round: int = 1000,
    max_rounds: int = 10,
    device: str = "cuda:0",
    model_variant: Literal[
        "high_similarity",
        "medium_similarity",
        "similarity",
        "mmp",
        "scaffold",
        "scaffold_generic",
    ] = "high_similarity",
    temperature: float = 1.0,
    random_seed: int = 42,
    exclude_seed: bool = True,
    output_csv_path: str = "",
) -> str:
    """Generate random Mol2Mol candidates satisfying a similarity constraint.

    The tool uses multinomial Mol2Mol sampling and filters every generated
    molecule with the SMDD-Bench fingerprint definition: Morgan radius 2,
    2048-bit binary fingerprints, no chirality, and Tanimoto similarity. It
    repeats sampling with successive random seeds until enough unique
    candidates pass or max_rounds is reached.

    Args:
        seed_smiles: Initial molecule used for local Mol2Mol generation.
        similarity_threshold: Minimum SMDD_Tanimoto value, inclusive.
        num_candidates: Number of unique passing molecules requested.
        samples_per_round: Number of random Mol2Mol samples per round.
        max_rounds: Maximum number of sampling rounds.
        device: REINVENT torch device.
        model_variant: Mol2Mol prior. High similarity is the default for local
            lead optimization with a strict similarity constraint.
        temperature: Multinomial decoding temperature.
        random_seed: Random seed for the first round; later rounds increment it.
        exclude_seed: Exclude an unchanged copy of the initial molecule.
        output_csv_path: Optional final CSV path. The default is under tmp/pool.

    Returns:
        JSON containing the chainable output CSV, sampling statistics, exact
        similarity definition, and accepted candidate records.
    """
    return await asyncio.to_thread(
        _generate_similarity_constrained_mol2mol_sync,
        seed_smiles,
        similarity_threshold,
        num_candidates,
        samples_per_round,
        max_rounds,
        device,
        model_variant,
        temperature,
        random_seed,
        exclude_seed,
        output_csv_path,
    )


# =============================================================================
# LIBINVENT
# =============================================================================

@app.tool()
async def setup_libinvent(
    num_samples: int = 200,
    device: str = "cuda:0",
) -> list:
    """
    Setup LibInvent generation.

    Args:
        num_samples:
            number of generated molecules

        device:
            cuda device

    Returns:
        [config_file, rand_str]

        A list containing the configuration file path and a random string used for unique file naming in subsequent steps. 
        The random string ensures that each generation task has a unique identifier, preventing file naming conflicts when multiple tasks are run concurrently.
    """

    rand_str = ''.join(
        secrets.choice(string.ascii_lowercase)
        for _ in range(10)
    )

    config_file = update_libinvent_config(
        model_type="LibInvent",
        device=device,
        num_samples=num_samples,
        rand_str=rand_str,
    )

    return [str(config_file), rand_str]


@app.tool()
async def prepare_scaffold(
    scaffold_smiles_list_str: str,
    rand_str: str,
) -> str:
    """
    Receive a string containing a list of multiple scaffold SMILES and prepare a file with attachment points for the R-group generation.
    
    Args:
        scaffold_smiles_list_str: A string containing a list of multiple scaffold SMILES.
        rand_str: A random string to ensure unique file naming for each generation task.
        
    Returns:
        A descriptive string containing the status of the processing result

    Example input of `scaffold_smiles_list_str`: '["[*]c1ccccc1[*]", "[*]c1ncncc1"]'
    You **must** use a fresh setup using [*] wildcard notation. You **must** use a fresh setup using [*] wildcard notation. You **must** use a fresh setup using [*] wildcard notation.
    Note: Please do not fabricate `rand_str`. Use the `rand_str` returned by the `setup_libinvent` function to ensure consistency and correctness in file naming.
    """

    try:

        scaffold_smiles_list = ast.literal_eval(
            scaffold_smiles_list_str
        )

        result = save_smi_for_rgroup(
            scaffold_smiles_list,
            rand_str=rand_str,
        )

        return str(result)

    except Exception as e:

        return f"Error: {str(e)}"


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.tool()
async def health_check() -> dict:
    """
    Health check API.
    """

    return {
        "status": "running",
        "server": "reinvent4",
        "protocol": "streamable-http",
    }


# =============================================================================
# RESOURCES
# =============================================================================

@app.resource("reinvent://models")
async def get_models():

    models = {
        "Reinvent": {
            "description": "de novo generation",
        },
        "Mol2Mol": {
            "description": "molecule optimization",
        },
        "LibInvent": {
            "description": "scaffold decoration",
        },
        "LinkInvent": {
            "description": "fragment linking",
        },
    }

    return json.dumps(models, indent=2)


# =============================================================================
# SHUTDOWN
# =============================================================================

def signal_handler(signum, frame):

    logger.info("Stopping REINVENT MCP server...")
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:

        logger.info("Starting REINVENT MCP Server...")
        logger.info("HTTP endpoint: http://0.0.0.0:8041")
        logger.info("Transport: streamable-http")

        app.run(transport="streamable-http")

    except KeyboardInterrupt:

        logger.info("Server stopped.")

    except Exception as e:

        logger.error(f"Fatal error: {e}")
        sys.exit(1)
