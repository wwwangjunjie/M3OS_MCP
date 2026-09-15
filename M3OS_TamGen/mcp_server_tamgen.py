#!/usr/bin/env python3
"""MCP server for protein-pocket-aware TamGen molecule generation."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import string
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from mcp.server import FastMCP

from tamgen_context import WORK_DIR, load_context, prepare_context
from tamgen_runtime import TamGenRuntime


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = WORK_DIR / "outputs"


def get_mcp_port(default_port: int = 8057) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))


app = FastMCP("tamgen", host="0.0.0.0", port=get_mcp_port())
_gpu_semaphore = asyncio.Semaphore(1)
_runtime_lock = threading.Lock()
_runtime: TamGenRuntime | None = None


def _get_runtime() -> TamGenRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = TamGenRuntime()
    return _runtime


def _suffix(length: int = 10) -> str:
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def _write_candidates(records: list[dict[str, Any]], output_csv_path: str) -> Path:
    if output_csv_path:
        path = Path(output_csv_path).expanduser().resolve()
        if path.suffix.lower() != ".csv":
            raise ValueError("output_csv_path must end in .csv")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = (OUTPUT_DIR / f"tamgen_candidates_{_suffix()}.csv").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "SMILES",
        "TamGen_generation_score",
        "TamGen_seed_similarity",
        "TamGen_generation_count",
        "TamGen_context_id",
        "TamGen_mode",
        "Input_SMILES",
        "TamGen_scaffold_smiles",
        "TamGen_scaffold_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def _generate_sync(
    context_path: str,
    num_candidates: int,
    beam_size: int,
    sample_beta: float,
    max_random_seeds: int,
    random_seed: int,
    min_seed_similarity: float,
    use_task_similarity_threshold: bool,
    require_scaffold_match: bool,
    output_csv_path: str,
) -> str:
    context = load_context(context_path)
    effective_similarity = float(min_seed_similarity)
    task_threshold = context.get("task_similarity_threshold")
    if use_task_similarity_threshold and task_threshold is not None:
        effective_similarity = max(effective_similarity, float(task_threshold))
    records, timing = _get_runtime().generate(
        context,
        num_candidates=int(num_candidates),
        beam_size=int(beam_size),
        sample_beta=float(sample_beta),
        max_random_seeds=int(max_random_seeds),
        random_seed=int(random_seed),
        min_seed_similarity=effective_similarity,
        require_scaffold_match=bool(require_scaffold_match),
    )
    csv_path = _write_candidates(records, str(output_csv_path or "").strip())
    status = "success" if len(records) >= int(num_candidates) else "partial"
    return json.dumps(
        {
            "status": status,
            "stage": "tamgen_target_aware_generation",
            "context_id": context["context_id"],
            "context_path": context["context_path"],
            "output_csv_path": str(csv_path),
            "requested_count": int(num_candidates),
            "output_count": len(records),
            "min_seed_similarity": effective_similarity,
            "candidates": records,
            **timing,
        },
        ensure_ascii=False,
    )


async def _run_gpu(label: str, function, *args) -> str:
    queued_at = time.perf_counter()
    async with _gpu_semaphore:
        started_at = time.perf_counter()
        work = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            LOGGER.warning("%s client disconnected; completing active GPU work", label)
            try:
                await asyncio.shield(work)
            except Exception:
                LOGGER.exception("Disconnected %s job later failed", label)
            raise
        finally:
            finished_at = time.perf_counter()
            LOGGER.info(
                "%s queue_wait=%.3fs run=%.3fs total=%.3fs",
                label,
                started_at - queued_at,
                finished_at - started_at,
                finished_at - queued_at,
            )


@app.tool()
async def tamgen_prepare_target_context(
    complex_pdb_path: str = "",
    reference_ligand_path: str = "",
    seed_smiles: str = "",
    task_yaml_path: str = "",
    pocket_residues: str = "",
    ligand_resname: str = "",
    ligand_chain: str = "",
    ligand_residue_number: int = 0,
    pocket_radius: float = 10.0,
    conditional: bool = True,
    seed_augmentations: int = 12,
    scaffold_smiles: str = "",
) -> str:
    """Prepare and cache a protein-pocket context for TamGen.

    Args:
        complex_pdb_path: Protein-ligand complex PDB path. This may be omitted
            when task_yaml_path declares a PDB in input_files.
        reference_ligand_path: SDF, MOL, or MOL2 seed molecule path. For
            conditional generation, provide this or seed_smiles.
        seed_smiles: Seed molecule SMILES used instead of a ligand file.
        task_yaml_path: Optional generic task YAML. Declared input files,
            pocket_residues, target_ligand, and a Tanimoto hard constraint are
            recognized when present.
        pocket_residues: Optional JSON list such as [["A",22],["A",23]], or
            compact text such as A:22,A:23. It overrides task YAML residues.
            When absent, residues within pocket_radius of the complex ligand
            are used.
        ligand_resname: Optional ligand residue name in the complex.
        ligand_chain: Optional ligand chain in the complex.
        ligand_residue_number: Optional ligand residue number in the complex.
        pocket_radius: Ligand-distance pocket radius in angstroms.
        conditional: Use the seed-conditioned TamGen pathway. Set false for
            pocket-conditioned de novo generation.
        seed_augmentations: Number of equivalent randomized seed SMILES used
            by conditional generation.
        scaffold_smiles: Optional explicit molecular scaffold used as the
            conditional input instead of the complete seed molecule.

    Returns:
        JSON string containing the reusable context_path, resolved pocket,
        seed molecule, and dataset metadata.
    """
    result = await asyncio.to_thread(
        prepare_context,
        complex_pdb_path=complex_pdb_path,
        reference_ligand_path=reference_ligand_path,
        seed_smiles=seed_smiles,
        task_yaml_path=task_yaml_path,
        pocket_residues=pocket_residues,
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
        ligand_residue_number=ligand_residue_number,
        pocket_radius=pocket_radius,
        conditional=conditional,
        seed_augmentations=seed_augmentations,
        scaffold_smiles=scaffold_smiles,
    )
    return json.dumps(result, ensure_ascii=False)


@app.tool()
async def tamgen_generate_target_aware_candidates(
    context_path: str,
    num_candidates: int = 200,
    beam_size: int = 20,
    sample_beta: float = 1.0,
    max_random_seeds: int = 50,
    random_seed: int = 1,
    min_seed_similarity: float = 0.0,
    use_task_similarity_threshold: bool = False,
    require_scaffold_match: bool = False,
    output_csv_path: str = "",
) -> str:
    """Generate target-aware molecules from a prepared TamGen context.

    Args:
        context_path: Path returned by tamgen_prepare_target_context.
        num_candidates: Number of unique valid molecules requested.
        beam_size: TamGen beam size per pocket/seed input.
        sample_beta: Latent sampling scale; larger values increase diversity.
        max_random_seeds: Maximum stochastic decoding rounds.
        random_seed: First stochastic decoding seed.
        min_seed_similarity: Minimum Morgan Tanimoto similarity to the seed.
        use_task_similarity_threshold: Also enforce a similarity threshold
            parsed from the optional task YAML used to prepare the context.
            It is disabled by default so generation and hard-constraint
            screening remain separate stages.
        require_scaffold_match: Require every returned molecule to contain the
            scaffold stored in a scaffold-conditioned context.
        output_csv_path: Optional output CSV destination.

    Returns:
        JSON string containing the candidate CSV path and generated molecules.
        The CSV has a standard SMILES column and can be passed directly to
        downstream ADMET or activity screening tools.
    """
    return await _run_gpu(
        "tamgen_generate_target_aware_candidates",
        _generate_sync,
        context_path,
        num_candidates,
        beam_size,
        sample_beta,
        max_random_seeds,
        random_seed,
        min_seed_similarity,
        use_task_similarity_threshold,
        require_scaffold_match,
        output_csv_path,
    )


def _prepare_and_generate_sync(
    preparation: dict[str, Any],
    generation: dict[str, Any],
) -> str:
    context = prepare_context(**preparation)
    return _generate_sync(context["context_path"], **generation)


@app.tool()
async def tamgen_optimize_lead_from_complex(
    complex_pdb_path: str = "",
    reference_ligand_path: str = "",
    seed_smiles: str = "",
    task_yaml_path: str = "",
    pocket_residues: str = "",
    ligand_resname: str = "",
    ligand_chain: str = "",
    ligand_residue_number: int = 0,
    pocket_radius: float = 10.0,
    seed_augmentations: int = 12,
    num_candidates: int = 200,
    beam_size: int = 20,
    sample_beta: float = 1.0,
    max_random_seeds: int = 50,
    random_seed: int = 1,
    min_seed_similarity: float = 0.0,
    use_task_similarity_threshold: bool = False,
    output_csv_path: str = "",
) -> str:
    """Run seed-conditioned, protein-pocket-aware lead generation end to end.

    The structural inputs use the same meanings as
    tamgen_prepare_target_context. The generation inputs use the same meanings
    as tamgen_generate_target_aware_candidates. Prepared protein contexts are
    cached and reused automatically.

    Returns:
        JSON string containing a downstream-compatible candidate CSV path and
        generated candidate records.
    """
    preparation = {
        "complex_pdb_path": complex_pdb_path,
        "reference_ligand_path": reference_ligand_path,
        "seed_smiles": seed_smiles,
        "task_yaml_path": task_yaml_path,
        "pocket_residues": pocket_residues,
        "ligand_resname": ligand_resname,
        "ligand_chain": ligand_chain,
        "ligand_residue_number": ligand_residue_number,
        "pocket_radius": pocket_radius,
        "conditional": True,
        "seed_augmentations": seed_augmentations,
        "scaffold_smiles": "",
    }
    generation = {
        "num_candidates": num_candidates,
        "beam_size": beam_size,
        "sample_beta": sample_beta,
        "max_random_seeds": max_random_seeds,
        "random_seed": random_seed,
        "min_seed_similarity": min_seed_similarity,
        "use_task_similarity_threshold": use_task_similarity_threshold,
        "require_scaffold_match": False,
        "output_csv_path": output_csv_path,
    }
    return await _run_gpu(
        "tamgen_optimize_lead_from_complex",
        _prepare_and_generate_sync,
        preparation,
        generation,
    )


@app.tool()
async def tamgen_generate_scaffold_conditioned_candidates(
    scaffold_smiles: str,
    complex_pdb_path: str = "",
    reference_ligand_path: str = "",
    seed_smiles: str = "",
    task_yaml_path: str = "",
    pocket_residues: str = "",
    ligand_resname: str = "",
    ligand_chain: str = "",
    ligand_residue_number: int = 0,
    pocket_radius: float = 10.0,
    scaffold_augmentations: int = 12,
    num_candidates: int = 200,
    beam_size: int = 20,
    sample_beta: float = 0.1,
    max_random_seeds: int = 50,
    random_seed: int = 1,
    min_seed_similarity: float = 0.0,
    require_scaffold_match: bool = True,
    output_csv_path: str = "",
) -> str:
    """Generate pocket-aware molecules conditioned on an explicit scaffold.

    The scaffold is passed to TamGen through its VAE conditional target path.
    A seed molecule remains the reference used for Tanimoto reporting. When
    require_scaffold_match is true, candidates that do not contain the
    scaffold as an RDKit substructure are excluded.

    Returns:
        JSON string containing conditional-path diagnostics, candidate records,
        and a downstream-compatible CSV path.
    """
    preparation = {
        "complex_pdb_path": complex_pdb_path,
        "reference_ligand_path": reference_ligand_path,
        "seed_smiles": seed_smiles,
        "task_yaml_path": task_yaml_path,
        "pocket_residues": pocket_residues,
        "ligand_resname": ligand_resname,
        "ligand_chain": ligand_chain,
        "ligand_residue_number": ligand_residue_number,
        "pocket_radius": pocket_radius,
        "conditional": True,
        "seed_augmentations": scaffold_augmentations,
        "scaffold_smiles": scaffold_smiles,
    }
    generation = {
        "num_candidates": num_candidates,
        "beam_size": beam_size,
        "sample_beta": sample_beta,
        "max_random_seeds": max_random_seeds,
        "random_seed": random_seed,
        "min_seed_similarity": min_seed_similarity,
        "use_task_similarity_threshold": False,
        "require_scaffold_match": require_scaffold_match,
        "output_csv_path": output_csv_path,
    }
    return await _run_gpu(
        "tamgen_generate_scaffold_conditioned_candidates",
        _prepare_and_generate_sync,
        preparation,
        generation,
    )


@app.tool()
async def health_check() -> dict[str, Any]:
    """Return MCP and resident model status."""
    runtime = _runtime
    return {
        "status": "running",
        "server": "tamgen",
        "protocol": "streamable-http",
        "model": runtime.metadata() if runtime is not None else {"loaded": False},
    }


def _signal_handler(_signum, _frame) -> None:
    LOGGER.info("Stopping TamGen MCP server")
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if os.environ.get("TAMGEN_PRELOAD", "1").strip().lower() not in {"0", "false", "no"}:
        _get_runtime()
    LOGGER.info("Starting TamGen MCP server on port %s", get_mcp_port())
    app.run(transport="streamable-http")
