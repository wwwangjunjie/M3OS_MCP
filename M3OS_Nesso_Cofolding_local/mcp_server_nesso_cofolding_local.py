#!/usr/bin/env python3
"""MCP server for the host-local, resident Nesso runtime."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import secrets
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from mcp.server import FastMCP
import pandas as pd

from nesso_activity_tools import (
    ACTIVITY_COLUMNS,
    TMP_DIR,
    canonicalize_smiles,
    format_missing_generated_csv_message,
    prediction_records,
    prepare_generated_csv,
    prepare_smiles_list_csv,
    rank_by_binding_probability,
    read_activity_results,
    resolve_generated_csv_path,
)
from nesso_prediction_cache import NessoPredictionCache

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
SKILL_DIR = PROJECT_DIR / "nesso_cofolding_skill"
RUNS_DIR = PROJECT_DIR / "nesso_runs"
os.environ.setdefault("NESSO_LOCAL_RUNS_ROOT", str(RUNS_DIR))


def get_mcp_port(default_port: int = 8051) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))


def _resolve_protein_input(protein_sequence: str, protein_fasta_path: str) -> str:
    sequence_value = str(protein_sequence or "").strip()
    fasta_value = str(protein_fasta_path or "").strip()
    if bool(sequence_value) == bool(fasta_value):
        raise ValueError(
            "Provide exactly one of protein_sequence and protein_fasta_path"
        )
    if fasta_value:
        fasta = Path(fasta_value).expanduser().resolve()
        if not fasta.is_file():
            raise FileNotFoundError(f"Protein FASTA not found: {fasta}")
        sequence_value = fasta.read_text(encoding="utf-8")
    residues = []
    for line in sequence_value.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(">"):
            residues.append("".join(stripped.split()))
    sequence = "".join(residues).upper()
    if not sequence:
        raise ValueError("Protein input contains no amino-acid sequence")
    return sequence


app = FastMCP("nesso_cofolding_local", host="0.0.0.0", port=get_mcp_port())
_run_semaphore = asyncio.Semaphore(1)
_prediction_cache: NessoPredictionCache | None = None
_prediction_cache_lock = threading.Lock()


def _get_prediction_cache() -> NessoPredictionCache:
    global _prediction_cache
    if _prediction_cache is None:
        with _prediction_cache_lock:
            if _prediction_cache is None:
                _prediction_cache = NessoPredictionCache()
    return _prediction_cache


def run_nesso_cofolding(data: str, gpu_id: str | None = None) -> str:
    """Validate one request, submit it to the local worker, and return JSON."""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {
                "success": False,
                "error_code": "JSONDecodeError",
                "error_message": f"data is not valid JSON: {exc}",
            },
            ensure_ascii=False,
            indent=2,
        )
    if not isinstance(payload, dict):
        return json.dumps(
            {
                "success": False,
                "error_code": "ValueError",
                "error_message": "data must be a JSON object",
            },
            ensure_ascii=False,
            indent=2,
        )
    if "ligand_csv_path" not in payload:
        return json.dumps(
            {
                "success": False,
                "error_code": "ValueError",
                "error_message": "data must contain 'ligand_csv_path'",
            },
            ensure_ascii=False,
            indent=2,
        )
    has_fasta_path = payload.get("protein_fasta_path") is not None
    has_sequence = payload.get("protein_sequence") is not None
    if has_fasta_path == has_sequence:
        return json.dumps(
            {
                "success": False,
                "error_code": "ValueError",
                "error_message": (
                    "data must contain exactly one of 'protein_fasta_path' "
                    "and 'protein_sequence'"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"nesso_{uuid.uuid4().hex[:8]}_", dir=str(RUNS_DIR)
        )
    )
    input_json_path = run_dir / "input.json"
    input_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(SKILL_DIR / "main.py"), str(input_json_path)]
    LOGGER.info("Running local Nesso skill: %s", " ".join(command))
    try:
        environment = os.environ.copy()
        if gpu_id is not None:
            environment["NESSO_LOCAL_GPU_ID"] = str(gpu_id)
        process = subprocess.run(
            command,
            cwd=SKILL_DIR,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        output_json_path = run_dir / "output.json"
        if output_json_path.is_file():
            result = json.loads(output_json_path.read_text(encoding="utf-8"))
        else:
            result = {
                "success": False,
                "error_code": "SkillNoOutput",
                "error_message": (
                    f"Local skill exited with {process.returncode}; "
                    f"log follows:\n{process.stdout[-4000:]}"
                ),
            }
        result.setdefault("mcp_run_dir", str(run_dir))
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001 - return structured tool failure
        LOGGER.exception("Local Nesso skill invocation failed")
        return json.dumps(
            {
                "success": False,
                "error_code": exc.__class__.__name__,
                "error_message": str(exc),
                "mcp_run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        )


def _random_suffix(length: int = 10) -> str:
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def _resolve_candidate_csv_path(candidate_csv_path: str, rand_str: str) -> Path:
    explicit_path = str(candidate_csv_path or "").strip()
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.suffix.lower() != ".csv":
            raise ValueError(f"candidate_csv_path must point to a CSV file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Candidate CSV not found: {path}")
        return path

    generated_csv = resolve_generated_csv_path(str(rand_str or "").strip())
    if generated_csv is None:
        raise FileNotFoundError(format_missing_generated_csv_message(rand_str))
    return generated_csv.resolve()


def _validate_top_n(top_n: int) -> int:
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    return top_n


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _numeric_gpu_ids(value: str) -> list[str]:
    ids = []
    for item in str(value or "").split(","):
        gpu_id = item.strip()
        if not gpu_id:
            continue
        if not gpu_id.isdigit():
            raise ValueError(f"GPU IDs must be numeric physical IDs, got {gpu_id!r}")
        if gpu_id not in ids:
            ids.append(gpu_id)
    return ids


def _available_gpu_ids() -> list[str]:
    configured = os.environ.get("NESSO_LOCAL_GPU_IDS", "").strip()
    if configured:
        gpu_ids = _numeric_gpu_ids(configured)
    else:
        single = os.environ.get("NESSO_LOCAL_GPU_ID", "").strip()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if single:
            gpu_ids = _numeric_gpu_ids(single)
        elif visible:
            gpu_ids = _numeric_gpu_ids(visible)
        else:
            try:
                process = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                process = None
            gpu_ids = (
                _numeric_gpu_ids(process.stdout.replace("\n", ","))
                if process is not None and process.returncode == 0
                else []
            )
    if not gpu_ids:
        return []
    maximum = _positive_int_env("NESSO_MCP_MAX_GPUS", len(gpu_ids))
    return gpu_ids[:maximum]


def _predict_csv_with_nesso(
    *,
    ligand_csv: Path,
    protein_sequence: str,
    protein_id: str,
    job_name: str,
    gpu_id: str | None = None,
) -> Path:
    payload = {
        "job_name": job_name,
        "protein_sequence": protein_sequence,
        "protein_id": protein_id,
        "ligand_csv_path": str(ligand_csv.resolve()),
        "smiles_column": "SMILES",
    }
    result = json.loads(run_nesso_cofolding(json.dumps(payload), gpu_id=gpu_id))
    result_csv_value = str(result.get("result_csv", "")).strip()
    if not result_csv_value or not Path(result_csv_value).is_file():
        raise RuntimeError(
            "Local Nesso prediction produced no result CSV: "
            f"{result.get('error_code')}: {result.get('error_message')}"
        )
    return Path(result_csv_value)


def _predict_frame_in_batches(
    frame: pd.DataFrame,
    *,
    protein_sequence: str,
    protein_id: str,
    job_name: str,
    suffix: str,
) -> pd.DataFrame:
    batch_size = _positive_int_env("NESSO_MCP_BATCH_SIZE", 32)
    max_attempts = _positive_int_env("NESSO_MCP_BATCH_MAX_ATTEMPTS", 3)
    prepared = frame.copy().reset_index(drop=True)
    canonical_keys = [canonicalize_smiles(value) for value in prepared["SMILES"]]
    invalid = [
        str(prepared.iloc[index]["SMILES"])
        for index, key in enumerate(canonical_keys)
        if key is None
    ]
    if invalid:
        raise ValueError(f"Nesso received invalid SMILES: {invalid}")
    resolved_keys = [str(key) for key in canonical_keys]

    cache = _get_prediction_cache()
    try:
        cached = cache.get_many(protein_sequence, resolved_keys)
    except Exception:
        LOGGER.exception("Nesso prediction cache lookup failed; computing requested molecules")
        cached = {}

    missing_keys = [key for key in dict.fromkeys(resolved_keys) if key not in cached]
    first_index = {key: resolved_keys.index(key) for key in missing_keys}
    missing_frame = prepared.iloc[[first_index[key] for key in missing_keys]].copy()
    total_batches = (len(missing_frame) + batch_size - 1) // batch_size
    batches = [
        (
            batch_index,
            missing_frame.iloc[offset : offset + batch_size].copy(),
        )
        for batch_index, offset in enumerate(
            range(0, len(missing_frame), batch_size),
            start=1,
        )
    ]
    gpu_ids: list[str | None] = _available_gpu_ids() or [None]
    worker_count = min(len(gpu_ids), len(batches)) if batches else 0

    def predict_batch(
        batch_index: int,
        batch: pd.DataFrame,
        gpu_id: str | None,
    ) -> list[dict[str, object]]:
        batch_csv = TMP_DIR / f"{job_name}_{suffix}_batch_{batch_index:04d}.csv"
        batch.to_csv(batch_csv, index=False)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result_csv = _predict_csv_with_nesso(
                    ligand_csv=batch_csv,
                    protein_sequence=protein_sequence,
                    protein_id=protein_id,
                    job_name=f"{job_name}_{suffix}_{batch_index:04d}",
                    gpu_id=gpu_id,
                )
                scored_batch = read_activity_results(result_csv)
                if len(scored_batch) != len(batch):
                    raise RuntimeError(
                        "Nesso batch output row count does not match its input: "
                        f"batch={batch_index}, input={len(batch)}, output={len(scored_batch)}"
                    )
                batch_records: list[dict[str, object]] = []
                for _, row in scored_batch.iterrows():
                    key = canonicalize_smiles(row["SMILES"])
                    if key is None:
                        raise RuntimeError("Nesso returned an invalid SMILES value")
                    record = {
                        "SMILES": key,
                        "Nesso_affinity_pred_value": float(
                            row["Nesso_affinity_pred_value"]
                        ),
                        "Nesso_affinity_probability_binary": float(
                            row["Nesso_affinity_probability_binary"]
                        ),
                    }
                    batch_records.append(record)
                expected_keys = {
                    str(canonicalize_smiles(value)) for value in batch["SMILES"]
                }
                returned_keys = {str(record["SMILES"]) for record in batch_records}
                if not expected_keys.issubset(returned_keys):
                    raise RuntimeError(
                        f"Nesso batch {batch_index} did not return every requested molecule"
                    )
                return batch_records
            except Exception as exc:  # noqa: BLE001 - worker exits require retry
                last_error = exc
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Nesso batch {batch_index} failed after {max_attempts} attempts"
                    ) from exc
                LOGGER.warning(
                    "Nesso batch %s/%s attempt %s failed: %s; retrying with worker recovery",
                    batch_index,
                    total_batches,
                    attempt,
                    exc,
                )
                time.sleep(0.5)
        raise RuntimeError(f"Nesso batch {batch_index} produced no result") from last_error

    def predict_gpu_queue(
        gpu_id: str | None,
        assigned_batches: list[tuple[int, pd.DataFrame]],
    ) -> list[tuple[int, list[dict[str, object]]]]:
        return [
            (batch_index, predict_batch(batch_index, batch, gpu_id))
            for batch_index, batch in assigned_batches
        ]

    completed: list[tuple[int, list[dict[str, object]]]] = []
    if batches:
        assignments: list[list[tuple[int, pd.DataFrame]]] = [
            [] for _ in range(worker_count)
        ]
        for index, batch_item in enumerate(batches):
            assignments[index % worker_count].append(batch_item)
        LOGGER.info(
            "%s dispatching %d Nesso batches across GPUs=%s",
            job_name,
            len(batches),
            [gpu_id for gpu_id in gpu_ids[:worker_count] if gpu_id is not None],
        )
        if worker_count == 1:
            completed = predict_gpu_queue(gpu_ids[0], assignments[0])
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="nesso-gpu",
            ) as executor:
                futures = [
                    executor.submit(predict_gpu_queue, gpu_ids[index], assignment)
                    for index, assignment in enumerate(assignments)
                ]
                for future in as_completed(futures):
                    completed.extend(future.result())

    new_records: list[dict[str, object]] = []
    for _batch_index, batch_records in sorted(completed):
        for record in batch_records:
            cached[str(record["SMILES"])] = record
            new_records.append(record)
    if new_records:
        try:
            cache.upsert_many(
                protein_sequence,
                new_records,
                source_tool=job_name,
            )
        except Exception:
            LOGGER.exception("Nesso prediction cache write failed")

    scored = prepared.copy()
    for column in ACTIVITY_COLUMNS:
        scored[column] = [cached[key][column] for key in resolved_keys]
    LOGGER.info(
        "%s prediction cache hits=%d misses=%d total=%d",
        job_name,
        len(resolved_keys) - len(missing_keys),
        len(missing_keys),
        len(resolved_keys),
    )
    return scored


def _filter_by_nesso_sync(
    protein_sequence: str,
    rand_str: str,
    protein_id: str,
    candidate_csv_path: str,
    top_n: int,
) -> str:
    selected_top_n = _validate_top_n(top_n)
    source_csv = _resolve_candidate_csv_path(candidate_csv_path, rand_str)
    suffix = _random_suffix()
    prediction_input = TMP_DIR / f"nesso_filter_input_{suffix}.csv"
    prepared = prepare_generated_csv(source_csv, prediction_input)
    scored = _predict_frame_in_batches(
        prepared,
        protein_sequence=protein_sequence,
        protein_id=protein_id,
        job_name=f"nesso_filter_{suffix}",
        suffix=suffix,
    )
    output_csv = TMP_DIR / f"nesso_ranked_{suffix}.csv"
    top_candidates = rank_by_binding_probability(
        scored,
        output_csv=output_csv,
        top_n=selected_top_n,
    )
    payload = {
        "status": "success",
        "stage": "nesso",
        "rand_str": str(rand_str or "").strip(),
        "input_csv_path": str(source_csv),
        "output_csv_path": str(output_csv.resolve()),
        "input_count": int(len(prepared)),
        "output_count": int(len(top_candidates)),
        "candidates": top_candidates,
    }
    return json.dumps(payload, ensure_ascii=False)


def _predict_by_nesso_sync(
    smiles_list: str,
    protein_sequence: str,
    protein_id: str,
) -> str:
    suffix = _random_suffix()
    prediction_input = TMP_DIR / f"nesso_predict_input_{suffix}.csv"
    parsed_smiles = prepare_smiles_list_csv(smiles_list, prediction_input)
    prepared = pd.DataFrame({"SMILES": parsed_smiles})
    scored = _predict_frame_in_batches(
        prepared,
        protein_sequence=protein_sequence,
        protein_id=protein_id,
        job_name=f"nesso_predict_{suffix}",
        suffix=suffix,
    )
    records = prediction_records(scored)
    if not records:
        raise ValueError("Nesso produced no valid activity predictions")
    return str(records)


async def _run_blocking_nesso_call(label: str, function, *args) -> str:
    queued_at = time.perf_counter()
    async with _run_semaphore:
        started_at = time.perf_counter()
        work = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            LOGGER.warning(
                "%s client disconnected; waiting for its Nesso job to finish",
                label,
            )
            try:
                await asyncio.shield(work)
            except Exception:
                LOGGER.exception("Disconnected %s job later failed", label)
            raise
        except Exception:
            LOGGER.exception("%s failed", label)
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
async def nesso_filter_by_cofolding(
    protein_sequence: str = "",
    rand_str: str = "",
    protein_id: str = "target",
    candidate_csv_path: str = "",
    top_n: int = 5,
    protein_fasta_path: str = "",
) -> str:
    """Filter candidate molecules by Nesso binder probability.

    It scores target-protein and ligand pairs, persists raw predictions for
    later evaluation, and ranks by Nesso_affinity_probability_binary.

    Args:
        protein_sequence: Raw amino-acid sequence of the target protein.
        rand_str: Reinvent generation identifier. The server searches the same
            Reinvent pool directories and filename prefixes as ADMET-AI.
        protein_id: Optional identifier used for the generated FASTA record.
        candidate_csv_path: Exact path to a candidate CSV. When provided, this
            takes precedence over rand_str and may point to a previous
            filtering stage.
        top_n: Number of ranked candidates to retain.
        protein_fasta_path: Path to a target FASTA file. Supply exactly one
            protein input.

    Returns:
        JSON string containing candidates ranked in descending
        Nesso_affinity_probability_binary order and the output CSV path.
    """
    return await _run_blocking_nesso_call(
        "nesso_filter_by_cofolding",
        _filter_by_nesso_sync,
        _resolve_protein_input(protein_sequence, protein_fasta_path),
        rand_str,
        protein_id,
        candidate_csv_path,
        top_n,
    )


@app.tool()
async def nesso_predict_by_cofolding(
    smiles_list: str,
    protein_sequence: str = "",
    protein_id: str = "target",
    protein_fasta_path: str = "",
) -> str:
    """Predict Nesso activity metrics for a supplied molecule list.

    It returns cached target-protein and ligand predictions when available,
    calculates only unseen pairs, and is suitable for comparing current-round
    candidates.

    Args:
        smiles_list: String representation of a Python/JSON list of SMILES,
            for example ``'["CCO", "CCN"]'``.
        protein_sequence: Raw amino-acid sequence of the target protein.
        protein_id: Optional identifier used for the generated FASTA record.
        protein_fasta_path: Path to a target FASTA file. Supply exactly one
            protein input.

    Returns:
        String representation of a list of dictionaries. Each dictionary
        contains SMILES, Nesso_affinity_pred_value, and
        Nesso_affinity_probability_binary, matching the ADMET-AI critic format.
    """
    return await _run_blocking_nesso_call(
        "nesso_predict_by_cofolding",
        _predict_by_nesso_sync,
        smiles_list,
        _resolve_protein_input(protein_sequence, protein_fasta_path),
        protein_id,
    )


def signal_handler(_signum, _frame) -> None:
    LOGGER.info("Stopping local Nesso MCP server; resident worker is retained")
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    LOGGER.info("Starting local Nesso MCP server on port %s", get_mcp_port())
    LOGGER.info("Nesso batches use resident workers on GPUs=%s", _available_gpu_ids())
    LOGGER.info(
        "Available tools: nesso_filter_by_cofolding, nesso_predict_by_cofolding"
    )
    app.run(transport="streamable-http")
