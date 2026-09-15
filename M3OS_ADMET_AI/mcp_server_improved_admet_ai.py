import asyncio
import os
import sys
import signal
import secrets
import string
import logging
import threading
import time


import ast
import csv
import json
from pathlib import Path

import pandas as pd

from mcp.server import FastMCP
from admet_prediction_admetai import (
    canonicalize_smiles,
    canonicalize_unique_smiles,
    generate_props_for_smiles,
    get_rank_based_candidates,
    select_props_for_preferences,
)
from admet_prediction_cache import AdmetPredictionCache
from leadopt_constraints import (
    assess_candidate_frame,
    parse_task_contract,
    write_constraint_results,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mcp_port(default_port: int) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))


app = FastMCP('admet_ai_property_prediction_server', host="0.0.0.0", port=get_mcp_port(8000))


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


ADMET_MAX_CONCURRENT_REQUESTS = _get_positive_int_env(
    "ADMET_MAX_CONCURRENT_REQUESTS", 1
)
_admet_request_semaphore = asyncio.Semaphore(ADMET_MAX_CONCURRENT_REQUESTS)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REINVENT_ROOT = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "M3OS_REINVENT_ROOT",
            os.path.join(PROJECT_ROOT, "M3OS_REINVENT"),
        )
    )
)
REINVENT_TMP_POOL = os.path.join(REINVENT_ROOT, "tmp", "pool")
REINVENT_POOL = os.path.join(REINVENT_ROOT, "pool")
M3OS_POOL = os.path.join(PROJECT_ROOT, "m3os", "pool")
GENERATED_CSV_PREFIXES = (
    "sampling_REINVENT4",
    "sampling_LibInvent",
    "sampling_Mol2Mol",
    "sampling_LinkInvent",
    "sampling_Reinvent",
)
GENERATED_CSV_SEARCH_DIRS = (
    REINVENT_TMP_POOL,
    REINVENT_POOL,
    M3OS_POOL,
)
TMP_DIR = Path(__file__).resolve().parent / "tmp"
_prediction_cache: AdmetPredictionCache | None = None
_prediction_cache_lock = threading.Lock()


def _get_prediction_cache() -> AdmetPredictionCache:
    global _prediction_cache
    if _prediction_cache is None:
        with _prediction_cache_lock:
            if _prediction_cache is None:
                _prediction_cache = AdmetPredictionCache()
    return _prediction_cache


def _source_smiles(source_df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in source_df.columns]
    smiles_column = next(
        (column for column in ("SMILES", "smiles") if column in columns),
        columns[0] if columns else None,
    )
    if smiles_column is None:
        raise ValueError("Candidate CSV has no columns")
    return [
        str(value).strip()
        for value in source_df[smiles_column].dropna().tolist()
        if str(value).strip()
    ]


def _predict_admet_frame_cached(
    smiles: list[str],
    *,
    source_tool: str,
) -> pd.DataFrame:
    canonical_smiles = canonicalize_unique_smiles(smiles)
    if not canonical_smiles:
        raise ValueError("No valid SMILES after RDKit canonicalization.")

    cache = _get_prediction_cache()
    try:
        cached = cache.get_many(canonical_smiles)
    except Exception:
        logger.exception("ADMET prediction cache lookup failed; computing requested molecules")
        cached = {}

    missing = [smiles for smiles in canonical_smiles if smiles not in cached]
    if missing:
        predicted = generate_props_for_smiles(missing)
        predicted_records = predicted.to_dict(orient="records")
        for record in predicted_records:
            key = str(record.get("SMILES") or "").strip()
            if key:
                cached[key] = dict(record)
        try:
            cache.upsert_many(predicted_records, source_tool=source_tool)
        except Exception:
            logger.exception("ADMET prediction cache write failed; returning fresh predictions")

    ordered_records = []
    for smiles_value in canonical_smiles:
        record = cached.get(smiles_value)
        if record is None:
            raise RuntimeError(f"ADMET produced no prediction row for {smiles_value}")
        normalized = dict(record)
        normalized["SMILES"] = smiles_value
        ordered_records.append(normalized)
    logger.info(
        "%s prediction cache hits=%d misses=%d total=%d",
        source_tool,
        len(canonical_smiles) - len(missing),
        len(missing),
        len(canonical_smiles),
    )
    return pd.DataFrame(ordered_records)


def _create_generation_metadata(smiles_path: str) -> dict:
    """Build compact generation metadata for LibInvent, Mol2Mol, and LinkInvent CSV outputs."""
    metadata = {}
    with open(smiles_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            smiles = (row.get("SMILES") or row.get("smiles") or "").strip()
            if not smiles:
                continue
            entry = {
                "Template": (
                    row.get("Scaffold")
                    or row.get("Input_SMILES")
                    or row.get("Warheads")
                    or ""
                ),
                "R-groups": (
                    row.get("R-groups")
                    or row.get("Linker")
                    or (f"Tanimoto={row.get('Tanimoto')}" if row.get("Tanimoto") else "")
                ),
            }
            metadata[smiles] = entry
            canonical = canonicalize_smiles(smiles)
            if canonical:
                metadata.setdefault(canonical, entry)
    return metadata


def _candidate_generated_csv_paths(rand_str: str) -> list[str]:
    if not rand_str:
        return []
    return [
        os.path.join(pool_dir, f"{prefix}_{rand_str}.csv")
        for pool_dir in GENERATED_CSV_SEARCH_DIRS
        for prefix in GENERATED_CSV_PREFIXES
    ]


def _resolve_generated_csv_path(rand_str: str) -> str | None:
    for candidate in _candidate_generated_csv_paths(rand_str):
        if os.path.exists(candidate):
            return candidate
    return None


def _format_missing_generated_csv_message(rand_str: str) -> str:
    candidates = _candidate_generated_csv_paths(rand_str)
    if not candidates:
        return "Generated molecule CSV not found: provide rand_str."

    searched = ", ".join(candidates)
    return f"Generated molecule CSV not found for rand_str={rand_str!r}. Searched: {searched}"


def _resolve_candidate_csv_path(candidate_csv_path: str, rand_str: str) -> Path:
    explicit_path = str(candidate_csv_path or "").strip()
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.suffix.lower() != ".csv":
            raise ValueError(f"candidate_csv_path must point to a CSV file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Candidate CSV not found: {path}")
        return path

    generated_path = _resolve_generated_csv_path(str(rand_str or "").strip())
    if generated_path is None:
        raise FileNotFoundError(_format_missing_generated_csv_message(rand_str))
    return Path(generated_path).resolve()


def _validate_top_n(top_n: int) -> int:
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    return top_n


async def _run_blocking_admet_call(label: str, function, *args) -> str:
    """Run synchronous ADMET work off the event loop with bounded admission."""
    request_started = time.perf_counter()
    async with _admet_request_semaphore:
        execution_started = time.perf_counter()
        wait_seconds = execution_started - request_started
        try:
            result = await asyncio.to_thread(function, *args)
        except Exception:
            logger.exception(
                "%s failed after queue_wait=%.3fs run=%.3fs",
                label,
                wait_seconds,
                time.perf_counter() - execution_started,
            )
            raise
        run_seconds = time.perf_counter() - execution_started
    logger.info(
        "%s completed queue_wait=%.3fs run=%.3fs total=%.3fs",
        label,
        wait_seconds,
        run_seconds,
        time.perf_counter() - request_started,
    )
    return result


def _filter_by_admetai_sync(
    preference_json: str,
    rand_str: str,
    candidate_csv_path: str,
    top_n: int,
) -> str:
    selected_top_n = _validate_top_n(top_n)
    smiles_path = _resolve_candidate_csv_path(candidate_csv_path, rand_str)
    source_df = pd.read_csv(smiles_path)
    props_df = _predict_admet_frame_cached(
        _source_smiles(source_df),
        source_tool="admet_filter_by_admetai",
    )
    smiles2template_rgroups = _create_generation_metadata(str(smiles_path))
    intermediate_rand_str = "".join(
        secrets.choice(string.ascii_lowercase) for _ in range(10)
    )
    candidates = get_rank_based_candidates(
        preference_json=preference_json,
        df=props_df,
        smiles2template_rgroups=smiles2template_rgroups,
        output_suffix=intermediate_rand_str,
        top_n=selected_top_n,
        source_df=source_df,
    )
    output_csv = (TMP_DIR / f"admet_ranked_{intermediate_rand_str}.csv").resolve()
    payload = {
        "status": "success",
        "stage": "admet",
        "rand_str": str(rand_str or "").strip(),
        "input_csv_path": str(smiles_path),
        "output_csv_path": str(output_csv),
        "input_count": int(len(source_df)),
        "output_count": int(len(candidates)),
        "candidates": candidates,
    }
    return json.dumps(payload, ensure_ascii=False)


def _filter_by_task_constraints_sync(
    task_contract_json: str,
    reference_smiles: str,
    rand_str: str,
    candidate_csv_path: str,
) -> str:
    contract = parse_task_contract(task_contract_json)
    smiles_path = _resolve_candidate_csv_path(candidate_csv_path, rand_str)
    source_df = pd.read_csv(smiles_path)
    canonical_smiles = canonicalize_unique_smiles(_source_smiles(source_df))
    if not canonical_smiles:
        raise ValueError("Candidate CSV contains no valid SMILES")
    props_df = _predict_admet_frame_cached(
        canonical_smiles,
        source_tool="admet_filter_by_task_constraints",
    )
    assessed = assess_candidate_frame(
        source_df,
        props_df,
        contract,
        reference_smiles=str(reference_smiles or contract.get("reference_smiles") or ""),
    )
    suffix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    scored_csv = (TMP_DIR / f"leadopt_constraints_scored_{suffix}.csv").resolve()
    eligible_csv = (TMP_DIR / f"leadopt_constraints_pass_{suffix}.csv").resolve()
    eligible = write_constraint_results(
        assessed,
        scored_csv=scored_csv,
        eligible_csv=eligible_csv,
    )
    candidates = [
        {"SMILES": str(value)}
        for value in eligible.get("SMILES", pd.Series(dtype=str)).head(10).tolist()
    ]
    return json.dumps(
        {
            "status": "success",
            "stage": "leadopt_constraints",
            "rand_str": str(rand_str or "").strip(),
            "input_csv_path": str(smiles_path),
            "scored_csv_path": str(scored_csv),
            "output_csv_path": str(eligible_csv),
            "input_count": int(len(assessed)),
            "output_count": int(len(eligible)),
            "candidates": candidates,
        },
        ensure_ascii=False,
    )


def _predict_by_admetai_sync(smiles_list: str, preference_json: str) -> str:
    parsed_smiles_list = ast.literal_eval(smiles_list)
    if not isinstance(parsed_smiles_list, (list, tuple)):
        raise TypeError("smiles_list must be a Python/JSON list of SMILES")
    props_df = _predict_admet_frame_cached(
        [str(value) for value in parsed_smiles_list],
        source_tool="admet_predict_by_admetai",
    )
    selected_props_json = select_props_for_preferences(
        props_df,
        preference_json,
    ).to_dict(orient="records")
    return str(selected_props_json)


@app.tool()
async def admet_filter_by_admetai(
    preference_json: str,
    rand_str: str = "",
    candidate_csv_path: str = "",
    top_n: int = 5,
) -> str:
    """
    Filters and ranks candidate molecules based on their predicted ADMET properties using ADMET-AI.

    This tool calculates pharmacokinetic properties for a set of molecules and identifies the
    requested number of candidates that best align with user-defined optimization goals. Raw
    predictions are persisted for reuse by later evaluation calls.

    Use this ranking tool when the selected ADMET objectives are monotonic and
    the task has no fully specified absolute, interval, hold, or
    baseline-relative constraint. A property with only an increase/decrease
    direction is a ranking preference, not a runtime constraint; do not invent
    a task contract for it.

    Note: You should only select the attributes that are most directly relevant to the current task for prediction. You do not need to consider too many factors. Avoid requiring predictions for a large number of attributes, as this will interfere with your judgment. In general, only handle a very small number of attributes!!!

    Supported exact physicochemical property keys are: "molecular_weight", "logP",
    "hydrogen_bond_acceptors", "hydrogen_bond_donors", "Lipinski", "QED",
    "stereo_centers", and "tpsa". Here "logP" is RDKit MolLogP.

    Args:
        preference_json (str): A JSON-formatted string defining the optimization direction for each property.
            - Keys: ADMET property names (e.g., "BBB_Martins", "Bioavailability_Ma", "hERG").
            - Values: Optimization direction, either "lower" or "higher". Only higher or lower!!!
            - Logic: 
                - For regression variables: "lower"/"higher" refers to the numerical value.
                - For classification variables: "higher" prioritizes molecules likely to be True (probability closer to 1), 
                  while "lower" prioritizes molecules likely to be False (probability closer to 0).
                - Metadata preferences such as "Intermediate" or "Context_Specific" are descriptive only and are not valid values here. If the task explicitly asks to increase or decrease such a property, use "higher" or "lower", respectively.
                - Target-value, interval, or "keep intermediate" objectives are not supported by this monotonic ranker; do not translate them into an arbitrary "higher" or "lower" direction.
            - Example: '{"Lipinski": "higher", "Bioavailability_Ma": "higher", "BBB_Martins": "higher"}'
        rand_str (str): A random string used for unique file naming. The random string ensures that each generation task has a unique identifier, preventing file naming conflicts when multiple tasks are run concurrently.
        candidate_csv_path (str): Exact path to a candidate CSV. When provided,
            this takes precedence over rand_str and may point to a previous
            filtering stage.
        top_n (int): Number of ranked candidates to retain.

    Returns:
        str: A JSON string containing the selected candidates and the output
            CSV path for an optional downstream filtering stage.

    Note: 
        The selected property preference should be most directly related to the current optimization task. 
        For example, if optimizing BBBP, set the preference only to BBB_Martins and do not consider other properties; 
        if optimizing both BBBP and hERG simultaneously, set the preference only to BBB_Martins and hERG, and do not consider other properties. 
        In any case, focus on the properties directly related to the current optimization task and involve as few properties as possible.
    """
    return await _run_blocking_admet_call(
        "admet_filter_by_admetai",
        _filter_by_admetai_sync,
        preference_json,
        rand_str,
        candidate_csv_path,
        top_n,
    )


@app.tool()
async def admet_filter_by_task_constraints(
    task_contract_json: str,
    reference_smiles: str,
    rand_str: str = "",
    candidate_csv_path: str = "",
) -> str:
    """Return candidates satisfying a runtime-defined task contract.

    This is not a general ADMET-ranking tool. Call it only when the JSON
    contract contains at least one fully specified, directly evaluable
    constraint: an operator with its value or interval; a hold with baseline
    and tolerance; or a baseline-relative objective with an exact
    source/endpoint, baseline, direction, and numeric threshold. A property
    with only an increase/decrease direction is not a constraint; use
    ``admet_filter_by_admetai`` for that monotonic ranking task. Do not invent
    missing fields to make this tool applicable.

    The JSON contract may contain optimization objectives, hold tolerances,
    and hard constraints. Each hard constraint supplies its property, source
    (``rdkit``, ``admet``, or ``activity``), operator, and runtime threshold.
    ADMET constraints may name an exact ADMET-AI endpoint. Activity constraints
    are validated and recorded for the downstream activity model rather than
    evaluated here. The eligible CSV is not truncated before downstream set
    intersection.
    """
    return await _run_blocking_admet_call(
        "admet_filter_by_task_constraints",
        _filter_by_task_constraints_sync,
        task_contract_json,
        reference_smiles,
        rand_str,
        candidate_csv_path,
    )


@app.tool()
async def admet_predict_by_admetai(smiles_list: str, preference_json: str) -> str:
    """
    Predicts ADMET properties for a provided list of molecules using ADMET-AI.

    This tool returns cached pharmacokinetic profiles when available and calculates only
    previously unseen molecules for the requested list.

    Note: You should only select the attributes that are most directly relevant to the current task for prediction. You do not need to consider too many factors. Avoid requiring predictions for a large number of attributes, as this will interfere with your judgment. In general, only handle a very small number of attributes!!!

    Supported exact physicochemical property keys are: "molecular_weight", "logP",
    "hydrogen_bond_acceptors", "hydrogen_bond_donors", "Lipinski", "QED",
    "stereo_centers", and "tpsa". Here "logP" is RDKit MolLogP.

    Args:
        smiles_list (str): A string representation of a Python list containing SMILES strings of molecules.
            - Example: '["C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1C(=O)O"]'
        preference_json (str): A JSON-formatted string defining the optimization direction for each property.
            - Keys: ADMET property names (e.g., "BBB_Martins", "Bioavailability_Ma", "hERG").
            - Values: Optimization direction, either "lower" or "higher". Only higher or lower!!!
            - Logic: 
                - For regression variables: "lower"/"higher" refers to the numerical value.
                - For classification variables: "higher" prioritizes molecules likely to be True (probability closer to 1), 
                  while "lower" prioritizes molecules likely to be False (probability closer to 0).
                - Metadata preferences such as "Intermediate" or "Context_Specific" are descriptive only and are not valid values here. If the task explicitly asks to increase or decrease such a property, use "higher" or "lower", respectively.
                - Target values and intervals cannot be encoded in preference_json. This tool can return the selected raw property values for separate comparison, but preference_json itself must still use only "higher" or "lower".
            - Example: '{"Lipinski": "higher", "Bioavailability_Ma": "higher", "BBB_Martins": "higher"}'

    Returns:
        str: A string representation of a list of dictionaries. Each dictionary contains 
            the SMILES string and its corresponding predicted property values.
    """
    return await _run_blocking_admet_call(
        "admet_predict_by_admetai",
        _predict_by_admetai_sync,
        smiles_list,
        preference_json,
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutting down m3os MCP server...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting m3os MCP Server with FastMCP...")
        logger.info(
            "ADMET request concurrency limit: %s",
            ADMET_MAX_CONCURRENT_REQUESTS,
        )
        logger.info("Available tools: 3 tools across 1 categories")
        logger.info("Available prompts: 0 expert prompt templates")
        logger.info("Available resources: 0 information resources")
        app.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)
