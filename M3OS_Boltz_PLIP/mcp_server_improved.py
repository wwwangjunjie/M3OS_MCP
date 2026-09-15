#!/usr/bin/env python3
"""MCP server exposing independent Boltz and PLIP tools."""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server import FastMCP

from m3os.tools.generation import generate_complex_structure
from m3os.tools.interaction import analyze_complex_interactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mcp_port(default_port: int) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))


app = FastMCP("boltz2", host="0.0.0.0", port=get_mcp_port(8040))


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _get_positive_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


BOLTZ_MAX_CONCURRENT_REQUESTS = _get_positive_int_env(
    "BOLTZ_MAX_CONCURRENT_REQUESTS",
    1,
)
PLIP_MAX_CONCURRENT_REQUESTS = _get_positive_int_env(
    "PLIP_MAX_CONCURRENT_REQUESTS",
    4,
)
BOLTZ_QUEUE_TIMEOUT_SECONDS = _get_positive_float_env(
    "BOLTZ_QUEUE_TIMEOUT_SECONDS",
    7200.0,
)
PLIP_QUEUE_TIMEOUT_SECONDS = _get_positive_float_env(
    "PLIP_QUEUE_TIMEOUT_SECONDS",
    300.0,
)
BOLTZ_SUBPROCESS_TIMEOUT_SECONDS = _get_positive_float_env(
    "BOLTZ_SUBPROCESS_TIMEOUT_SECONDS",
    7200.0,
)
PLIP_SUBPROCESS_TIMEOUT_SECONDS = _get_positive_float_env(
    "PLIP_SUBPROCESS_TIMEOUT_SECONDS",
    300.0,
)

_boltz_semaphore = asyncio.Semaphore(BOLTZ_MAX_CONCURRENT_REQUESTS)
_plip_semaphore = asyncio.Semaphore(PLIP_MAX_CONCURRENT_REQUESTS)


def _serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


async def _run_blocking_tool(
    label: str,
    function: Callable[..., Any],
    *args: Any,
    semaphore: asyncio.Semaphore,
    queue_timeout_seconds: float,
) -> Any:
    """Run synchronous work off-loop with bounded admission and timing logs."""
    queued_at = time.perf_counter()
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=queue_timeout_seconds,
        )
    except TimeoutError:
        wait_seconds = time.perf_counter() - queued_at
        logger.warning(
            "%s queue timeout after %.3fs (limit %.3fs)",
            label,
            wait_seconds,
            queue_timeout_seconds,
        )
        return {
            "status": "error",
            "error_code": "QueueTimeout",
            "error_message": (
                f"{label} waited more than {queue_timeout_seconds:.3f}s "
                "for an execution slot"
            ),
            "queue_wait_seconds": wait_seconds,
        }

    started_at = time.perf_counter()
    work = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        result = await asyncio.shield(work)
        if isinstance(result, str):
            logger.error("%s returned an error: %s", label, result[:2000])
        return result
    except asyncio.CancelledError:
        logger.warning(
            "%s client disconnected; retaining its execution slot until the "
            "subprocess finishes",
            label,
        )
        try:
            await asyncio.shield(work)
        except Exception:
            logger.exception("Disconnected %s job later failed", label)
        raise
    except Exception:
        logger.exception("%s failed", label)
        raise
    finally:
        finished_at = time.perf_counter()
        semaphore.release()
        logger.info(
            "%s queue_wait=%.3fs run=%.3fs total=%.3fs",
            label,
            started_at - queued_at,
            finished_at - started_at,
            finished_at - queued_at,
        )


@app.tool()
async def generate_protein_ligand_complex(data: str) -> str:
    """
    Create one initial protein-ligand complex structure using Boltz.

    This tool only runs Boltz; it does not run PLIP interaction analysis.
    Provide a JSON dictionary with protein ``sequence`` and ligand ``smiles``
    keys. Boltz is computationally expensive. In M3OS it is reserved for
    Rational Designer analysis of the task's initial ligand when no uploaded
    complex structure is available. It must not be used for activity scoring,
    batch screening, or generated/optimized candidates; use Nesso for those
    activity workflows.

    Args:
        data: valid JSON Dictionary with "sequence" (str) and "smiles" (str).

    Returns:
        A JSON string containing the generated PDB structure path, Boltz output
        paths, and the predicted pIC50 value.
    """
    result = await _run_blocking_tool(
        "generate_protein_ligand_complex",
        generate_complex_structure,
        data,
        BOLTZ_SUBPROCESS_TIMEOUT_SECONDS,
        semaphore=_boltz_semaphore,
        queue_timeout_seconds=BOLTZ_QUEUE_TIMEOUT_SECONDS,
    )
    return _serialize_result(result)


@app.tool()
async def analyze_protein_ligand_interactions(structure_path: str) -> str:
    """Extract protein-ligand interactions from an existing complex structure.

    This tool only runs PLIP; it does not invoke Boltz. The structure may come
    from ``generate_protein_ligand_complex`` or from another compatible source.
    In M3OS it is reserved for the Rational Designer's initial-complex
    analysis. PLIP contacts are textual structural evidence, not activity
    predictions or candidate-ranking scores.

    Args:
        structure_path: Absolute or server-local path to a PDB, CIF, or mmCIF
            protein-ligand complex structure.

    Returns:
        A JSON string containing PLIP interaction counts/details and the saved
        interaction JSON path.
    """
    result = await _run_blocking_tool(
        "analyze_protein_ligand_interactions",
        analyze_complex_interactions,
        structure_path,
        PLIP_SUBPROCESS_TIMEOUT_SECONDS,
        semaphore=_plip_semaphore,
        queue_timeout_seconds=PLIP_QUEUE_TIMEOUT_SECONDS,
    )
    return _serialize_result(result)


# =============================================================================
# MAIN SERVER EXECUTION
# =============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutting down Boltz/PLIP MCP server...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting Boltz/PLIP MCP Server with FastMCP...")
        logger.info(
            "Available tools: generate_protein_ligand_complex, "
            "analyze_protein_ligand_interactions"
        )
        logger.info(
            "Concurrency limits: Boltz=%s PLIP=%s",
            BOLTZ_MAX_CONCURRENT_REQUESTS,
            PLIP_MAX_CONCURRENT_REQUESTS,
        )
        logger.info(
            "Queue timeouts: Boltz=%.1fs PLIP=%.1fs; subprocess timeouts: "
            "Boltz=%.1fs PLIP=%.1fs",
            BOLTZ_QUEUE_TIMEOUT_SECONDS,
            PLIP_QUEUE_TIMEOUT_SECONDS,
            BOLTZ_SUBPROCESS_TIMEOUT_SECONDS,
            PLIP_SUBPROCESS_TIMEOUT_SECONDS,
        )
        app.run(transport="streamable-http")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)
