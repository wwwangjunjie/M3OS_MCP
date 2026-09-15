#!/usr/bin/env python3
"""Small streamable-HTTP benchmark for the ADMET prediction MCP tool."""

import argparse
import asyncio
import json
import math
import statistics
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_arguments(smiles_per_request: int) -> dict[str, str]:
    # Straight-chain alkanes provide deterministic, valid, unique benchmark
    # molecules without requiring an input fixture.
    smiles = ["C" * length for length in range(1, smiles_per_request + 1)]
    return {
        "smiles_list": json.dumps(smiles),
        "preference_json": json.dumps({"QED": "higher", "hERG": "lower"}),
    }


async def run_one_session(
    server_url: str,
    request_queue: asyncio.Queue,
    arguments: dict[str, str],
    latencies: list[float],
    failures: list[str],
    timeout_seconds: float,
) -> None:
    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            while True:
                request_index = await request_queue.get()
                if request_index is None:
                    request_queue.task_done()
                    return
                started = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(
                            "admet_predict_by_admetai",
                            arguments,
                        ),
                        timeout=timeout_seconds,
                    )
                    if result.isError:
                        failures.append(f"request {request_index}: MCP tool error")
                except Exception as exc:  # noqa: BLE001 - benchmark records all failures
                    failures.append(
                        f"request {request_index}: {exc.__class__.__name__}: {exc}"
                    )
                finally:
                    latencies.append(time.perf_counter() - started)
                    request_queue.task_done()


async def run_benchmark(args: argparse.Namespace) -> dict:
    arguments = make_arguments(args.smiles_per_request)

    if args.warmup:
        warmup_queue: asyncio.Queue = asyncio.Queue()
        for request_index in range(args.warmup):
            warmup_queue.put_nowait(request_index)
        warmup_queue.put_nowait(None)
        warmup_latencies: list[float] = []
        warmup_failures: list[str] = []
        await run_one_session(
            args.url,
            warmup_queue,
            arguments,
            warmup_latencies,
            warmup_failures,
            args.timeout_seconds,
        )
        if warmup_failures:
            raise RuntimeError("warm-up failed: " + "; ".join(warmup_failures))

    request_queue = asyncio.Queue()
    for request_index in range(args.requests):
        request_queue.put_nowait(request_index)
    for _ in range(args.concurrency):
        request_queue.put_nowait(None)

    latencies: list[float] = []
    failures: list[str] = []
    started = time.perf_counter()
    workers = [
        asyncio.create_task(
            run_one_session(
                args.url,
                request_queue,
                arguments,
                latencies,
                failures,
                args.timeout_seconds,
            )
        )
        for _ in range(args.concurrency)
    ]
    await asyncio.gather(*workers)
    wall_seconds = time.perf_counter() - started

    completed = args.requests - len(failures)
    return {
        "url": args.url,
        "requests": args.requests,
        "completed": completed,
        "failed": len(failures),
        "concurrency": args.concurrency,
        "smiles_per_request": args.smiles_per_request,
        "warmup_requests": args.warmup,
        "timeout_seconds": args.timeout_seconds,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(completed / wall_seconds, 6),
        "latency_seconds": {
            "min": round(min(latencies), 6) if latencies else None,
            "mean": round(statistics.fmean(latencies), 6) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 6) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 6) if latencies else None,
            "max": round(max(latencies), 6) if latencies else None,
        },
        "failures": failures[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/mcp",
        help="Streamable-HTTP MCP endpoint.",
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--smiles-per-request", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up requests excluded from reported measurements.",
    )
    args = parser.parse_args()
    for name in ("requests", "concurrency", "smiles_per_request"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_benchmark(parse_args())), indent=2))
