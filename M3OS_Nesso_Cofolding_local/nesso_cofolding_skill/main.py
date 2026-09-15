#!/usr/bin/env python3
"""Local single-GPU Nesso skill entry point."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from local_client import run_local_job
from schema import Input, Output, ResultRow

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent


class NoIdleGpuError(RuntimeError):
    pass


class RuntimeSettings:
    def __init__(self) -> None:
        self.python = Path(
            os.environ.get(
                "NESSO_LOCAL_PYTHON",
                str(PROJECT_DIR / ".nesso_venv" / "bin" / "python"),
            )
        ).expanduser().absolute()
        self.cache_dir = Path(
            os.environ.get("NESSO_CACHE_DIR", str(PROJECT_DIR / "nesso_cache"))
        ).expanduser().resolve()


def _safe_name(value: str, default: str = "Nesso") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return safe or default


def _resolve_user_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_or_materialize_protein_fasta(
    inp: Input, base_dir: Path
) -> tuple[Path, str]:
    if inp.protein_sequence is None:
        return _resolve_user_path(str(inp.protein_fasta_path), base_dir), "fasta_path"
    base_dir.mkdir(parents=True, exist_ok=True)
    fasta = base_dir / "protein_sequence.fasta"
    record_id = _safe_name(inp.protein_id or inp.job_name or "protein", "protein")
    wrapped = "\n".join(
        inp.protein_sequence[index:index + 80]
        for index in range(0, len(inp.protein_sequence), 80)
    )
    fasta.write_text(f">{record_id}\n{wrapped}\n", encoding="utf-8")
    return fasta.resolve(), "sequence"


def _query_gpus() -> list[dict[str, int]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "nvidia-smi failed")
    found = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            found.append(
                {"index": int(parts[0]), "memory": int(parts[1]), "util": int(parts[2])}
            )
    return found


def _visible_gpu_ids() -> set[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    allowed = {item.strip() for item in visible.split(",") if item.strip()}
    if any(not item.isdigit() for item in allowed):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES must contain physical numeric IDs: {visible!r}"
        )
    return allowed


def _validate_gpu_id(gpu_id: str) -> str:
    if not gpu_id.isdigit():
        raise ValueError("NESSO_LOCAL_GPU_ID must be one numeric physical GPU ID")
    allowed = _visible_gpu_ids()
    if allowed is not None and gpu_id not in allowed:
        raise ValueError(
            f"NESSO_LOCAL_GPU_ID={gpu_id} is outside CUDA_VISIBLE_DEVICES"
        )
    return gpu_id


def _select_gpu(runs_root: Path) -> str:
    configured = os.environ.get("NESSO_LOCAL_GPU_ID", "").strip()
    if configured:
        return _validate_gpu_id(configured)

    state_dir = runs_root / ".nesso_local"
    state_dir.mkdir(parents=True, exist_ok=True)
    selection_path = state_dir / "selected_gpu_id"
    with (state_dir / "gpu_selection.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if selection_path.is_file():
            return _validate_gpu_id(
                selection_path.read_text(encoding="utf-8").strip()
            )
        allowed = _visible_gpu_ids()
        candidates = _query_gpus()
        if allowed is not None:
            candidates = [
                gpu for gpu in candidates if str(gpu["index"]) in allowed
            ]
        if not candidates:
            raise NoIdleGpuError("No GPU is visible to the local Nesso skill")
        selected = str(
            min(
                candidates,
                key=lambda gpu: (gpu["memory"], gpu["util"], gpu["index"]),
            )["index"]
        )
        temporary = state_dir / f".selected_gpu_id.{os.getpid()}.tmp"
        temporary.write_text(selected + "\n", encoding="utf-8")
        os.replace(temporary, selection_path)
        return selected


def _stage_input(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    destination = destination.resolve()
    if source != destination:
        shutil.copy2(source, destination)
    return destination


def _pick_column(
    columns: list[str], requested: str | None, candidates: list[str]
) -> str | None:
    if requested:
        if requested not in columns:
            raise ValueError(f"Column {requested!r} not found; columns={columns}")
        return requested
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_results(result_csv: Path, inp: Input) -> list[ResultRow]:
    frame = pd.read_csv(result_csv)
    columns = list(frame.columns)
    id_column = _pick_column(
        columns,
        inp.ligand_id_column,
        ["INDEX", "ID", "ligand_id", "compound_id", "compound_name", "name"],
    )
    smiles_column = _pick_column(
        columns,
        inp.smiles_column,
        ["SMILES", "smiles", "smi", "canonical_smiles", "ligand"],
    )
    if smiles_column is None:
        raise ValueError(f"Cannot find SMILES column in {result_csv}")
    scored = frame.copy()
    scored["_score"] = pd.to_numeric(
        scored.get("Nesso_affinity_probability_binary"), errors="coerce"
    )
    scored = scored.sort_values("_score", ascending=False, na_position="last")
    rows = []
    for rank, (_, row) in enumerate(scored.iterrows(), start=1):
        ligand_id = (
            str(row.get(id_column, ""))
            if id_column
            else str(row.get("Nesso_record_id", ""))
        )
        confidence_warning = row.get("Nesso_confidence_warning", "")
        error = row.get("Nesso_error", "")
        rows.append(
            ResultRow(
                rank=rank,
                ligand_id=ligand_id,
                smiles=str(row.get(smiles_column, "")),
                nesso_record_id=str(row.get("Nesso_record_id", "")),
                nesso_affinity_pred_value=_float_or_none(
                    row.get("Nesso_affinity_pred_value")
                ),
                nesso_affinity_probability_binary=_float_or_none(
                    row.get("Nesso_affinity_probability_binary")
                ),
                nesso_entropy_pl=_float_or_none(row.get("Nesso_entropy_pl")),
                nesso_entropy_crop_pl=_float_or_none(
                    row.get("Nesso_entropy_crop_pl")
                ),
                status=str(row.get("Nesso_status", "")),
                confidence_warning=(
                    "" if pd.isna(confidence_warning) else str(confidence_warning)
                ),
                error="" if pd.isna(error) else str(error),
            )
        )
    return rows


def run_input(inp: Input, input_json_path: Path) -> Output:
    output_root = input_json_path.parent.resolve()
    runs_root_value = os.environ.get("NESSO_LOCAL_RUNS_ROOT", "").strip()
    if not runs_root_value:
        raise ValueError("NESSO_LOCAL_RUNS_ROOT is not configured")
    runs_root = Path(runs_root_value).expanduser().resolve()
    if output_root != runs_root and runs_root not in output_root.parents:
        raise ValueError(f"Local output root {output_root} is outside {runs_root}")

    fasta, protein_input_mode = _resolve_or_materialize_protein_fasta(
        inp, output_root
    )
    ligand_csv = _resolve_user_path(inp.ligand_csv_path, output_root)
    if not fasta.is_file():
        raise FileNotFoundError(f"Protein FASTA not found: {fasta}")
    if not ligand_csv.is_file():
        raise FileNotFoundError(f"Ligand CSV not found: {ligand_csv}")

    task_name = _safe_name(inp.job_name or ligand_csv.stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / "nesso_cofolding" / f"nesso_result_{timestamp}"
    staged_dir = output_root / "local_inputs"
    staged_fasta = _stage_input(fasta, staged_dir / "protein.fasta")
    staged_csv = _stage_input(ligand_csv, staged_dir / "ligands.csv")
    gpu_id = _select_gpu(runs_root)
    settings = RuntimeSettings()
    worker_result, command = run_local_job(
        gpu_id=gpu_id,
        skill_dir=HERE,
        runs_root=runs_root,
        python=settings.python,
        cache_dir=settings.cache_dir,
        payload={
            "task_name": task_name,
            "timestamp": timestamp,
            "output_root": str(output_root),
            "protein_fasta": str(staged_fasta),
            "ligand_csv": str(staged_csv),
            "smiles_column": inp.smiles_column,
            "max_records": inp.max_ligands,
            "prepare_only": inp.prepare_only,
            "seed": 42,
        },
    )
    log_path = output_root / f"nesso_local_{timestamp}.json"
    log_path.write_text(
        json.dumps(worker_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_json = run_dir / "run_summary.json"
    if not summary_json.is_file():
        raise FileNotFoundError(f"Nesso summary was not produced: {summary_json}")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    result_csv_value = str(summary.get("result_csv", "")).strip()
    result_csv = Path(result_csv_value) if result_csv_value else None
    results = (
        _make_results(result_csv, inp)
        if result_csv is not None and result_csv.is_file()
        else []
    )
    scored = [
        row
        for row in results
        if row.nesso_affinity_probability_binary is not None
    ]
    best = scored[0] if scored else None
    success = (
        summary.get("status") in {"success", "prepared"}
        and int(summary.get("n_failed", 0) or 0) == 0
    )
    output = Output(
        success=success,
        job_name=task_name,
        output_dir=str(run_dir),
        result_csv=str(result_csv) if result_csv is not None else "",
        summary_json=str(summary_json),
        input_manifest_csv=str(run_dir / "input_manifest.csv"),
        n_input_rows=int(summary.get("n_input_rows", 0) or 0),
        n_success=int(summary.get("n_success", 0) or 0),
        n_failed=int(summary.get("n_failed", 0) or 0),
        n_low_confidence=int(
            summary.get("n_low_confidence_entropy_crop_pl_zero", 0) or 0
        ),
        best_ligand_id=best.ligand_id if best else None,
        best_nesso_affinity_pred_value=(
            best.nesso_affinity_pred_value if best else None
        ),
        best_nesso_affinity_probability_binary=(
            best.nesso_affinity_probability_binary if best else None
        ),
        wall_seconds=_float_or_none(summary.get("wall_seconds")),
        gpu_ids=[gpu_id],
        results=results,
        command=command,
        error_code="" if success else "PARTIAL_FAILURE",
        error_message="" if success else f"Nesso status={summary.get('status')}",
        metadata={
            "algorithm_type": "cofolding_affinity",
            "primary_score_column": "Nesso_affinity_probability_binary",
            "score_direction": "higher_is_better",
            "affinity_value_column": "Nesso_affinity_pred_value",
            "affinity_value_direction": "lower_is_better",
            "confidence_columns": ["Nesso_entropy_pl", "Nesso_entropy_crop_pl"],
            "backend": "local_resident_single_gpu",
            "nesso_python": str(settings.python),
            "nesso_cache": str(settings.cache_dir),
            "protein_input_mode": protein_input_mode,
            "resolved_protein_fasta_path": str(fasta),
            "worker_result_path": str(log_path),
        },
    )
    (run_dir / "output.json").write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main.py input.json")
    input_json_path = Path(sys.argv[1]).resolve()
    try:
        inp = Input.model_validate_json(
            input_json_path.read_text(encoding="utf-8")
        )
        output = run_input(inp, input_json_path)
    except Exception as exc:  # noqa: BLE001 - serialize skill errors
        output = Output(
            success=False,
            job_name=input_json_path.stem,
            output_dir="",
            result_csv="",
            summary_json="",
            error_code=exc.__class__.__name__,
            error_message=str(exc),
        )
    output_path = input_json_path.parent / "output.json"
    output_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    print(output.model_dump_json(indent=2))
    if not output.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
