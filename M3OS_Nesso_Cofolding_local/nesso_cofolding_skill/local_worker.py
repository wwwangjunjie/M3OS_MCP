#!/usr/bin/env python3
"""Single-GPU resident Nesso worker running directly on the host."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

VENDORED_NESSO_DIR = Path(__file__).resolve().parent.parent / "vendor" / "nesso"
if str(VENDORED_NESSO_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_NESSO_DIR))

SCORE_KEYS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
    "entropy_crop_pp",
    "entropy_crop_pl",
    "entropy_crop_ll",
)


def _initialize_preprocess_worker(
    ccd_pkl: Path | None,
    startup_barrier: Any,
    startup_timeout_seconds: float,
) -> None:
    from nesso.main import _init_worker

    _init_worker(ccd_pkl)
    startup_barrier.wait(timeout=startup_timeout_seconds)


def _preprocess_worker_probe() -> int:
    return os.getpid()


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_path_within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path {resolved} is outside resident runs root {root}")
    return resolved


def _physical_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [item.strip() for item in visible.split(",") if item.strip()] or ["0"]


def _read_single_fasta(path: Path) -> str:
    """Read exactly one FASTA record and return its normalized sequence."""
    records: list[str] = []
    current: list[str] = []
    saw_header = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if saw_header:
                records.append("".join(current))
                current = []
            saw_header = True
            continue
        if not saw_header:
            raise ValueError(f"FASTA sequence appears before a header in {path}")
        current.append("".join(line.split()).upper())
    if saw_header:
        records.append("".join(current))
    if len(records) != 1 or not records[0]:
        raise ValueError(f"Expected one non-empty FASTA record in {path}")
    return records[0]


def _pick_smiles_column(frame: pd.DataFrame, requested: str | None) -> str:
    columns = [str(column) for column in frame.columns]
    if requested:
        if requested not in columns:
            raise ValueError(
                f"SMILES column {requested!r} not found; columns={columns}"
            )
        return requested
    lowered = {column.lower(): column for column in columns}
    for candidate in ("smiles", "smi", "canonical_smiles", "ligand"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Cannot find a SMILES column; columns={columns}")


def prepare_local_inputs(
    payload: dict[str, Any], runs_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Convert one CSV and one FASTA into official Nesso YAML inputs."""
    started = time.perf_counter()
    output_root = _safe_path_within(Path(payload["output_root"]), runs_root)
    ligand_csv = _safe_path_within(Path(payload["ligand_csv"]), runs_root)
    protein_fasta = _safe_path_within(Path(payload["protein_fasta"]), runs_root)
    task_name = str(payload["task_name"])
    timestamp = str(payload["timestamp"])
    run_dir = output_root / "nesso_cofolding" / f"nesso_result_{timestamp}"
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    protein_sequence = _read_single_fasta(protein_fasta)
    frame = pd.read_csv(ligand_csv)
    max_records = payload.get("max_records")
    if max_records is not None:
        frame = frame.head(int(max_records)).copy()
    if frame.empty:
        raise ValueError(f"Ligand CSV contains no input rows: {ligand_csv}")
    smiles_column = _pick_smiles_column(frame, payload.get("smiles_column"))

    manifest_rows: list[dict[str, Any]] = []
    for source_row, (_, source) in enumerate(frame.iterrows(), start=1):
        raw_smiles = source.get(smiles_column)
        smiles = "" if pd.isna(raw_smiles) else str(raw_smiles).strip()
        record_id = f"{task_name}-{source_row:04d}"
        yaml_path = inputs_dir / f"{record_id}.yaml"
        yaml_document = {
            "sequences": [
                {"protein": {"id": "A", "sequence": protein_sequence}},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        # JSON is a YAML 1.2 subset and avoids adding another preparation-time
        # dependency to the MCP process.
        yaml_path.write_text(
            json.dumps(yaml_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                **source.to_dict(),
                "Nesso_record_id": record_id,
                "Nesso_source_row": source_row,
                "Nesso_yaml_path": str(yaml_path),
            }
        )

    manifest_path = run_dir / "input_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    summary = {
        "task_name": task_name,
        "timestamp_utc": timestamp,
        "input_csv": str(ligand_csv),
        "input_fasta": str(protein_fasta),
        "smiles_column": smiles_column,
        "protein_sequence_length": len(protein_sequence),
        "gpu_ids": _physical_gpu_ids(),
        "logical_gpu_id": "0",
        "devices": 1,
        "n_input_rows": len(manifest_rows),
        "n_success": 0,
        "n_failed": 0,
        "n_low_confidence_entropy_crop_pl_zero": 0,
        "status": "prepared",
        "result_csv": "",
        "wall_seconds": time.perf_counter() - started,
        "backend": "local_resident_single_gpu",
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    return run_dir, summary


def aggregate_predictions(
    *,
    input_manifest_csv: Path,
    predictions_dir: Path,
    result_csv: Path,
) -> dict[str, int]:
    """Join per-record affinity JSON files back to the prepared input manifest."""
    manifest = pd.read_csv(input_manifest_csv)
    if "Nesso_record_id" not in manifest.columns:
        raise ValueError(f"Missing Nesso_record_id in {input_manifest_csv}")

    helper_columns = {"Nesso_record_id", "Nesso_source_row", "Nesso_yaml_path"}
    source_columns = [
        column for column in manifest.columns if column not in helper_columns
    ]
    rows: list[dict[str, Any]] = []
    n_success = 0
    n_low_confidence = 0

    for _, source in manifest.iterrows():
        record_id = str(source["Nesso_record_id"])
        row = {column: source[column] for column in source_columns}
        row["Nesso_record_id"] = record_id
        prediction_path = predictions_dir / record_id / "affinity.json"
        error = ""
        prediction: dict[str, Any] = {}
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - record per-molecule failures
            error = str(exc)

        for key in SCORE_KEYS:
            row[f"Nesso_{key}"] = prediction.get(key)
        success = prediction.get("affinity_pred_value") is not None and not error
        warning = ""
        if success:
            n_success += 1
            if prediction.get("entropy_crop_pl") == 0:
                n_low_confidence += 1
                warning = "Nesso_entropy_crop_pl is zero"
        row["Nesso_status"] = "success" if success else "failed"
        row["Nesso_confidence_warning"] = warning
        row["Nesso_error"] = error
        rows.append(row)

    result_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_csv, index=False)
    return {
        "n_input_rows": len(rows),
        "n_success": n_success,
        "n_failed": len(rows) - n_success,
        "n_low_confidence": n_low_confidence,
    }


class LocalNessoRuntime:
    """Own the reusable model, CCD data, ESM model, and CUDA allocator state."""

    def __init__(self, *, cache_dir: Path, no_kernels: bool, seed: int):
        import lightning.pytorch as pl
        import nesso.data.inference as inference_module
        import nesso.main as nesso_main
        import torch
        import yaml as pyyaml
        from nesso.data.esm import (
            DEFAULT_ESM2_MODEL,
            extract_esm_embedding,
            setup_esm_model,
        )
        from nesso.data.inference import STANDARD_AA, NessoInferenceDataModule
        from nesso.data.types import Manifest
        from nesso.data.writer import NessoWriter
        from nesso.data.yaml_input import validate_schema
        from nesso.model.models.nesso1 import Nesso1
        from safetensors.torch import save_file

        self.pl = pl
        self.save_file = save_file
        self.torch = torch
        self.nesso_main = nesso_main
        self.inference_module = inference_module
        self.NessoInferenceDataModule = NessoInferenceDataModule
        self.Manifest = Manifest
        self.NessoWriter = NessoWriter
        self.validate_schema = validate_schema
        self.pyyaml = pyyaml
        self.DEFAULT_ESM2_MODEL = DEFAULT_ESM2_MODEL
        self.extract_esm_embedding = extract_esm_embedding
        self.setup_esm_model = setup_esm_model
        self.cache_dir = cache_dir
        self.no_kernels = no_kernels
        self.seed = seed
        self.preprocess_workers = _positive_int_env("NESSO_PREPROCESS_WORKERS", 4)
        self.preprocess_start_timeout_seconds = _positive_float_env(
            "NESSO_PREPROCESS_START_TIMEOUT_SECONDS", 600.0
        )
        self._preprocess_context = multiprocessing.get_context("spawn")
        self._preprocess_executor: ProcessPoolExecutor | None = None
        self.device = torch.device("cuda:0")
        self.esm_model = None
        self.esm_tokenizer = None

        if not torch.cuda.is_available():
            raise RuntimeError("Local Nesso worker requires a visible CUDA GPU")
        torch.set_float32_matmul_precision("highest")
        torch.set_grad_enabled(False)
        pl.seed_everything(seed, workers=True)

        revision = nesso_main.resolve_model_revision(None)
        self.ccd_pkl, checkpoint = nesso_main.ensure_cache(
            cache_dir,
            revision=revision,
        )

        # Keep a parent-process CCD view for inference-time residue featurization.
        nesso_main._init_worker(self.ccd_pkl)
        ccd_dict = nesso_main._worker_ccd_dict or {}
        standard_aa_mols = {name: ccd_dict.get(name) for name in STANDARD_AA}
        inference_module.load_standard_aa_mols = lambda _path=None: standard_aa_mols

        self.model = Nesso1.from_pretrained(checkpoint)
        if no_kernels:
            self.model.use_kernels = False
        self.model.eval().to(self.device)
        torch.cuda.synchronize()

    def _get_preprocess_executor(self) -> ProcessPoolExecutor:
        if self._preprocess_executor is None:
            startup_barrier = self._preprocess_context.Barrier(
                self.preprocess_workers
            )
            self._preprocess_executor = ProcessPoolExecutor(
                max_workers=self.preprocess_workers,
                mp_context=self._preprocess_context,
                initializer=_initialize_preprocess_worker,
                initargs=(
                    self.ccd_pkl,
                    startup_barrier,
                    self.preprocess_start_timeout_seconds,
                ),
            )
        return self._preprocess_executor

    def warm_preprocess_pool(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.preprocess_workers == 1:
            return {
                "preprocess_worker_pids": [os.getpid()],
                "preprocess_warmup_seconds": 0.0,
            }

        executor = self._get_preprocess_executor()
        futures = [
            executor.submit(_preprocess_worker_probe)
            for _ in range(self.preprocess_workers)
        ]
        worker_pids = sorted(
            {
                future.result(timeout=self.preprocess_start_timeout_seconds)
                for future in futures
            }
        )
        return {
            "preprocess_worker_pids": worker_pids,
            "preprocess_warmup_seconds": time.perf_counter() - started,
        }

    def close(self) -> None:
        if self._preprocess_executor is not None:
            self._preprocess_executor.shutdown(wait=True, cancel_futures=True)
            self._preprocess_executor = None

    def _preprocess_yamls(
        self,
        yaml_paths: list[Path],
        paths: Any,
    ) -> tuple[Any, list[str]]:
        paths.structures_dir.mkdir(parents=True, exist_ok=True)
        paths.records_dir.mkdir(parents=True, exist_ok=True)

        if self.preprocess_workers == 1 or len(yaml_paths) == 1:
            records = []
            failed = []
            for yaml_path in yaml_paths:
                try:
                    record = self.nesso_main._process_single_yaml(
                        yaml_path,
                        paths.mol_dir,
                        paths.structures_dir,
                        paths.records_dir,
                    )
                    records.append(record)
                except Exception:  # noqa: BLE001 - preserve per-record failures
                    failed.append(yaml_path.stem)
                    traceback.print_exc()
            return self.Manifest(records), failed

        executor = self._get_preprocess_executor()
        records_by_index: list[Any | None] = [None] * len(yaml_paths)
        failed_by_index: list[str | None] = [None] * len(yaml_paths)
        futures = {
            executor.submit(
                self.nesso_main._process_single_yaml,
                yaml_path,
                paths.mol_dir,
                paths.structures_dir,
                paths.records_dir,
            ): (index, yaml_path)
            for index, yaml_path in enumerate(yaml_paths)
        }
        for future in as_completed(futures):
            index, yaml_path = futures[future]
            try:
                records_by_index[index] = future.result()
            except Exception:  # noqa: BLE001 - preserve per-record failures
                failed_by_index[index] = yaml_path.stem
                traceback.print_exc()

        records = [record for record in records_by_index if record is not None]
        failed = [name for name in failed_by_index if name is not None]
        return self.Manifest(records), failed

    def _ensure_esm_embeddings(
        self,
        *,
        yaml_paths: list[Path],
        job_esm_dir: Path,
        shared_esm_dir: Path,
    ) -> None:
        sequences, supplied_paths = self.nesso_main.collect_esm_from_yamls(yaml_paths)
        job_esm_dir.mkdir(parents=True, exist_ok=True)
        shared_esm_dir.mkdir(parents=True, exist_ok=True)

        for sequence_hash, supplied in supplied_paths.items():
            supplied_path = Path(supplied)
            if supplied_path.is_file():
                destination = shared_esm_dir / f"{sequence_hash}.safetensors"
                if not destination.exists():
                    shutil.copy2(supplied_path, destination)

        missing = {
            sequence_hash: sequence
            for sequence_hash, sequence in sequences.items()
            if not (shared_esm_dir / f"{sequence_hash}.safetensors").is_file()
        }
        if missing and self.esm_model is None:
            self.esm_model, self.esm_tokenizer = self.setup_esm_model(
                self.DEFAULT_ESM2_MODEL,
                self.device,
                cache_dir=self.cache_dir / "huggingface",
            )
        for sequence_hash, sequence in sorted(missing.items()):
            embedding = self.extract_esm_embedding(
                sequence,
                self.esm_model,
                self.esm_tokenizer,
            )
            self.save_file(
                {"embeddings": embedding},
                shared_esm_dir / f"{sequence_hash}.safetensors",
            )

        for sequence_hash in sequences:
            source = shared_esm_dir / f"{sequence_hash}.safetensors"
            destination = job_esm_dir / source.name
            if destination.exists():
                continue
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)

    def predict_job(
        self,
        *,
        inputs_dir: Path,
        output_dir: Path,
        shared_esm_dir: Path,
        seed: int,
    ) -> dict[str, Any]:
        # Match the official CLI seed placement. RDKit ETKDG remains stochastic
        # unless the conformer generator itself is given a random seed.
        self.pl.seed_everything(seed, workers=True)
        yaml_paths = self.nesso_main.check_inputs(inputs_dir)
        valid_yaml_paths = []
        invalid_schemas = []
        for yaml_path in yaml_paths:
            try:
                self.validate_schema(
                    self.pyyaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                )
            except Exception:  # noqa: BLE001 - retain per-record validation failures
                invalid_schemas.append(yaml_path.stem)
                traceback.print_exc()
            else:
                valid_yaml_paths.append(yaml_path)
        if not valid_yaml_paths:
            raise RuntimeError("No valid Nesso YAML inputs remain after validation")

        paths = self.nesso_main.resolve_paths(output_dir)
        preprocessing_started = time.perf_counter()
        manifest, failed_preprocessing = self._preprocess_yamls(
            valid_yaml_paths,
            paths,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        manifest.dump(paths.manifest_path)
        if not manifest.records:
            raise RuntimeError("No Nesso inputs could be preprocessed")

        self._ensure_esm_embeddings(
            yaml_paths=valid_yaml_paths,
            job_esm_dir=paths.esm_dir,
            shared_esm_dir=shared_esm_dir,
        )
        self.model.predict_args.update(
            {
                "pose_protein_cutoff": 15.0,
                "recycling_steps": 5,
                "affinity_protein_cutoff": 15.0,
                "refine_protein_inference": True,
                "refine_protein_cutoff": 22.0,
                "refine_protein_tokens_budget": 256,
                "save_metadata": False,
            }
        )
        dataloader_batch_size = int(
            os.environ.get("NESSO_DATALOADER_BATCH_SIZE", "32")
        )
        if dataloader_batch_size < 1:
            raise ValueError("NESSO_DATALOADER_BATCH_SIZE must be positive")
        datamodule = self.NessoInferenceDataModule(
            manifest=manifest,
            target_dir=paths.processed,
            esm_emb_dir=paths.esm_dir,
            ligand_dir=paths.mol_dir,
            ccd_pkl=self.ccd_pkl,
            num_workers=0,
            use_esm_all_layers=False,
            esm_emb_dim=1280,
            esm_num_layers=33,
            batch_size=dataloader_batch_size,
        )
        dataloader = datamodule.predict_dataloader()
        writer = self.NessoWriter(paths.predictions_dir, save_metadata=False)

        started = time.perf_counter()
        with (
            self.torch.inference_mode(),
            self.torch.autocast("cuda", dtype=self.torch.bfloat16),
        ):
            for batch_index, batch in enumerate(dataloader):
                batch = self.model.transfer_batch_to_device(batch, self.device)
                prediction = self.model.predict_step(batch, batch_index)
                writer.write_on_batch_end(
                    None,
                    self.model,
                    prediction,
                    [],
                    batch,
                    batch_index,
                    0,
                )
        self.torch.cuda.synchronize()
        return {
            "prediction_seconds": time.perf_counter() - started,
            "invalid_schemas": invalid_schemas,
            "failed_preprocessing": failed_preprocessing,
            "prediction_failures": writer.failed,
            "predictions_dir": str(paths.predictions_dir),
            "dataloader_batch_size": dataloader_batch_size,
            "preprocess_workers": self.preprocess_workers,
            "preprocessing_seconds": preprocessing_seconds,
        }


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.queue_dir = args.queue_dir.resolve()
        self.runs_root = self.queue_dir.parent.parent.resolve()
        self.running = True
        self.runtime: LocalNessoRuntime | None = None

    def stop(self, _signum=None, _frame=None) -> None:
        self.running = False

    def initialize(self) -> None:
        for child in ("pending", "running", "results", "completed", "esm_cache"):
            (self.queue_dir / child).mkdir(parents=True, exist_ok=True)
        # A container can be killed after claiming a job but before publishing
        # its result. Requeue those audit files before accepting new work.
        for active_path in sorted((self.queue_dir / "running").glob("*.json")):
            os.replace(active_path, self.queue_dir / "pending" / active_path.name)
        started = time.perf_counter()
        self.runtime = LocalNessoRuntime(
            cache_dir=self.args.cache_dir,
            no_kernels=self.args.no_kernels,
            seed=self.args.seed,
        )
        preprocess_warmup = self.runtime.warm_preprocess_pool()
        atomic_write_json(
            self.queue_dir / "ready.json",
            {
                "ready": True,
                "pid": os.getpid(),
                "gpu_id": "0",
                "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "startup_seconds": time.perf_counter() - started,
                "no_kernels": self.args.no_kernels,
                "dataloader_batch_size": int(
                    os.environ.get("NESSO_DATALOADER_BATCH_SIZE", "32")
                ),
                "preprocess_workers": self.runtime.preprocess_workers,
                **preprocess_warmup,
                "started_at_unix": time.time(),
            },
        )

    def _prepare_inputs(self, payload: dict[str, Any]) -> Path:
        run_dir, _summary = prepare_local_inputs(payload, self.runs_root)
        return run_dir

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.runtime is None:
            raise RuntimeError("Resident runtime has not been initialized")
        started = time.perf_counter()
        run_dir = self._prepare_inputs(payload)
        if payload.get("prepare_only"):
            summary_path = run_dir / "run_summary.json"
            return {
                "success": True,
                "status": "prepared",
                "run_dir": str(run_dir),
                "summary_json": str(summary_path),
                "wall_seconds": time.perf_counter() - started,
            }

        prediction = self.runtime.predict_job(
            inputs_dir=run_dir / "inputs",
            output_dir=run_dir / "output",
            shared_esm_dir=self.queue_dir / "esm_cache",
            seed=int(payload.get("seed", self.args.seed)),
        )
        result_csv = run_dir / (
            f"nesso_{payload['task_name']}_result_{payload['timestamp']}.csv"
        )
        counts = aggregate_predictions(
            input_manifest_csv=run_dir / "input_manifest.csv",
            predictions_dir=Path(prediction["predictions_dir"]),
            result_csv=result_csv,
        )
        wall_seconds = time.perf_counter() - started
        success = counts["n_failed"] == 0
        summary = {
            "task_name": payload["task_name"],
            "timestamp_utc": payload["timestamp"],
            "input_csv": payload["ligand_csv"],
            "input_fasta": payload["protein_fasta"],
            "smiles_column": payload.get("smiles_column"),
            "gpu_ids": _physical_gpu_ids(),
            "logical_gpu_id": "0",
            "devices": 1,
            **counts,
            "n_low_confidence_entropy_crop_pl_zero": counts["n_low_confidence"],
            "status": "success" if success else "partial_failure",
            "result_csv": str(result_csv),
            "nesso_exit_code": 0 if success else 1,
            "wall_seconds": wall_seconds,
            "prediction_seconds": prediction["prediction_seconds"],
            "dataloader_batch_size": prediction["dataloader_batch_size"],
            "preprocess_workers": prediction["preprocess_workers"],
            "preprocessing_seconds": prediction["preprocessing_seconds"],
            "backend": "local_resident_single_gpu",
            "model_parameter_policy": (
                "official model settings; resident single-GPU batched inference"
            ),
        }
        summary_path = run_dir / "run_summary.json"
        atomic_write_json(summary_path, summary)
        return {
            # The worker protocol succeeded even when individual molecules did
            # not. The run summary carries the partial-failure status so the
            # existing skill can return all usable per-molecule results.
            "success": True,
            "prediction_success": success,
            "status": summary["status"],
            "run_dir": str(run_dir),
            "result_csv": str(result_csv),
            "summary_json": str(summary_path),
            "wall_seconds": wall_seconds,
            "prediction_seconds": prediction["prediction_seconds"],
            "dataloader_batch_size": prediction["dataloader_batch_size"],
            "preprocess_workers": prediction["preprocess_workers"],
            "preprocessing_seconds": prediction["preprocessing_seconds"],
            **counts,
        }

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()

    def run_forever(self) -> None:
        pending_dir = self.queue_dir / "pending"
        running_dir = self.queue_dir / "running"
        completed_dir = self.queue_dir / "completed"
        results_dir = self.queue_dir / "results"
        while self.running:
            requests = sorted(pending_dir.glob("*.json"))
            if not requests:
                time.sleep(self.args.poll_seconds)
                continue
            request_path = requests[0]
            active_path = running_dir / request_path.name
            try:
                os.replace(request_path, active_path)
            except FileNotFoundError:
                continue
            job_id = active_path.stem
            try:
                payload = json.loads(active_path.read_text(encoding="utf-8"))
                result = {"job_id": job_id, **self.process(payload)}
            except Exception as exc:  # noqa: BLE001 - return structured job failure
                traceback.print_exc()
                result = {
                    "job_id": job_id,
                    "success": False,
                    "error_code": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            atomic_write_json(results_dir / f"{job_id}.json", result)
            os.replace(active_path, completed_dir / active_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/nesso"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--no-kernels", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    worker = Worker(args)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    try:
        worker.initialize()
        worker.run_forever()
    except Exception as exc:
        traceback.print_exc()
        atomic_write_json(
            args.queue_dir / "fatal.json",
            {
                "success": False,
                "error_code": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        worker.close()


if __name__ == "__main__":
    main()
