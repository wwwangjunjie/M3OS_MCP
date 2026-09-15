#!/usr/bin/env python3
"""Tests for the host-local backend that do not require Nesso or a GPU."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SKILL_DIR = PROJECT_DIR / "nesso_cofolding_skill"
sys.path.insert(0, str(SKILL_DIR))

import local_client  # noqa: E402
import local_worker  # noqa: E402
import main as skill_main  # noqa: E402
from local_worker import prepare_local_inputs  # noqa: E402
from schema import Input  # noqa: E402


class LocalPreparationTests(unittest.TestCase):
    def test_csv_and_fasta_are_prepared_as_nesso_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_dir = root / "request"
            request_dir.mkdir()
            fasta = request_dir / "protein.fasta"
            csv_path = request_dir / "ligands.csv"
            fasta.write_text(">protein\nACDE\nFG\n", encoding="utf-8")
            pd.DataFrame(
                [{"id": "mol-1", "smiles": "CCO", "source_score": 0.5}]
            ).to_csv(csv_path, index=False)

            run_dir, summary = prepare_local_inputs(
                {
                    "output_root": str(request_dir),
                    "protein_fasta": str(fasta),
                    "ligand_csv": str(csv_path),
                    "task_name": "test",
                    "timestamp": "20260822_000000",
                },
                root,
            )
            document = json.loads(
                (run_dir / "inputs/test-0001.yaml").read_text(encoding="utf-8")
            )
            manifest = pd.read_csv(run_dir / "input_manifest.csv")

            self.assertEqual(
                document["sequences"][0]["protein"]["sequence"], "ACDEFG"
            )
            self.assertEqual(
                document["sequences"][1]["ligand"]["smiles"], "CCO"
            )
            self.assertEqual(manifest.loc[0, "Nesso_record_id"], "test-0001")
            self.assertEqual(manifest.loc[0, "source_score"], 0.5)
            self.assertEqual(summary["status"], "prepared")

    def test_multiple_fasta_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fasta = root / "protein.fasta"
            csv_path = root / "ligands.csv"
            fasta.write_text(">one\nACDE\n>two\nFGHI\n", encoding="utf-8")
            csv_path.write_text("smiles\nCCO\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one non-empty FASTA"):
                prepare_local_inputs(
                    {
                        "output_root": str(root),
                        "protein_fasta": str(fasta),
                        "ligand_csv": str(csv_path),
                        "task_name": "test",
                        "timestamp": "20260822_000000",
                    },
                    root,
                )

    def test_parallel_preprocessing_preserves_input_order_and_failures(self):
        class FakeExecutor:
            def submit(self, function, yaml_path, *args):
                future = Future()
                try:
                    future.set_result(function(yaml_path, *args))
                except Exception as exc:  # noqa: BLE001 - exercise failure path
                    future.set_exception(exc)
                return future

        class FakeNessoMain:
            @staticmethod
            def _process_single_yaml(yaml_path, *_args):
                if yaml_path.stem == "second":
                    raise ValueError("invalid molecule")
                return SimpleNamespace(id=yaml_path.stem)

        class FakeManifest:
            def __init__(self, records):
                self.records = records

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = SimpleNamespace(
                mol_dir=root / "mols",
                structures_dir=root / "structures",
                records_dir=root / "records",
            )
            runtime = object.__new__(local_worker.LocalNessoRuntime)
            runtime.preprocess_workers = 4
            runtime.nesso_main = FakeNessoMain
            runtime.Manifest = FakeManifest
            runtime._get_preprocess_executor = lambda: FakeExecutor()

            with mock.patch.object(local_worker.traceback, "print_exc"):
                manifest, failed = runtime._preprocess_yamls(
                    [
                        root / "first.yaml",
                        root / "second.yaml",
                        root / "third.yaml",
                    ],
                    paths,
                )

        self.assertEqual(
            [record.id for record in manifest.records], ["first", "third"]
        )
        self.assertEqual(failed, ["second"])


class LocalClientTests(unittest.TestCase):
    def test_stale_queue_jobs_are_quarantined(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            for state in ("pending", "running"):
                state_dir = queue_dir / state
                state_dir.mkdir()
                (state_dir / f"{state}-job.json").write_text("{}", encoding="utf-8")

            moved = local_client._quarantine_stale_jobs(queue_dir)

            self.assertEqual(len(moved), 2)
            self.assertFalse(list((queue_dir / "pending").glob("*.json")))
            self.assertFalse(list((queue_dir / "running").glob("*.json")))
            self.assertEqual(len(list((queue_dir / "abandoned").glob("*.json"))), 2)

    def test_failed_job_is_removed_from_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            (queue_dir / "pending").mkdir()
            (queue_dir / "running").mkdir()
            (queue_dir / "results").mkdir()
            with (
                mock.patch.object(local_client, "_worker_running", return_value=False),
                mock.patch.dict(
                    os.environ,
                    {
                        "NESSO_LOCAL_JOB_TIMEOUT_SECONDS": "2",
                        "NESSO_LOCAL_POLL_SECONDS": "0.01",
                    },
                ),
            ):
                with self.assertRaises(local_client.LocalWorkerError):
                    local_client.submit_local_job(
                        queue_dir=queue_dir,
                        pid=123,
                        worker_script=SKILL_DIR / "local_worker.py",
                        payload={"job_id": "failed-job", "task_name": "test"},
                    )
            self.assertFalse((queue_dir / "pending/failed-job.json").exists())
            self.assertFalse((queue_dir / "running/failed-job.json").exists())

    def test_virtualenv_python_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "base-python"
            link = root / "venv-python"
            target.touch()
            link.symlink_to(target)
            absolute = local_client._absolute_without_resolving_symlinks(link)
            self.assertEqual(absolute, link.absolute())
            self.assertNotEqual(absolute, link.resolve())

    def test_worker_command_uses_local_python(self):
        command = local_client.build_worker_command(
            python=Path("/opt/local/bin/python"),
            worker_script=SKILL_DIR / "local_worker.py",
            queue_dir=PROJECT_DIR / "nesso_runs/.nesso_local/gpu_4",
            cache_dir=PROJECT_DIR / "nesso_cache",
        )
        self.assertEqual(command[0], "/opt/local/bin/python")
        self.assertNotIn("docker", command)
        self.assertIn(str(SKILL_DIR / "local_worker.py"), command)

    def test_file_queue_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            (queue_dir / "pending").mkdir()
            (queue_dir / "results").mkdir()

            def fake_worker() -> None:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    requests = list((queue_dir / "pending").glob("*.json"))
                    if requests:
                        request = json.loads(
                            requests[0].read_text(encoding="utf-8")
                        )
                        local_client._atomic_write_json(
                            queue_dir / "results" / requests[0].name,
                            {
                                "success": True,
                                "job_id": request["job_id"],
                                "result_csv": "/tmp/result.csv",
                            },
                        )
                        return
                    time.sleep(0.01)

            worker = threading.Thread(target=fake_worker)
            worker.start()
            with (
                mock.patch.object(local_client, "_worker_running", return_value=True),
                mock.patch.dict(
                    os.environ,
                    {
                        "NESSO_LOCAL_JOB_TIMEOUT_SECONDS": "2",
                        "NESSO_LOCAL_POLL_SECONDS": "0.01",
                    },
                ),
            ):
                result = local_client.submit_local_job(
                    queue_dir=queue_dir,
                    pid=123,
                    worker_script=SKILL_DIR / "local_worker.py",
                    payload={"task_name": "test"},
                )
            worker.join(timeout=2)
            self.assertTrue(result["success"])
            self.assertFalse(worker.is_alive())


class LocalSkillIntegrationTests(unittest.TestCase):
    def test_skill_results_rank_by_binding_probability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_csv = Path(temp_dir) / "result.csv"
            pd.DataFrame(
                [
                    {
                        "SMILES": "CCO",
                        "Nesso_record_id": "low",
                        "Nesso_affinity_pred_value": 0.1,
                        "Nesso_affinity_probability_binary": 0.2,
                        "Nesso_status": "success",
                    },
                    {
                        "SMILES": "CCN",
                        "Nesso_record_id": "high",
                        "Nesso_affinity_pred_value": 1.5,
                        "Nesso_affinity_probability_binary": 0.9,
                        "Nesso_status": "success",
                    },
                ]
            ).to_csv(result_csv, index=False)
            inp = Input(
                protein_sequence="ACDE",
                ligand_csv_path="ligands.csv",
            )
            results = skill_main._make_results(result_csv, inp)
        self.assertEqual(results[0].nesso_record_id, "high")
        self.assertEqual(results[0].rank, 1)

    def test_prepare_only_preserves_output_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            request_dir = runs_root / "request"
            source_dir = Path(temp_dir) / "source"
            request_dir.mkdir(parents=True)
            source_dir.mkdir()
            fasta = source_dir / "protein.fasta"
            ligands = source_dir / "ligands.csv"
            fasta.write_text(">protein\nACDEFG\n", encoding="utf-8")
            ligands.write_text("id,smiles\nmol-1,CCO\n", encoding="utf-8")
            input_json = request_dir / "input.json"
            input_json.write_text("{}", encoding="utf-8")
            inp = Input(
                protein_fasta_path=str(fasta),
                ligand_csv_path=str(ligands),
                ligand_id_column="id",
                prepare_only=True,
            )

            def fake_run_local_job(**kwargs):
                payload = kwargs["payload"]
                run_dir, _summary = prepare_local_inputs(payload, runs_root)
                return (
                    {
                        "success": True,
                        "status": "prepared",
                        "run_dir": str(run_dir),
                    },
                    [str(kwargs["python"]), "local_worker.py"],
                )

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "NESSO_LOCAL_RUNS_ROOT": str(runs_root),
                        "NESSO_LOCAL_GPU_ID": "4",
                        "CUDA_VISIBLE_DEVICES": "4",
                    },
                ),
                mock.patch.object(
                    skill_main, "run_local_job", side_effect=fake_run_local_job
                ),
            ):
                output = skill_main.run_input(inp, input_json)

            self.assertTrue(output.success)
            self.assertEqual(output.result_csv, "")
            self.assertEqual(output.gpu_ids, ["4"])
            self.assertEqual(output.metadata["backend"], "local_resident_single_gpu")


if __name__ == "__main__":
    unittest.main()
