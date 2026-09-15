#!/usr/bin/env python3
"""Tests for the ADMET-AI-compatible Nesso activity tools."""

import asyncio
import ast
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import mcp_server_nesso_cofolding_local as server
import nesso_activity_tools as activity
from nesso_prediction_cache import NessoPredictionCache, protein_sequence_hash


class ActivityFormatTests(unittest.TestCase):
    def test_reinvent_frame_is_normalized_and_deduplicated(self):
        frame = pd.DataFrame(
            {
                "smiles": [" CCO ", "CCO", None, "CCN"],
                "Scaffold": ["a", "duplicate", "missing", "b"],
            }
        )
        normalized = activity.normalize_generated_frame(frame)
        self.assertEqual(normalized["SMILES"].tolist(), ["CCO", "CCN"])
        self.assertEqual(normalized["Scaffold"].tolist(), ["a", "b"])

    def test_probability_ranking_and_admet_style_output(self):
        frame = pd.DataFrame(
            [
                {
                    "SMILES": "CCO",
                    "Scaffold": "template-low",
                    "R-groups": "r-low",
                    "Nesso_affinity_pred_value": 0.2,
                    "Nesso_affinity_probability_binary": 0.1,
                    "Nesso_entropy_pl": 0.5,
                },
                {
                    "SMILES": "CCN",
                    "Scaffold": "template-high",
                    "R-groups": "r-high",
                    "Nesso_affinity_pred_value": 1.5,
                    "Nesso_affinity_probability_binary": 0.9,
                    "Nesso_entropy_pl": 0.8,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv = Path(temp_dir) / "ranked.csv"
            selected = activity.rank_by_binding_probability(
                frame,
                output_csv=output_csv,
                top_n=5,
            )
            written = pd.read_csv(output_csv)

        self.assertEqual(selected[0]["SMILES"], "CCN")
        self.assertEqual(selected[0]["Template"], "template-high")
        self.assertEqual(selected[0]["R-groups"], "r-high")
        self.assertEqual(written["SMILES"].tolist(), ["CCN", "CCO"])
        self.assertNotIn("Nesso_entropy_pl", written.columns)
        self.assertIn("Nesso_affinity_pred_value", written.columns)
        self.assertIn("Nesso_affinity_probability_binary", written.columns)

    def test_prediction_records_return_only_requested_metrics(self):
        frame = pd.DataFrame(
            [
                {
                    "SMILES": "CCO",
                    "Nesso_affinity_pred_value": 0.2,
                    "Nesso_affinity_probability_binary": 0.8,
                    "Nesso_entropy_pl": 0.4,
                }
            ]
        )
        records = activity.prediction_records(frame)
        self.assertEqual(
            records,
            [
                {
                    "SMILES": "CCO",
                    "Nesso_affinity_pred_value": 0.2,
                    "Nesso_affinity_probability_binary": 0.8,
                }
            ],
        )

    def test_reinvent_rand_str_resolution_matches_admet_prefixes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pool = Path(temp_dir)
            generated = pool / "sampling_LibInvent_token123.csv"
            generated.write_text("SMILES\nCCO\n", encoding="utf-8")
            with mock.patch.object(
                activity,
                "GENERATED_CSV_SEARCH_DIRS",
                (pool,),
            ):
                resolved = activity.resolve_generated_csv_path("token123")
            self.assertEqual(resolved, generated)

    def test_json_cache_imports_existing_sqlite_records_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_path = Path(temp_dir) / "nesso.sqlite"
            with sqlite3.connect(sqlite_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE nesso_predictions (
                        model_namespace TEXT NOT NULL,
                        protein_sha256 TEXT NOT NULL,
                        canonical_smiles TEXT NOT NULL,
                        affinity_pred_value REAL NOT NULL,
                        affinity_probability_binary REAL NOT NULL,
                        source_tool TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO nesso_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "test-model",
                        protein_sequence_hash("ACDE"),
                        "CCO",
                        1.25,
                        0.75,
                        "legacy",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )

            cache = NessoPredictionCache(sqlite_path, "test-model")
            found = cache.get_many("ACDE", ["CCO"])

            self.assertEqual(found["CCO"]["Nesso_affinity_pred_value"], 1.25)
            self.assertEqual(
                found["CCO"]["Nesso_affinity_probability_binary"], 0.75
            )
            self.assertTrue(sqlite_path.with_suffix(".json").is_file())


class ActivityServerTests(unittest.TestCase):
    def setUp(self):
        self.cache_temp = tempfile.TemporaryDirectory()
        self.original_cache = server._prediction_cache
        self.cache_path = Path(self.cache_temp.name) / "nesso.sqlite"
        server._prediction_cache = NessoPredictionCache(
            self.cache_path,
            "test-model",
        )

    def tearDown(self):
        server._prediction_cache = self.original_cache
        self.cache_temp.cleanup()

    @staticmethod
    def _fake_predictor_factory(destination: Path):
        def fake_predictor(**kwargs):
            frame = pd.read_csv(kwargs["ligand_csv"])
            probabilities = [0.1, 0.9][: len(frame)]
            frame["Nesso_affinity_pred_value"] = [0.2, 1.5][: len(frame)]
            frame["Nesso_affinity_probability_binary"] = probabilities
            frame["Nesso_entropy_pl"] = 0.5
            frame.to_csv(destination, index=False)
            return destination

        return fake_predictor

    def test_filter_reads_reinvent_csv_and_returns_top_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sampling_REINVENT4_token.csv"
            source.write_text(
                "SMILES,Scaffold,R-groups\nCCO,first,r1\nCCN,second,r2\n",
                encoding="utf-8",
            )
            result_csv = root / "nesso_result.csv"
            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(
                    server, "resolve_generated_csv_path", return_value=source
                ),
                mock.patch.object(server, "_random_suffix", return_value="fixed"),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=self._fake_predictor_factory(result_csv),
                ),
            ):
                output = json.loads(
                    server._filter_by_nesso_sync(
                        "ACDE",
                        "token",
                        "target",
                        "",
                        5,
                    )
                )

            self.assertEqual(output["candidates"][0], {
                "SMILES": "CCN",
                "Template": "second",
                "R-groups": "r2",
            })
            self.assertEqual(output["status"], "success")
            self.assertEqual(output["input_count"], 2)
            self.assertEqual(output["output_count"], 2)
            ranked_csv = root / "nesso_ranked_fixed.csv"
            self.assertTrue(ranked_csv.is_file())

    def test_filter_accepts_previous_stage_csv_and_top_n(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "admet_filtered.csv"
            pd.DataFrame(
                {
                    "SMILES": ["CCO", "CCN"],
                    "Scaffold": ["first", "second"],
                    "admet_score": [0.9, 0.8],
                }
            ).to_csv(source, index=False)
            result_csv = root / "nesso_result.csv"
            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(server, "_random_suffix", return_value="fixed"),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=self._fake_predictor_factory(result_csv),
                ),
            ):
                output = json.loads(
                    server._filter_by_nesso_sync(
                        "ACDE",
                        "",
                        "target",
                        str(source),
                        1,
                    )
                )

            written = pd.read_csv(output["output_csv_path"])

        self.assertEqual(output["input_csv_path"], str(source.resolve()))
        self.assertEqual(output["output_count"], 1)
        self.assertEqual(output["candidates"][0]["SMILES"], "CCN")
        self.assertIn("admet_score", written.columns)

    def test_predict_returns_both_nesso_activity_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_csv = root / "nesso_result.csv"
            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(server, "_random_suffix", return_value="fixed"),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=self._fake_predictor_factory(result_csv),
                ),
            ):
                output = ast.literal_eval(
                    server._predict_by_nesso_sync(
                        '["CCO", "CCN"]', "ACDE", "target"
                    )
                )

            self.assertEqual(len(output), 2)
            self.assertEqual(
                set(output[0]),
                {
                    "SMILES",
                    "Nesso_affinity_pred_value",
                    "Nesso_affinity_probability_binary",
                },
            )

    def test_filter_cache_is_reused_for_only_missing_critic_molecules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "generated.csv"
            pd.DataFrame({"SMILES": ["CCO", "CCN"]}).to_csv(source, index=False)
            calls = []

            def fake_predictor(**kwargs):
                batch = pd.read_csv(kwargs["ligand_csv"])
                calls.append(batch["SMILES"].tolist())
                batch["Nesso_affinity_pred_value"] = [
                    float(len(value)) for value in batch["SMILES"]
                ]
                batch["Nesso_affinity_probability_binary"] = 0.8
                result_csv = root / f"result_{len(calls)}.csv"
                batch.to_csv(result_csv, index=False)
                return result_csv

            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(server, "_random_suffix", return_value="fixed"),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=fake_predictor,
                ),
            ):
                server._filter_by_nesso_sync(
                    "ACDE",
                    "",
                    "target",
                    str(source),
                    2,
                )
                server._prediction_cache = NessoPredictionCache(
                    self.cache_path,
                    "test-model",
                )
                result = ast.literal_eval(
                    server._predict_by_nesso_sync(
                        '["CCO", "CCC"]',
                        "ACDE",
                        "target",
                    )
                )

            self.assertEqual(calls, [["CCO", "CCN"], ["CCC"]])
            self.assertEqual([record["SMILES"] for record in result], ["CCO", "CCC"])

    def test_nesso_cache_is_scoped_to_the_protein_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_csv = root / "result.csv"
            calls = []

            def fake_predictor(**kwargs):
                calls.append(kwargs["protein_sequence"])
                batch = pd.read_csv(kwargs["ligand_csv"])
                batch["Nesso_affinity_pred_value"] = 1.0
                batch["Nesso_affinity_probability_binary"] = 0.8
                batch.to_csv(result_csv, index=False)
                return result_csv

            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(server, "_random_suffix", return_value="fixed"),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=fake_predictor,
                ),
            ):
                server._predict_by_nesso_sync('["CCO"]', "ACDE", "target")
                server._predict_by_nesso_sync('["CCO"]', "ACDF", "target")

            self.assertEqual(calls, ["ACDE", "ACDF"])

    def test_prediction_batches_retry_after_worker_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCF", "CCI"]})
            calls = 0

            def flaky_predictor(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("worker exited")
                batch = pd.read_csv(kwargs["ligand_csv"])
                batch["Nesso_affinity_pred_value"] = 1.0
                batch["Nesso_affinity_probability_binary"] = 0.8
                result_csv = root / f"result_{calls}.csv"
                batch.to_csv(result_csv, index=False)
                return result_csv

            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=flaky_predictor,
                ),
                mock.patch.object(server, "_available_gpu_ids", return_value=["0"]),
                mock.patch.dict(
                    os.environ,
                    {
                        "NESSO_MCP_BATCH_SIZE": "2",
                        "NESSO_MCP_BATCH_MAX_ATTEMPTS": "2",
                    },
                ),
            ):
                scored = server._predict_frame_in_batches(
                    frame,
                    protein_sequence="ACDE",
                    protein_id="target",
                    job_name="test",
                    suffix="fixed",
                )

        self.assertEqual(scored["SMILES"].tolist(), frame["SMILES"].tolist())
        self.assertEqual(calls, 4)

    def test_prediction_batches_use_each_available_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = pd.DataFrame(
                {"SMILES": ["CCO", "CCN", "CCC", "CCF", "CCI", "CCCl"]}
            )
            observed_gpu_ids = []
            active = 0
            max_active = 0
            state_lock = threading.Lock()

            def fake_predictor(**kwargs):
                nonlocal active, max_active
                gpu_id = kwargs["gpu_id"]
                with state_lock:
                    observed_gpu_ids.append(gpu_id)
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                batch = pd.read_csv(kwargs["ligand_csv"])
                batch["Nesso_affinity_pred_value"] = 1.0
                batch["Nesso_affinity_probability_binary"] = 0.8
                result_csv = root / f"result_gpu_{gpu_id}.csv"
                batch.to_csv(result_csv, index=False)
                with state_lock:
                    active -= 1
                return result_csv

            with (
                mock.patch.object(server, "TMP_DIR", root),
                mock.patch.object(
                    server,
                    "_predict_csv_with_nesso",
                    side_effect=fake_predictor,
                ),
                mock.patch.object(
                    server,
                    "_available_gpu_ids",
                    return_value=["0", "1", "2"],
                ),
                mock.patch.dict(os.environ, {"NESSO_MCP_BATCH_SIZE": "2"}),
            ):
                scored = server._predict_frame_in_batches(
                    frame,
                    protein_sequence="ACDE",
                    protein_id="target",
                    job_name="parallel-test",
                    suffix="fixed",
                )

        self.assertEqual(scored["SMILES"].tolist(), frame["SMILES"].tolist())
        self.assertEqual(set(observed_gpu_ids), {"0", "1", "2"})
        self.assertEqual(max_active, 3)

    def test_registered_tools_accept_fasta_paths(self):
        async def list_registered_tools():
            tools = await server.app.list_tools()
            return {tool.name: tool for tool in tools}

        registered_tools = asyncio.run(list_registered_tools())
        self.assertIn("protein_fasta_path", registered_tools["nesso_filter_by_cofolding"].inputSchema["properties"])
        self.assertIn("protein_fasta_path", registered_tools["nesso_predict_by_cofolding"].inputSchema["properties"])

    def test_protein_fasta_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = Path(temp_dir) / "protein.fasta"
            fasta.write_text(">target\nACDE\nFG\n", encoding="utf-8")
            self.assertEqual(server._resolve_protein_input("", str(fasta)), "ACDEFG")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                server._resolve_protein_input("ACDE", str(fasta))


class ActivitySchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_activity_calls_are_offloaded_and_serialized(self):
        original_semaphore = server._run_semaphore
        server._run_semaphore = asyncio.Semaphore(1)
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def blocking_call(value):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return value

        try:
            results = await asyncio.gather(
                server._run_blocking_nesso_call("test", blocking_call, "one"),
                server._run_blocking_nesso_call("test", blocking_call, "two"),
            )
        finally:
            server._run_semaphore = original_semaphore

        self.assertEqual(results, ["one", "two"])
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
