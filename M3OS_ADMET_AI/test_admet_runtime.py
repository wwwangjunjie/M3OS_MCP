#!/usr/bin/env python3
"""Unit tests for model reuse and bounded non-blocking MCP execution."""

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import admet_prediction_admetai as prediction
import mcp_server_improved_admet_ai as server
from admet_prediction_cache import AdmetPredictionCache
from leadopt_constraints import assess_candidate_frame, parse_task_contract


class FakeModel:
    device = "cpu"
    num_workers = 0
    fingerprint_multiprocessing_min = 100

    def predict(self, smiles):
        return smiles


class AdmetModelReuseTests(unittest.TestCase):
    def setUp(self):
        self.original_model = prediction._admet_model
        prediction._admet_model = None

    def tearDown(self):
        prediction._admet_model = self.original_model

    def test_model_is_built_once(self):
        fake_model = FakeModel()
        with patch.object(prediction, "build_admet_model", return_value=fake_model) as build:
            self.assertIs(prediction.get_admet_model(), fake_model)
            self.assertIs(prediction.get_admet_model(), fake_model)
        build.assert_called_once_with()

    def test_shared_model_prediction_is_serialized(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        class SlowModel(FakeModel):
            def predict(self, smiles):
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                with state_lock:
                    active -= 1
                return smiles

        prediction._admet_model = SlowModel()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(prediction.predict_with_admet_model, [["C"], ["CC"]])
            )
        self.assertEqual(results, [["C"], ["CC"]])
        self.assertEqual(max_active, 1)


class McpSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_work_is_offloaded_and_bounded(self):
        original_semaphore = server._admet_request_semaphore
        server._admet_request_semaphore = asyncio.Semaphore(1)
        active = 0
        max_active = 0
        ticker_ran = False
        state_lock = threading.Lock()

        def blocking_call(value):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1
            return value

        async def ticker():
            nonlocal ticker_ran
            await asyncio.sleep(0.01)
            ticker_ran = True

        try:
            results = await asyncio.gather(
                server._run_blocking_admet_call("test", blocking_call, "one"),
                server._run_blocking_admet_call("test", blocking_call, "two"),
                ticker(),
            )
        finally:
            server._admet_request_semaphore = original_semaphore

        self.assertEqual(results[:2], ["one", "two"])
        self.assertTrue(ticker_ran)
        self.assertEqual(max_active, 1)


class CandidateCsvFilteringTests(unittest.TestCase):
    def test_runtime_hard_constraints_use_contract_values(self):
        contract = parse_task_contract(
            json.dumps(
                {
                    "reference_smiles": "CCO",
                    "hard_constraints": [
                        {
                            "property": "molecular_weight",
                            "source": "rdkit",
                            "operator": "lt",
                            "value": 50,
                        },
                        {
                            "property": "logp",
                            "source": "rdkit",
                            "operator": "between",
                            "min": -1,
                            "max": 1,
                        },
                        {
                            "property": "hERG",
                            "source": "admet",
                            "endpoint": "hERG",
                            "operator": "lte",
                            "value": 0.25,
                        },
                        {
                            "property": "interaction_probability",
                            "source": "activity",
                            "operator": "gt",
                            "value": 0.7,
                        },
                    ],
                }
            )
        )
        assessed = assess_candidate_frame(
            pd.DataFrame({"SMILES": ["CCO", "CCCC"]}),
            pd.DataFrame({"SMILES": ["CCO", "CCCC"], "hERG": [0.2, 0.1]}),
            contract,
            reference_smiles="CCO",
        )

        self.assertEqual(assessed["leadopt_constraints_pass"].tolist(), [True, False])
        self.assertIn("interaction_probability", assessed.iloc[0]["deferred_activity_constraints"])

    def test_unknown_hard_constraint_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer a source"):
            parse_task_contract(
                json.dumps(
                    {
                        "hard_constraints": [
                            {"property": "unknown_metric", "operator": "gte", "value": 1}
                        ]
                    }
                )
            )

    def test_filter_accepts_candidate_csv_and_retains_requested_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "candidates.csv"
            smiles = ["C" * length for length in range(1, 61)]
            pd.DataFrame(
                {
                    "SMILES": smiles,
                    "Scaffold": [f"scaffold-{index}" for index in range(60)],
                    "Nesso_affinity_probability_binary": [0.5] * 60,
                }
            ).to_csv(source, index=False)
            props = pd.DataFrame(
                {
                    "SMILES": smiles,
                    "score": list(range(60)),
                }
            )

            with (
                patch.object(server, "TMP_DIR", root),
                patch.object(prediction, "TMP_DIR", root),
                patch.object(
                    server,
                    "_prediction_cache",
                    AdmetPredictionCache(root / "cache.sqlite", "test-model"),
                ),
                patch.object(server, "generate_props_for_smiles", return_value=props),
                patch.object(server.secrets, "choice", return_value="x"),
            ):
                payload = json.loads(
                    server._filter_by_admetai_sync(
                        '{"score": "higher"}',
                        "",
                        str(source),
                        50,
                    )
                )

            output = pd.read_csv(payload["output_csv_path"])

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["input_count"], 60)
        self.assertEqual(payload["output_count"], 50)
        self.assertEqual(len(payload["candidates"]), 50)
        self.assertEqual(len(output), 50)
        self.assertIn("Scaffold", output.columns)
        self.assertIn("Nesso_affinity_probability_binary", output.columns)
        self.assertIn("score", output.columns)

    def test_filter_cache_is_reused_for_only_missing_critic_molecules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "candidates.csv"
            pd.DataFrame({"SMILES": ["CCO", "CCN"]}).to_csv(source, index=False)
            calls = []

            def fake_predict(smiles):
                calls.append(list(smiles))
                return pd.DataFrame(
                    {
                        "SMILES": smiles,
                        "score": [float(len(value)) for value in smiles],
                    }
                )

            cache_path = root / "cache.sqlite"
            with (
                patch.object(server, "TMP_DIR", root),
                patch.object(prediction, "TMP_DIR", root),
                patch.object(
                    server,
                    "_prediction_cache",
                    AdmetPredictionCache(cache_path, "test-model"),
                ),
                patch.object(server, "generate_props_for_smiles", side_effect=fake_predict),
                patch.object(server.secrets, "choice", return_value="x"),
            ):
                server._filter_by_admetai_sync(
                    '{"score": "higher"}',
                    "",
                    str(source),
                    2,
                )
                result = ast.literal_eval(
                    server._predict_by_admetai_sync(
                        '["OCC", "CCC"]',
                        '{"score": "higher"}',
                    )
                )

            self.assertEqual(calls, [["CCO", "CCN"], ["CCC"]])
            self.assertEqual([record["SMILES"] for record in result], ["CCO", "CCC"])

            with (
                patch.object(
                    server,
                    "_prediction_cache",
                    AdmetPredictionCache(cache_path, "test-model"),
                ),
                patch.object(server, "generate_props_for_smiles") as predict_again,
            ):
                persisted = ast.literal_eval(
                    server._predict_by_admetai_sync(
                        '["CCO"]',
                        '{"score": "higher"}',
                    )
                )
            predict_again.assert_not_called()
            self.assertEqual(persisted[0]["score"], 3.0)

    def test_task_constraints_apply_objectives_holds_and_hard_gates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "candidates.csv"
            pd.DataFrame({"SMILES": ["CCO", "CCC"]}).to_csv(source, index=False)
            predictions = pd.DataFrame(
                {
                    "SMILES": ["CCO", "CCC"],
                    "hERG": [0.3, 0.45],
                    "Solubility_AqSolDB": [-1.0, -2.0],
                }
            )
            contract = {
                "reference_smiles": "CCO",
                "baseline_values": {"herg": 0.5, "solubility": -1.0},
                "optimization_objectives": [
                    {
                        "property": "herg",
                        "direction": "minimize",
                        "threshold": 0.1,
                    }
                ],
                "hold_constant": [
                    {
                        "property": "solubility",
                        "direction": "decrease",
                        "tolerance": 0.5,
                    }
                ],
                "hard_constraints": [{"property": "smiles_validity"}],
            }
            with (
                patch.object(server, "TMP_DIR", root),
                patch.object(server, "_predict_admet_frame_cached", return_value=predictions),
                patch.object(server.secrets, "choice", return_value="x"),
            ):
                payload = json.loads(
                    server._filter_by_task_constraints_sync(
                        json.dumps(contract),
                        "CCO",
                        "",
                        str(source),
                    )
                )
            eligible = pd.read_csv(payload["output_csv_path"])
            scored = pd.read_csv(payload["scored_csv_path"])

        self.assertEqual(payload["input_count"], 2)
        self.assertEqual(payload["output_count"], 1)
        self.assertEqual(eligible["SMILES"].tolist(), ["CCO"])
        self.assertEqual(scored["leadopt_constraints_pass"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
