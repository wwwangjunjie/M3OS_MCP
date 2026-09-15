#!/usr/bin/env python3
"""Lightweight tests for the decoupled Boltz and PLIP MCP tools."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from m3os.tools import generation, interaction
from m3os.tools.generation import _parse_complex_input
from m3os.tools.interaction import analyze_complex_interactions
from mcp_server_improved import _run_blocking_tool, app


class SplitToolsTest(unittest.TestCase):
    def test_boltz_generation_does_not_invoke_plip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            prediction_dir = (
                output_dir / "boltz_results_test_case" / "predictions" / "test_case"
            )

            def fake_boltz_run(command, **kwargs):
                prediction_dir.mkdir(parents=True)
                (prediction_dir / "test_case_model_0.pdb").write_text(
                    "MODEL        1\nENDMDL\n",
                    encoding="utf-8",
                )
                (prediction_dir / "affinity_test_case.json").write_text(
                    json.dumps({"affinity_pred_value": 5.0}),
                    encoding="utf-8",
                )
                self.assertIn("predict", command)
                self.assertNotIn("cal_interaction.py", " ".join(command))
                self.assertEqual(kwargs["timeout"], 12.0)
                return SimpleNamespace(stdout="mock Boltz output")

            with (
                patch.object(generation, "BOLTZ_OUTPUT_DIR", str(output_dir)),
                patch.object(generation, "BOLTZ_CACHE_DIR", str(root / "cache")),
                patch.object(generation, "_safe_output_name", return_value="test_case"),
                patch.object(generation.torch.cuda, "is_available", return_value=False),
                patch.object(
                    generation.subprocess,
                    "run",
                    side_effect=fake_boltz_run,
                ) as run_mock,
            ):
                result = generation.generate_complex_structure(
                    {"sequence": "ACDE", "smiles": "CCO"},
                    12.0,
                )

            self.assertEqual(run_mock.call_count, 1)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["pred_pIC50"], 1.364)
            self.assertEqual(
                Path(result["structure_path"]).name,
                "test_case_model_0.pdb",
            )

    def test_plip_subprocess_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            structure = root / "complex.pdb"
            structure.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
            with (
                patch.object(interaction, "BOLTZ_OUTPUT_DIR", str(root / "output")),
                patch.object(
                    interaction.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("plip", 3.0),
                ),
            ):
                result = interaction.analyze_complex_interactions(
                    str(structure),
                    3.0,
                )
            self.assertEqual(result, "PLIP command timed out after 3.0 seconds.")

    def test_input_validation(self) -> None:
        self.assertTrue(_parse_complex_input("not-json").startswith("Error:"))
        self.assertTrue(analyze_complex_interactions("").startswith("Error:"))
        self.assertTrue(
            analyze_complex_interactions("/does/not/exist.pdb").startswith("Error:")
        )

    def test_registered_tools(self) -> None:
        async def list_registered_tools():
            tools = await app.list_tools()
            return {tool.name: tool for tool in tools}

        registered_tools = asyncio.run(list_registered_tools())
        self.assertEqual(
            set(registered_tools),
            {
                "generate_protein_ligand_complex",
                "analyze_protein_ligand_interactions",
            },
        )
        self.assertIn(
            "must not be used for activity scoring",
            registered_tools["generate_protein_ligand_complex"].description,
        )
        self.assertIn(
            "textual structural evidence",
            registered_tools["analyze_protein_ligand_interactions"].description,
        )
        self.assertIn(
            "candidate-ranking scores",
            registered_tools["analyze_protein_ligand_interactions"].description,
        )

    def test_json_serialization_contract(self) -> None:
        payload = {
            "status": "success",
            "structure_path": "/tmp/example.pdb",
        }
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_bounded_background_concurrency(self) -> None:
        async def measure_peak(limit: int) -> tuple[int, list[int]]:
            semaphore = asyncio.Semaphore(limit)
            state_lock = threading.Lock()
            active = 0
            peak = 0

            def work(value: int) -> int:
                nonlocal active, peak
                with state_lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with state_lock:
                    active -= 1
                return value

            results = await asyncio.gather(
                *(
                    _run_blocking_tool(
                        "test_work",
                        work,
                        value,
                        semaphore=semaphore,
                        queue_timeout_seconds=1.0,
                    )
                    for value in range(4)
                )
            )
            return peak, results

        serial_peak, serial_results = asyncio.run(measure_peak(1))
        parallel_peak, parallel_results = asyncio.run(measure_peak(3))
        self.assertEqual(serial_peak, 1)
        self.assertEqual(parallel_peak, 3)
        self.assertEqual(serial_results, list(range(4)))
        self.assertEqual(parallel_results, list(range(4)))

    def test_queue_timeout(self) -> None:
        async def run_timeout():
            semaphore = asyncio.Semaphore(1)
            await semaphore.acquire()
            try:
                return await _run_blocking_tool(
                    "queued_test",
                    lambda: "not run",
                    semaphore=semaphore,
                    queue_timeout_seconds=0.01,
                )
            finally:
                semaphore.release()

        result = asyncio.run(run_timeout())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "QueueTimeout")


if __name__ == "__main__":
    unittest.main()
