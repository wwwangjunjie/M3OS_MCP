#!/usr/bin/env python3
"""Tests for structured REINVENT generation results."""

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mcp_server_improved as server
from m3os.tools import generation


class GenerationResultTests(unittest.TestCase):
    def test_similarity_generation_defaults_to_ten_thousand_samples(self):
        parameters = inspect.signature(
            server.generate_similarity_constrained_mol2mol
        ).parameters

        self.assertEqual(parameters["num_candidates"].default, 10000)
        self.assertEqual(parameters["samples_per_round"].default, 1000)
        self.assertEqual(parameters["max_rounds"].default, 10)

    def test_run_generation_returns_exact_csv_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "jobtoken"
            job_dir.mkdir()
            output_csv = root / "sampling_LibInvent_jobtoken.csv"
            output_csv.write_text("SMILES\nCCO\nCCN\n", encoding="utf-8")
            config_file = job_dir / "LibInvent.toml"
            config_file.write_text(
                "[parameters]\n"
                f'output_file = "{output_csv}"\n',
                encoding="utf-8",
            )

            with mock.patch.object(
                server,
                "run_reinvent",
                return_value="REINVENT execution completed successfully.",
            ):
                result = json.loads(asyncio.run(server.run_generation(str(config_file))))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rand_str"], "jobtoken")
        self.assertEqual(result["generated_csv_path"], str(output_csv.resolve()))
        self.assertEqual(result["molecule_count"], 2)

    def test_mol2mol_config_supports_random_high_similarity_sampling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                mock.patch.object(generation, "REINVENT_JOB_TMP_PATH", str(root / "jobs")),
                mock.patch.object(generation, "POOL_PATH", str(root / "pool")),
                mock.patch.object(generation, "REINVENT_PATH", str(root / "reinvent")),
            ):
                config_file = generation.update_reinvent_config(
                    model_type="Mol2Mol",
                    device="cuda:0",
                    num_samples=500,
                    rand_str="testjob",
                    mol2mol_model="high_similarity",
                    sample_strategy="multinomial",
                    temperature=0.8,
                    random_seed=17,
                )
                config = Path(config_file).read_text(encoding="utf-8")

        self.assertIn("mol2mol_high_similarity.prior", config)
        self.assertIn('sample_strategy = "multinomial"', config)
        self.assertIn("temperature = 0.8", config)
        self.assertIn("seed = 17", config)

    def test_similarity_filter_uses_smdd_benchmark_fingerprint(self):
        seed_smiles = "Nc1ncnc2c1c(-c1ccc(Cl)c(O)c1)nn2[C@@H]1CCNC1"
        passing_smiles = "Nc1ncnc2c1c(-c1ccc(O)c(F)c1)nn2C1CCNC1"
        failing_smiles = "CCO"
        seed_canonical, seed_molecule = server._canonical_molecule(seed_smiles)
        generator = server.rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            includeChirality=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            generated_csv = Path(temp_dir) / "generated.csv"
            generated_csv.write_text(
                "SMILES,Input_SMILES,Tanimoto,NLL\n"
                f"{passing_smiles},{seed_smiles},0.8082,3.2\n"
                f"{failing_smiles},{seed_smiles},0.1,5.1\n"
                f"{seed_smiles},{seed_smiles},1.0,1.0\n",
                encoding="utf-8",
            )
            candidates, counts = server._collect_similarity_candidates(
                generated_csv,
                seed_fingerprint=generator.GetFingerprint(seed_molecule),
                seed_canonical=seed_canonical,
                similarity_threshold=0.7,
                seen=set(),
                model_variant="high_similarity",
                sampling_strategy="multinomial",
                round_index=1,
                exclude_seed=True,
            )

        self.assertEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0]["SMDD_Tanimoto"], 0.7)
        self.assertEqual(counts["threshold_passes"], 2)
        self.assertEqual(counts["seed_matches"], 1)


if __name__ == "__main__":
    unittest.main()
