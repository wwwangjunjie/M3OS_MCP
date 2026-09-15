from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import mcp_server_tamgen as server


def test_generation_result_is_downstream_csv(tmp_path: Path, monkeypatch):
    context_dir = tmp_path / "context"
    dataset_dir = context_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    metadata = {
        "context_id": "abc123",
        "context_path": str(context_dir),
        "dataset_path": str(dataset_dir),
        "subset": "test",
        "conditional": True,
        "seed_smiles": "CCO",
        "task_similarity_threshold": 0.7,
    }
    (context_dir / "context.json").write_text(json.dumps(metadata), encoding="utf-8")

    class FakeRuntime:
        def generate(self, context, **kwargs):
            assert kwargs["min_seed_similarity"] == 0.7
            assert kwargs["require_scaffold_match"] is False
            return ([{
                "SMILES": "CCN",
                "TamGen_generation_score": -0.1,
                "TamGen_seed_similarity": 0.8,
                "TamGen_generation_count": 2,
                "TamGen_context_id": context["context_id"],
                "TamGen_mode": "conditional",
                "Input_SMILES": "CCO",
            }], {
                "generation_seconds": 0.1,
                "random_seeds_used": 1,
                "unique_valid_count": 1,
                "requested_count": 1,
                "returned_count": 1,
            })

    monkeypatch.setattr(server, "_get_runtime", lambda: FakeRuntime())
    output = tmp_path / "candidates.csv"
    result = json.loads(server._generate_sync(
        str(context_dir), 1, 2, 1.0, 1, 1, 0.0, True, False, str(output)
    ))
    assert result["status"] == "success"
    assert result["output_count"] == 1
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["SMILES"] == "CCN"


def test_prepare_tool_returns_json(monkeypatch):
    monkeypatch.setattr(
        server,
        "prepare_context",
        lambda **_kwargs: {"status": "success", "context_path": "/tmp/context"},
    )
    result = json.loads(asyncio.run(server.tamgen_prepare_target_context(seed_smiles="CCO")))
    assert result["status"] == "success"
