from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem

import tamgen_context as context


PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       1.500   1.000   0.000  1.00  0.00           C  
ATOM      4  N   GLY A   2      15.000   0.000   0.000  1.00  0.00           N  
ATOM      5  CA  GLY A   2      16.000   0.000   0.000  1.00  0.00           C  
ATOM      6  C   GLY A   2      16.500   1.000   0.000  1.00  0.00           C  
HETATM    7  C1  LIG L   1       2.000   0.000   0.000  1.00  0.00           C  
HETATM    8  O1  LIG L   1       2.500   0.000   0.000  1.00  0.00           O  
TER
END
"""


def test_extract_ligand_radius_pocket(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PDB_TEXT, encoding="utf-8")
    data, metadata = context.extract_pocket(
        pdb,
        pocket_residues=[],
        pocket_radius=4.0,
        ligand_resname="LIG",
        ligand_chain="L",
        ligand_residue_number=1,
    )
    assert metadata["pocket_method"] == "ligand_radius"
    assert metadata["pocket_residue_count"] == 1
    assert metadata["pocket_residues"] == [["A", 1]]
    assert data["sequences"]["A"] == "AG"
    assert np.array_equal(data["near_center_masks"]["A"], np.asarray([1, 0]))


def test_prepare_context_reads_generic_task_yaml(tmp_path: Path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "complex.pdb").write_text(PDB_TEXT, encoding="utf-8")
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(task_dir / "reference.sdf"))
    writer.write(mol)
    writer.close()
    (task_dir / "task.yaml").write_text(
        "input_files:\n"
        "  - complex.pdb\n"
        "  - reference.sdf\n"
        "target_ligand:\n"
        "  resname: LIG\n"
        "  chain: L\n"
        "  residue_number: 1\n"
        "pocket_residues:\n"
        "  - [A, 1]\n"
        "hard_constraints:\n"
        "  - property: tanimoto_similarity\n"
        "    threshold: '>=0.7'\n",
        encoding="utf-8",
    )
    contexts = tmp_path / "contexts"
    monkeypatch.setattr(context, "CONTEXT_DIR", contexts)
    monkeypatch.setattr(context, "LOCK_DIR", tmp_path / "locks")

    def fake_dataset(_target, dataset_dir, subset, seed_variants, conditional):
        dataset_dir.mkdir(parents=True)
        (dataset_dir / f"{subset}.ready").write_text(
            json.dumps({"seed_variants": seed_variants, "conditional": conditional}),
            encoding="utf-8",
        )

    monkeypatch.setattr(context, "_build_fairseq_dataset", fake_dataset)
    result = context.prepare_context(task_yaml_path=str(task_dir / "task.yaml"))
    assert result["conditional"] is True
    assert result["task_similarity_threshold"] == 0.7
    assert result["pocket_residues"] == [["A", 1]]
    assert result["seed_variant_count"] >= 1
    assert Path(result["context_path"], "dataset", "test.ready").is_file()


def test_compact_pocket_residue_parser():
    assert context._parse_pocket_residues("A:2,A:3;B:7", {}) == [
        ("A", 2),
        ("A", 3),
        ("B", 7),
    ]


def test_scaffold_condition_keeps_seed_as_similarity_reference(tmp_path: Path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "complex.pdb").write_text(PDB_TEXT, encoding="utf-8")
    captured = {}

    def fake_dataset(_target, dataset_dir, subset, seed_variants, conditional):
        captured["seed_variants"] = seed_variants
        captured["conditional"] = conditional
        dataset_dir.mkdir(parents=True)
        (dataset_dir / f"{subset}.ready").write_text("ready", encoding="utf-8")

    monkeypatch.setattr(context, "CONTEXT_DIR", tmp_path / "contexts")
    monkeypatch.setattr(context, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(context, "_build_fairseq_dataset", fake_dataset)
    result = context.prepare_context(
        complex_pdb_path=str(task_dir / "complex.pdb"),
        seed_smiles="CCOc1ccccc1",
        scaffold_smiles="c1ccccc1",
        pocket_residues="A:1",
        seed_augmentations=2,
    )
    assert result["condition_type"] == "scaffold"
    assert result["seed_smiles"] == "CCOc1ccccc1"
    assert result["conditioning_smiles"] == "c1ccccc1"
    assert result["scaffold_smiles"] == "c1ccccc1"
    assert captured["conditional"] is True
    assert all(Chem.MolFromSmiles(value).GetNumAtoms() == 6 for value in captured["seed_variants"])
