#!/usr/bin/env python3
"""Build reusable TamGen inputs from protein-ligand complex structures."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
WORK_DIR = Path(os.environ.get("TAMGEN_WORK_DIR", ROOT / "workspaces")).resolve()
CONTEXT_DIR = WORK_DIR / "contexts"
LOCK_DIR = WORK_DIR / "locks"

WATER_NAMES = {"DOD", "HOH", "WAT"}
COMMON_IONS = {
    "CA", "CD", "CL", "CO", "CU", "FE", "K", "LI", "MG", "MN", "NA",
    "NI", "SR", "ZN",
}


def _absolute_file(value: str, label: str, suffixes: set[str] | None = None) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if suffixes is not None and path.suffix.lower() not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(f"{label} must use one of these extensions: {allowed}")
    return path


def _task_data(task_yaml_path: str) -> tuple[Path | None, dict[str, Any]]:
    if not str(task_yaml_path or "").strip():
        return None, {}
    path = _absolute_file(task_yaml_path, "task_yaml_path", {".yaml", ".yml"})
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("task_yaml_path must contain a YAML mapping")
    return path, value


def _declared_input(task_path: Path, task: dict[str, Any], suffixes: set[str]) -> Path | None:
    values = task.get("input_files", [])
    if not isinstance(values, list):
        return None
    for value in values:
        candidate = (task_path.parent / str(value)).resolve()
        if candidate.suffix.lower() in suffixes and candidate.is_file():
            return candidate
    return None


def resolve_structure_inputs(
    complex_pdb_path: str,
    reference_ligand_path: str,
    task_yaml_path: str,
) -> tuple[Path, Path | None, Path | None, dict[str, Any]]:
    task_path, task = _task_data(task_yaml_path)
    pdb_value = str(complex_pdb_path or "").strip()
    ligand_value = str(reference_ligand_path or "").strip()
    if not pdb_value and task_path is not None:
        declared = _declared_input(task_path, task, {".pdb"})
        pdb_value = str(declared or "")
    if not ligand_value and task_path is not None:
        declared = _declared_input(task_path, task, {".sdf", ".mol", ".mol2"})
        ligand_value = str(declared or "")
    if not pdb_value:
        raise ValueError("complex_pdb_path is required unless task_yaml_path declares a PDB input")
    pdb_path = _absolute_file(pdb_value, "complex_pdb_path", {".pdb"})
    ligand_path = (
        _absolute_file(ligand_value, "reference_ligand_path", {".sdf", ".mol", ".mol2"})
        if ligand_value
        else None
    )
    return pdb_path, ligand_path, task_path, task


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        raise ValueError("seed_smiles is not a valid molecule")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _smiles_from_structure(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    else:
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None:
        raise ValueError(f"No valid molecule found in reference_ligand_path: {path}")
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def resolve_seed_smiles(seed_smiles: str, reference_ligand_path: Path | None) -> str | None:
    direct = str(seed_smiles or "").strip()
    if direct and reference_ligand_path is not None:
        raise ValueError("Provide seed_smiles or reference_ligand_path, not both")
    if direct:
        return _canonical_smiles(direct)
    if reference_ligand_path is not None:
        return _smiles_from_structure(reference_ligand_path)
    return None


def resolve_scaffold_smiles(
    scaffold_smiles: str,
    seed_smiles: str | None,
) -> str | None:
    direct = str(scaffold_smiles or "").strip()
    if direct:
        return _canonical_smiles(direct)
    if not seed_smiles:
        return None
    seed = Chem.MolFromSmiles(seed_smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(seed)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        raise ValueError("No Bemis-Murcko scaffold could be derived from the seed molecule")
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)


def _parse_pocket_residues(value: str, task: dict[str, Any]) -> list[tuple[str, int]]:
    raw: Any = str(value or "").strip()
    if raw:
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pairs = []
            for token in re.split(r"[,;\s]+", str(raw)):
                if not token:
                    continue
                chain, separator, number = token.partition(":")
                if not separator:
                    raise ValueError("pocket_residues must be JSON pairs or tokens such as A:22,A:23")
                pairs.append((chain, int(number)))
            raw = pairs
    else:
        raw = task.get("pocket_residues", [])
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("pocket_residues must be a list")
    result: list[tuple[str, int]] = []
    for item in raw:
        if isinstance(item, dict):
            chain = item.get("chain")
            number = item.get("position", item.get("residue_number"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            chain, number = item
        else:
            raise ValueError(f"Invalid pocket residue: {item!r}")
        chain_value = str(chain or "").strip()
        if not chain_value:
            raise ValueError(f"Pocket residue has no chain: {item!r}")
        result.append((chain_value, int(number)))
    return list(dict.fromkeys(result))


def _target_ligand_spec(
    task: dict[str, Any],
    ligand_resname: str,
    ligand_chain: str,
    ligand_residue_number: int,
) -> tuple[str, str, int | None]:
    task_spec = task.get("target_ligand", {})
    if not isinstance(task_spec, dict):
        task_spec = {}
    name = str(ligand_resname or task_spec.get("resname") or "").strip().upper()
    chain = str(ligand_chain or task_spec.get("chain") or "").strip()
    number_value = ligand_residue_number or task_spec.get("residue_number")
    return name, chain, int(number_value) if number_value not in (None, "", 0) else None


def _atom_mass(element: str) -> float:
    symbol = str(element or "C").strip().capitalize()
    if symbol == "D":
        symbol = "H"
    try:
        return float(Chem.GetPeriodicTable().GetAtomicWeight(symbol))
    except Exception:
        return 12.011


def _heavy_atom_coordinates(residue) -> np.ndarray:
    values = [
        np.asarray(atom.coord, dtype=np.float32)
        for atom in residue.get_atoms()
        if str(atom.element or "").strip().upper() != "H"
    ]
    return np.asarray(values, dtype=np.float32)


def _residue_center(residue) -> np.ndarray:
    coordinates = []
    weights = []
    for atom in residue.get_atoms():
        coordinates.append(np.asarray(atom.coord, dtype=np.float32))
        weights.append(_atom_mass(atom.element))
    if not coordinates:
        return np.zeros(3, dtype=np.float32)
    return np.average(np.asarray(coordinates), axis=0, weights=np.asarray(weights)).astype(np.float32)


def _protein_letter(residue) -> str:
    name = residue.get_resname().strip().upper()
    table = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M",
    }
    return table.get(name, "X")


def _choose_ligand(model, name: str, chain_id: str, residue_number: int | None):
    candidates = []
    for chain in model:
        for residue in chain:
            hetflag, number, _insertion = residue.id
            resname = residue.get_resname().strip().upper()
            if hetflag == " " or is_aa(residue, standard=False):
                continue
            if resname in WATER_NAMES:
                continue
            coordinates = _heavy_atom_coordinates(residue)
            if coordinates.size == 0:
                continue
            candidates.append((chain.id, int(number), resname, residue, len(coordinates)))
    if name or chain_id or residue_number is not None:
        matches = [
            item for item in candidates
            if (not name or item[2] == name)
            and (not chain_id or item[0] == chain_id)
            and (residue_number is None or item[1] == residue_number)
        ]
        if not matches:
            raise ValueError(
                "Target ligand was not found in complex_pdb_path: "
                f"resname={name or '*'}, chain={chain_id or '*'}, residue_number={residue_number or '*'}"
            )
        return max(matches, key=lambda item: item[4])
    filtered = [item for item in candidates if item[2] not in COMMON_IONS]
    if not filtered:
        raise ValueError("No ligand-like HETATM residue found in complex_pdb_path")
    return max(filtered, key=lambda item: item[4])


def extract_pocket(
    pdb_path: Path,
    *,
    pocket_residues: list[tuple[str, int]],
    pocket_radius: float,
    ligand_resname: str,
    ligand_chain: str,
    ligand_residue_number: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pocket_radius <= 0:
        raise ValueError("pocket_radius must be positive")
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    ligand = None
    ligand_coordinates = None
    try:
        ligand = _choose_ligand(
            model, ligand_resname, ligand_chain, ligand_residue_number
        )
        ligand_coordinates = _heavy_atom_coordinates(ligand[3])
    except ValueError:
        if not pocket_residues:
            raise

    selected_set = set(pocket_residues)
    sequences: dict[str, str] = {}
    coordinates: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    selected_ids: dict[str, np.ndarray] = {}
    observed_pairs: set[tuple[str, int]] = set()

    for chain in model:
        residues = [
            residue for residue in chain
            if residue.id[0] == " " and is_aa(residue, standard=False)
        ]
        if not residues:
            continue
        chain_coordinates = np.asarray([_residue_center(residue) for residue in residues], dtype=np.float32)
        chain_mask = np.zeros(len(residues), dtype=np.int32)
        for index, residue in enumerate(residues):
            pair = (str(chain.id), int(residue.id[1]))
            observed_pairs.add(pair)
            if selected_set:
                selected = pair in selected_set
            else:
                atom_coordinates = _heavy_atom_coordinates(residue)
                selected = bool(
                    atom_coordinates.size
                    and ligand_coordinates is not None
                    and np.any(
                        np.linalg.norm(
                            atom_coordinates[:, None, :] - ligand_coordinates[None, :, :],
                            axis=2,
                        ) <= pocket_radius
                    )
                )
            chain_mask[index] = int(selected)
        sequences[str(chain.id)] = "".join(_protein_letter(residue) for residue in residues)
        coordinates[str(chain.id)] = chain_coordinates
        masks[str(chain.id)] = chain_mask
        selected_ids[str(chain.id)] = np.asarray(
            [int(residues[index].id[1]) for index in np.nonzero(chain_mask)[0]],
            dtype=np.int32,
        )

    if selected_set:
        missing = sorted(selected_set - observed_pairs)
        if missing:
            preview = ", ".join(f"{chain}:{number}" for chain, number in missing[:12])
            raise ValueError(f"Pocket residues not present in the PDB coordinates: {preview}")
    selected_count = int(sum(mask.sum() for mask in masks.values()))
    if selected_count == 0:
        raise ValueError("Pocket definition selected no protein residues")

    selected_centers = np.concatenate([
        coordinates[chain][np.nonzero(masks[chain])[0]]
        for chain in sorted(sequences)
        if np.any(masks[chain])
    ])
    if ligand_coordinates is not None:
        center = ligand_coordinates.mean(axis=0).astype(np.float32)
    else:
        center = selected_centers.mean(axis=0).astype(np.float32)

    data = {
        "index": 0,
        "pdb_id": pdb_path.stem.lower(),
        "sequences": sequences,
        "uniprot_refs": {},
        "coordinates": coordinates,
        "zero_masks": {
            chain: np.any(value != 0, axis=1).astype(np.int32)
            for chain, value in coordinates.items()
        },
        "near_center_masks": masks,
        "res_ids": selected_ids,
        "center": center,
        "all_chain_ids": sorted(sequences),
    }
    metadata = {
        "pocket_method": "explicit_residues" if selected_set else "ligand_radius",
        "pocket_radius": float(pocket_radius),
        "pocket_residue_count": selected_count,
        "pocket_residues": [
            [chain, int(number)]
            for chain in sorted(selected_ids)
            for number in selected_ids[chain].tolist()
        ],
        "pocket_center": [float(value) for value in center],
        "target_ligand": (
            {
                "chain": ligand[0],
                "residue_number": ligand[1],
                "resname": ligand[2],
                "heavy_atom_count": ligand[4],
            }
            if ligand is not None
            else None
        ),
    }
    return data, metadata


def augment_seed_smiles(smiles: str, count: int, random_seed: int = 1234) -> list[str]:
    if count < 1 or count > 100:
        raise ValueError("seed_augmentations must be between 1 and 100")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid seed molecule")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    result = [canonical]
    rng = random.Random(random_seed)
    indices = list(range(mol.GetNumAtoms()))
    attempts = 0
    while len(result) < count and attempts < count * 20:
        attempts += 1
        rng.shuffle(indices)
        randomized = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, indices),
            isomericSmiles=True,
            canonical=False,
        )
        if randomized not in result:
            result.append(randomized)
    return result


def _similarity_threshold(task: dict[str, Any]) -> float | None:
    constraints = task.get("hard_constraints", [])
    if not isinstance(constraints, list):
        return None
    for item in constraints:
        if not isinstance(item, dict):
            continue
        prop = str(item.get("property", "")).lower()
        if "similarity" not in prop:
            continue
        match = re.search(r"(?:>=|>)?\s*(-?\d+(?:\.\d+)?)", str(item.get("threshold", "")))
        if match:
            value = float(match.group(1))
            if 0 <= value <= 1:
                return value
    return None


def _context_digest(paths: Iterable[Path | None], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for path in paths:
        if path is None:
            continue
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()[:20]


@contextlib.contextmanager
def _context_lock(context_id: str):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{context_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_fairseq_dataset(
    target_data: dict[str, Any],
    dataset_dir: Path,
    subset: str,
    seed_variants: list[str],
    conditional: bool,
) -> None:
    from fairseq.molecule_utils.external.fairseq_dataset_build_utils import (
        dump_center_data,
        dump_center_data_ligand,
    )

    common = {
        "all_data": [target_data],
        "name": subset,
        "output_dir": dataset_dir,
        "fairseq_root": SOURCE_DIR,
        "pre_dicts_root": SOURCE_DIR / "dict",
        "max_len": 1023,
    }
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{original_path}"
    try:
        if conditional:
            dump_center_data_ligand(**common, ligand_list=seed_variants)
        else:
            dump_center_data(**common)
    finally:
        os.environ["PATH"] = original_path


def prepare_context(
    *,
    complex_pdb_path: str = "",
    reference_ligand_path: str = "",
    seed_smiles: str = "",
    task_yaml_path: str = "",
    pocket_residues: str = "",
    ligand_resname: str = "",
    ligand_chain: str = "",
    ligand_residue_number: int = 0,
    pocket_radius: float = 10.0,
    conditional: bool = True,
    seed_augmentations: int = 12,
    scaffold_smiles: str = "",
) -> dict[str, Any]:
    pdb_path, ligand_path, task_path, task = resolve_structure_inputs(
        complex_pdb_path, reference_ligand_path, task_yaml_path
    )
    if str(seed_smiles or "").strip() and not str(reference_ligand_path or "").strip():
        ligand_path = None
    resolved_seed = resolve_seed_smiles(seed_smiles, ligand_path)
    resolved_scaffold = (
        resolve_scaffold_smiles(scaffold_smiles, resolved_seed)
        if str(scaffold_smiles or "").strip()
        else None
    )
    conditioning_smiles = resolved_scaffold or resolved_seed
    if conditional and conditioning_smiles is None:
        raise ValueError(
            "Conditional generation requires seed_smiles or reference_ligand_path"
        )
    residues = _parse_pocket_residues(pocket_residues, task)
    target_name, target_chain, target_number = _target_ligand_spec(
        task,
        ligand_resname,
        ligand_chain,
        ligand_residue_number,
    )
    identity = {
        "seed_smiles": resolved_seed,
        "pocket_residues": residues,
        "ligand_resname": target_name,
        "ligand_chain": target_chain,
        "ligand_residue_number": target_number,
        "pocket_radius": float(pocket_radius),
        "conditional": bool(conditional),
        "seed_augmentations": int(seed_augmentations),
        "scaffold_smiles": resolved_scaffold,
    }
    context_id = _context_digest([pdb_path, ligand_path, task_path], identity)
    context_path = CONTEXT_DIR / context_id
    metadata_path = context_path / "context.json"

    with _context_lock(context_id):
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{context_id}-", dir=CONTEXT_DIR))
        try:
            target_data, pocket_metadata = extract_pocket(
                pdb_path,
                pocket_residues=residues,
                pocket_radius=float(pocket_radius),
                ligand_resname=target_name,
                ligand_chain=target_chain,
                ligand_residue_number=target_number,
            )
            seed_variants = (
                augment_seed_smiles(
                    str(conditioning_smiles), int(seed_augmentations),
                    random_seed=int(context_id[:8], 16),
                )
                if conditional
                else []
            )
            dataset_dir = temporary / "dataset"
            _build_fairseq_dataset(
                target_data, dataset_dir, "test", seed_variants, bool(conditional)
            )
            metadata = {
                "status": "success",
                "context_id": context_id,
                "context_path": str(context_path),
                "dataset_path": str(context_path / "dataset"),
                "subset": "test",
                "complex_pdb_path": str(pdb_path),
                "reference_ligand_path": str(ligand_path) if ligand_path else "",
                "task_yaml_path": str(task_path) if task_path else "",
                "conditional": bool(conditional),
                "seed_smiles": resolved_seed or "",
                "conditioning_smiles": conditioning_smiles or "",
                "condition_type": "scaffold" if resolved_scaffold else (
                    "seed" if conditional else "none"
                ),
                "scaffold_smiles": resolved_scaffold or "",
                "seed_variant_count": len(seed_variants),
                "task_similarity_threshold": _similarity_threshold(task),
                **pocket_metadata,
            }
            (temporary / "context.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, context_path)
            return metadata
        except Exception:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise


def load_context(context_path: str) -> dict[str, Any]:
    path = Path(str(context_path or "").strip()).expanduser().resolve()
    metadata = path / "context.json" if path.is_dir() else path
    if not metadata.is_file():
        raise FileNotFoundError(f"TamGen context not found: {metadata}")
    value = json.loads(metadata.read_text(encoding="utf-8"))
    dataset_path = Path(value["dataset_path"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"TamGen dataset not found: {dataset_path}")
    return value
