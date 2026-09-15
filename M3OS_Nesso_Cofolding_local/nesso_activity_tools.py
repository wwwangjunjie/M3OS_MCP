"""ADMET-AI-style CSV adapters for local Nesso activity scoring."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rdkit import Chem
except ModuleNotFoundError:
    Chem = None

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
REINVENT_ROOT = Path(
    os.environ.get("M3OS_REINVENT_ROOT", PROJECT_ROOT / "M3OS_REINVENT")
).expanduser().resolve()
REINVENT_TMP_POOL = REINVENT_ROOT / "tmp" / "pool"
REINVENT_POOL = REINVENT_ROOT / "pool"
M3OS_POOL = PROJECT_ROOT / "m3os" / "pool"
TMP_DIR = PROJECT_DIR / "tmp"

GENERATED_CSV_PREFIXES = (
    "sampling_REINVENT4",
    "sampling_LibInvent",
    "sampling_Mol2Mol",
    "sampling_LinkInvent",
    "sampling_Reinvent",
)
GENERATED_CSV_SEARCH_DIRS = (
    REINVENT_TMP_POOL,
    REINVENT_POOL,
    M3OS_POOL,
)
ACTIVITY_COLUMNS = (
    "Nesso_affinity_pred_value",
    "Nesso_affinity_probability_binary",
)


def canonicalize_smiles(smiles: Any) -> str | None:
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    if Chem is None:
        return normalized
    try:
        molecule = Chem.MolFromSmiles(normalized)
        if molecule is None:
            return None
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def candidate_generated_csv_paths(rand_str: str) -> list[Path]:
    if not rand_str:
        return []
    return [
        pool_dir / f"{prefix}_{rand_str}.csv"
        for pool_dir in GENERATED_CSV_SEARCH_DIRS
        for prefix in GENERATED_CSV_PREFIXES
    ]


def resolve_generated_csv_path(rand_str: str) -> Path | None:
    for candidate in candidate_generated_csv_paths(rand_str):
        if candidate.is_file():
            return candidate
    return None


def format_missing_generated_csv_message(rand_str: str) -> str:
    candidates = candidate_generated_csv_paths(rand_str)
    if not candidates:
        return "Generated molecule CSV not found: provide rand_str."
    searched = ", ".join(str(path) for path in candidates)
    return (
        f"Generated molecule CSV not found for rand_str={rand_str!r}. "
        f"Searched: {searched}"
    )


def _pick_smiles_column(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    for exact in ("SMILES", "smiles"):
        if exact in columns:
            return exact
    lowered = {column.lower(): column for column in columns}
    for candidate in ("smi", "canonical_smiles", "ligand"):
        if candidate in lowered:
            return lowered[candidate]
    if columns:
        return columns[0]
    raise ValueError("Generated molecule CSV has no columns")


def normalize_generated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the SMILES column and deduplicate while preserving row order."""
    smiles_column = _pick_smiles_column(frame)
    normalized = frame.copy()
    smiles = normalized[smiles_column].astype("string").str.strip()
    normalized = normalized.loc[smiles.notna() & smiles.ne("")].copy()
    normalized[smiles_column] = smiles.loc[normalized.index]
    if smiles_column != "SMILES":
        normalized.insert(0, "SMILES", normalized[smiles_column])
    normalized = normalized.drop_duplicates(subset=["SMILES"], keep="first")
    return normalized.reset_index(drop=True)


def prepare_generated_csv(source_csv: Path, destination_csv: Path) -> pd.DataFrame:
    frame = normalize_generated_frame(pd.read_csv(source_csv))
    if frame.empty:
        raise ValueError(f"No non-empty SMILES found in {source_csv}")
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination_csv, index=False)
    return frame


def parse_smiles_list(smiles_list: str) -> list[str]:
    try:
        parsed = ast.literal_eval(smiles_list)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("smiles_list must be a Python/JSON list of SMILES") from exc
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("smiles_list must be a Python/JSON list of SMILES")
    frame = normalize_generated_frame(pd.DataFrame({"SMILES": parsed}))
    if frame.empty:
        raise ValueError("smiles_list contains no non-empty SMILES")
    return frame["SMILES"].astype(str).tolist()


def prepare_smiles_list_csv(smiles_list: str, destination_csv: Path) -> list[str]:
    parsed = parse_smiles_list(smiles_list)
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": parsed}).to_csv(destination_csv, index=False)
    return parsed


def read_activity_results(result_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(result_csv)
    smiles_column = _pick_smiles_column(frame)
    if smiles_column != "SMILES":
        frame.insert(0, "SMILES", frame[smiles_column])
    missing = [column for column in ACTIVITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Nesso result is missing columns: {missing}")
    for column in ACTIVITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prediction_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    valid = frame.dropna(subset=list(ACTIVITY_COLUMNS))
    return [
        {
            "SMILES": str(row["SMILES"]),
            "Nesso_affinity_pred_value": float(row["Nesso_affinity_pred_value"]),
            "Nesso_affinity_probability_binary": float(
                row["Nesso_affinity_probability_binary"]
            ),
        }
        for _, row in valid.iterrows()
    ]


def _clean_metadata_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _generation_metadata(row: pd.Series) -> tuple[str, str]:
    template = next(
        (
            _clean_metadata_value(row.get(column))
            for column in ("Scaffold", "Input_SMILES", "Warheads")
            if _clean_metadata_value(row.get(column))
        ),
        "",
    )
    r_groups = next(
        (
            _clean_metadata_value(row.get(column))
            for column in ("R-groups", "Linker")
            if _clean_metadata_value(row.get(column))
        ),
        "",
    )
    if not r_groups:
        tanimoto = _clean_metadata_value(row.get("Tanimoto"))
        if tanimoto:
            r_groups = f"Tanimoto={tanimoto}"
    return template, r_groups


def rank_by_binding_probability(
    frame: pd.DataFrame,
    *,
    output_csv: Path,
    top_n: int = 5,
) -> list[dict[str, str]]:
    """Sort by binder probability descending and write an ADMET-style top CSV."""
    valid = frame.dropna(subset=list(ACTIVITY_COLUMNS)).copy()
    if valid.empty:
        raise ValueError("Nesso produced no valid activity predictions")
    probability_column = "Nesso_affinity_probability_binary"
    rank_column = f"{probability_column}_rank"
    valid[rank_column] = valid[probability_column].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    valid["average_rank"] = valid[rank_column]
    valid = valid.sort_values(
        [probability_column], ascending=[False], kind="stable"
    ).head(top_n)

    source_columns = [
        column
        for column in valid.columns
        if not str(column).startswith("Nesso_") and column != "average_rank"
    ]
    output_columns = [
        *source_columns,
        *ACTIVITY_COLUMNS,
        rank_column,
        "average_rank",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid[output_columns].to_csv(output_csv, index=False)

    top_candidates = []
    for _, row in valid.iterrows():
        template, r_groups = _generation_metadata(row)
        top_candidates.append(
            {
                "SMILES": str(row["SMILES"]),
                "Template": template,
                "R-groups": r_groups,
            }
        )
    return top_candidates
