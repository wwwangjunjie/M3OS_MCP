from admet_ai import ADMETModel
import logging
import numpy as np
import pandas as pd
import os
import random
import json
import threading
import time
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


seed = 0
random.seed(seed)
np.random.seed(seed)


PROJECT_DIR = Path(__file__).resolve().parent
TMP_DIR = PROJECT_DIR / "tmp"

logger = logging.getLogger(__name__)

_admet_model: ADMETModel | None = None
_admet_model_init_lock = threading.Lock()
_admet_model_predict_lock = threading.Lock()


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "" or value.strip().lower() == "auto":
        return default
    return int(value)


def build_admet_model() -> ADMETModel:
    # DataLoader subprocesses are fragile when prediction is dispatched from
    # the MCP worker thread. Keep them disabled by default; deployments can
    # explicitly opt in after benchmarking their platform.
    num_workers = _get_int_env("ADMET_NUM_WORKERS", 0)
    fingerprint_min = _get_int_env("ADMET_FINGERPRINT_THREAD_MIN", 100)
    model = ADMETModel(
        num_workers=num_workers,
        fingerprint_multiprocessing_min=fingerprint_min,
    )
    logger.info(
        "ADMETModel initialized "
        f"device={model.device} num_workers={model.num_workers} "
        f"fingerprint_thread_min={model.fingerprint_multiprocessing_min}"
    )
    return model


def get_admet_model() -> ADMETModel:
    """Return the process-wide model, loading it only on the first request."""
    global _admet_model
    if _admet_model is None:
        with _admet_model_init_lock:
            if _admet_model is None:
                started = time.perf_counter()
                logger.info("Loading process-wide ADMETModel...")
                _admet_model = build_admet_model()
                logger.info(
                    "Process-wide ADMETModel loaded in %.3f seconds",
                    time.perf_counter() - started,
                )
    return _admet_model


def predict_with_admet_model(smiles):
    """Run one thread-safe prediction against the shared model instance."""
    model = get_admet_model()
    # The model and Chemprop/RDKit caches are shared process-wide. Serialize
    # predict calls until concurrent access is explicitly proven safe.
    with _admet_model_predict_lock:
        return model.predict(smiles)


# Function to canonicalize SMILES
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canonicalize_unique_smiles(smiles):
    """Canonicalize valid SMILES and deduplicate them without reordering."""
    normalized = []
    seen = set()
    for value in smiles:
        canonical = canonicalize_smiles(value)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def generate_props_for_smiles(smiles):
    """Predict the complete ADMET property row for each canonical molecule."""
    canonical_smiles = canonicalize_unique_smiles(smiles)
    print(f"Number of postprocessed SMILES: {len(canonical_smiles)}")
    if not canonical_smiles:
        raise ValueError("No valid SMILES after RDKit canonicalization.")
    props = predict_with_admet_model(canonical_smiles)
    props_df = pd.DataFrame(props)
    props_df.reset_index(inplace=True)
    props_df.rename(columns={"index": "SMILES"}, inplace=True)
    return props_df


def select_props_for_preferences(props_df, preference_json):
    """Return only the requested ADMET columns while preserving row order."""
    try:
        pref_dict = json.loads(preference_json)
    except (TypeError, json.JSONDecodeError):
        return props_df.iloc[0:0][["SMILES"]]
    if not isinstance(pref_dict, dict):
        return props_df.iloc[0:0][["SMILES"]]
    valid_columns = props_df.columns.intersection(pref_dict.keys())
    return props_df[["SMILES", *valid_columns.tolist()]]


def generate_props(smiles_path, props_path=None):
    """
    Generates properties for the preprocessed SMILES files using the ADMET model
    :param smiles_path: str, path to the preprocessed SMILES file
    :param output_path: str, path to save the properties as a csv file
    """

    # load the preprocessed SMILES
    try:
        # Read only the SMILES column to reduce memory use.
        df_input = pd.read_csv(smiles_path, usecols=['SMILES'])
        smiles = df_input['SMILES'].dropna().tolist()
    except ValueError:
        # If the column is not named SMILES, use the first column instead.
        print("Warning: 'SMILES' column not found, attempting to read the first column.")
        df_input = pd.read_csv(smiles_path)
        smiles = df_input.iloc[:, 0].tolist()
    # ---------------------------------------

    # Deduplicate after canonicalization while preserving source order.
    print(f"Number of preprocessed SMILES: {len(smiles)}")

    props_df = generate_props_for_smiles(smiles)
    if props_path:
        props_df.to_csv(props_path, index=False)
    return props_df


def generate_props_critic(smiles_list, preference_json, props_path=None):
    smiles = list(smiles_list)
    print(f"Number of preprocessed SMILES: {len(smiles)}")

    props_df = generate_props_for_smiles(smiles)
    selected_props_df = select_props_for_preferences(props_df, preference_json)
    if props_path:
        selected_props_df.to_csv(props_path, index=False)
    selected_props_json = selected_props_df.to_dict(orient='records')
    return selected_props_json
    

def _merge_candidate_source_columns(ranked_df, source_df):
    if source_df is None or source_df.empty:
        return ranked_df

    source = source_df.copy()
    source_columns = [str(column) for column in source.columns]
    smiles_column = next(
        (column for column in ("SMILES", "smiles") if column in source_columns),
        source_columns[0] if source_columns else None,
    )
    if smiles_column is None:
        return ranked_df

    source["_canonical_smiles"] = source[smiles_column].map(canonicalize_smiles)
    source = source.dropna(subset=["_canonical_smiles"]).drop_duplicates(
        subset=["_canonical_smiles"], keep="first"
    )
    source_by_smiles = source.set_index("_canonical_smiles", drop=False)

    records = []
    for _, ranked_row in ranked_df.iterrows():
        smiles = str(ranked_row["SMILES"])
        canonical = canonicalize_smiles(smiles) or smiles
        if canonical in source_by_smiles.index:
            source_row = source_by_smiles.loc[canonical]
            if isinstance(source_row, pd.DataFrame):
                source_row = source_row.iloc[0]
            record = {
                key: value
                for key, value in source_row.to_dict().items()
                if key != "_canonical_smiles"
            }
        else:
            record = {}
        record["SMILES"] = smiles
        record.update(ranked_row.to_dict())
        records.append(record)
    return pd.DataFrame(records)


def get_rank_based_candidates(
    preference_json,
    df,
    smiles2template_rgroups,
    output_suffix=None,
    top_n=5,
    source_df=None,
):
    """Rank candidates by mean property rank and write the selected CSV."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    try:
        pref_dict = json.loads(preference_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("preference_json must be a JSON object string") from exc
    if not isinstance(pref_dict, dict):
        raise ValueError("preference_json must decode to a JSON object")

    temp_df = df.copy()
    rank_cols = []
    direction_aliases = {
        "higher": "higher",
        "higher_better": "higher",
        "higher-is-better": "higher",
        "higher is better": "higher",
        "high": "higher",
        "increase": "higher",
        "increased": "higher",
        "maximize": "higher",
        "maximise": "higher",
        "max": "higher",
        "up": "higher",
        "true": "higher",
        "positive": "higher",
        "1": "higher",
        "lower": "lower",
        "low": "lower",
        "decrease": "lower",
        "decreased": "lower",
        "minimize": "lower",
        "minimise": "lower",
        "min": "lower",
        "down": "lower",
        "false": "lower",
        "negative": "lower",
        "0": "lower",
        "lower_better": "lower",
        "lower-is-better": "lower",
        "lower is better": "lower",
    }
    invalid_preferences = {}

    for col, pref in pref_dict.items():
        if col not in temp_df.columns:
            continue
            
        raw_pref = str(pref).strip()
        pref = direction_aliases.get(raw_pref.lower())
        if pref is None:
            invalid_preferences[col] = raw_pref
            continue
        rank_col_name = f"{col}_rank"
        
        if pref in ['higher', 'lower']:
            # Convert the property column to numeric values.
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            
            if pref == 'higher':
                # Larger values rank closer to 1.
                temp_df[rank_col_name] = temp_df[col].rank(ascending=False, method='min', na_option='bottom')
            else:
                # Smaller values rank closer to 1.
                temp_df[rank_col_name] = temp_df[col].rank(ascending=True, method='min', na_option='bottom')
            
        if rank_col_name in temp_df.columns:
            rank_cols.append(rank_col_name)

    if invalid_preferences:
        invalid_text = ", ".join(f"{col}={value!r}" for col, value in invalid_preferences.items())
        raise ValueError(
            "Invalid preference_json direction(s): "
            f"{invalid_text}. Use exactly 'higher' or 'lower' as values, "
            "for example {\"BBB_Martins\": \"higher\"}."
        )

    if rank_cols:
        temp_df['average_rank'] = temp_df[rank_cols].mean(axis=1)
        selected = temp_df.nsmallest(top_n, 'average_rank')
    else:
        selected = temp_df.head(top_n)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_name = "admet_ranked.csv"
    if output_suffix:
        output_name = f"admet_ranked_{output_suffix}.csv"
    output_df = _merge_candidate_source_columns(selected, source_df)
    output_df.to_csv(TMP_DIR / output_name, index=False)

    candidates = [
        {
            "SMILES": smiles,
            "Template": smiles2template_rgroups.get(smiles, {}).get("Template", ""),
            "R-groups": smiles2template_rgroups.get(smiles, {}).get("R-groups", ""),
        }
        for smiles in selected['SMILES'].tolist()
    ]
    
    return candidates
