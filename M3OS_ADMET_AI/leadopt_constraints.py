"""Runtime-defined ADMET and molecular constraint screening."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore  # noqa: E402

from admet_prediction_admetai import canonicalize_smiles


PROPERTY_TO_DATASET = {
    "herg": "hERG",
    "bbb": "BBB_Martins",
    "solubility": "Solubility_AqSolDB",
    "cyp3a4": "CYP3A4_Veith",
    "clearance": "Clearance_Hepatocyte_AZ",
    "caco2": "Caco2_Wang",
    "ppb": "PPBR_AZ",
    "ames": "AMES",
    "molecular_weight": "molecular_weight",
    "logp": "logP",
    "hydrogen_bond_acceptors": "hydrogen_bond_acceptors",
    "hydrogen_bond_donors": "hydrogen_bond_donors",
    "lipinski": "Lipinski",
    "qed": "QED",
    "stereo_centers": "stereo_centers",
    "tpsa": "tpsa",
}

MOLECULAR_PROPERTY_ALIASES = {
    "smiles_validity": "smiles_validity",
    "validity": "smiles_validity",
    "molecular_weight": "molecular_weight",
    "mol_weight": "molecular_weight",
    "molwt": "molecular_weight",
    "logp": "logp",
    "tpsa": "tpsa",
    "hbd": "hbd",
    "hydrogen_bond_donors": "hbd",
    "hba": "hba",
    "hydrogen_bond_acceptors": "hba",
    "rotatable_bonds": "rotatable_bonds",
    "formal_charge": "formal_charge",
    "synthetic_accessibility": "synthetic_accessibility",
    "sa_score": "synthetic_accessibility",
    "pains_filter": "pains_filter",
    "structural_alerts": "structural_alerts",
    "brenk_filter": "structural_alerts",
    "tanimoto_similarity": "tanimoto_similarity",
}

ACTIVITY_PROPERTY_NAMES = {
    "binding_affinity",
    "binding_probability",
    "interaction_probability",
    "activity_probability",
    "boltz_binding_probability",
    "transformercpi2_interaction_probability",
    "nesso_affinity_probability_binary",
    "nesso_affinity_pred_value",
}

OPERATOR_ALIASES = {
    "<": "lt",
    "lt": "lt",
    "less_than": "lt",
    "<=": "lte",
    "lte": "lte",
    "at_most": "lte",
    ">": "gt",
    "gt": "gt",
    "greater_than": "gt",
    ">=": "gte",
    "gte": "gte",
    "at_least": "gte",
    "=": "eq",
    "==": "eq",
    "eq": "eq",
    "between": "between",
    "range": "between",
    "inside": "between",
    "outside": "outside",
    "pass": "pass",
    "must_pass": "pass",
    "fail": "fail",
    "must_fail": "fail",
}

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(_pains_params)

_brenk_params = FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(_brenk_params)


def parse_task_contract(task_contract_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(task_contract_json or ""))
    except json.JSONDecodeError as exc:
        raise ValueError("task_contract_json must be a JSON object string") from exc
    if not isinstance(payload, dict):
        raise ValueError("task_contract_json must decode to a JSON object")
    validate_task_contract(payload)
    return payload


def _property_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("property") or entry.get("name") or "").strip().casefold()


def _entry_source(entry: Mapping[str, Any]) -> str:
    source = str(entry.get("source") or "").strip().casefold().replace("-", "_")
    if source in {"rdkit", "molecular", "molecule", "chemistry"}:
        return "molecular"
    if source in {"admet", "admet_ai", "admetai"}:
        return "admet"
    if source in {"activity", "target_activity", "binding"}:
        return "activity"
    if source:
        raise ValueError(f"Unsupported constraint source {source!r}")

    property_name = _property_name(entry)
    if property_name in ACTIVITY_PROPERTY_NAMES:
        return "activity"
    if property_name in MOLECULAR_PROPERTY_ALIASES:
        return "molecular"
    if entry.get("endpoint") or property_name in PROPERTY_TO_DATASET:
        return "admet"
    raise ValueError(
        f"Cannot infer a source for constraint property {property_name!r}; "
        "set source to 'rdkit', 'admet', or 'activity'."
    )


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Constraint field {field!r} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Constraint field {field!r} must be numeric") from exc


def _operator_spec(entry: Mapping[str, Any], *, allow_implicit_pass: bool) -> dict[str, Any]:
    raw_operator = str(entry.get("operator") or entry.get("comparator") or "").strip().casefold()
    operator = OPERATOR_ALIASES.get(raw_operator)
    if raw_operator and operator is None:
        raise ValueError(f"Unsupported constraint operator {raw_operator!r}")

    if operator in {"between", "outside"} or (
        not operator
        and any(key in entry for key in ("min", "minimum"))
        and any(key in entry for key in ("max", "maximum"))
    ):
        minimum = _number(entry.get("min", entry.get("minimum")), "min")
        maximum = _number(entry.get("max", entry.get("maximum")), "max")
        if minimum > maximum:
            raise ValueError("Constraint min cannot exceed max")
        return {"operator": operator or "between", "min": minimum, "max": maximum}

    if operator in {"lt", "lte", "gt", "gte", "eq"}:
        raw_value = entry.get("value", entry.get("threshold"))
        return {"operator": operator, "value": _number(raw_value, "value")}
    if operator in {"pass", "fail"}:
        return {"operator": operator}

    threshold = entry.get("threshold")
    if isinstance(threshold, str):
        text = threshold.strip().casefold().replace("−", "-")
        one_sided = re.match(r"^(<=|>=|<|>)\s*([-+]?\d+(?:\.\d+)?)", text)
        if one_sided:
            return {
                "operator": OPERATOR_ALIASES[one_sided.group(1)],
                "value": float(one_sided.group(2)),
            }
        ranged = re.match(
            r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:to|through)\s*([-+]?\d+(?:\.\d+)?)",
            text,
        )
        if ranged:
            minimum, maximum = float(ranged.group(1)), float(ranged.group(2))
            if minimum > maximum:
                raise ValueError("Constraint range minimum cannot exceed maximum")
            return {"operator": "between", "min": minimum, "max": maximum}

    description = str(entry.get("description") or "").casefold()
    if allow_implicit_pass and (
        threshold in (None, "") or "must pass" in description or "no " in description
    ):
        return {"operator": "pass"}
    raise ValueError(
        f"Constraint for {_property_name(entry)!r} needs an explicit operator and value, "
        "a min/max interval, or a parseable threshold string."
    )


def _prediction_column(entry: Mapping[str, Any]) -> str | None:
    if _entry_source(entry) == "activity":
        return None
    endpoint = str(entry.get("endpoint") or "").strip()
    if endpoint:
        return endpoint
    property_name = _property_name(entry)
    column = PROPERTY_TO_DATASET.get(property_name)
    if column:
        return column
    if _entry_source(entry) == "admet" and property_name:
        return str(entry.get("property") or entry.get("name")).strip()
    return None


def _normalized_hard_constraints(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = contract.get("hard_constraints") or []
    if not isinstance(entries, list):
        raise ValueError("hard_constraints must be a list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"hard_constraints[{index}] must be an object")
        property_name = _property_name(entry)
        if not property_name:
            raise ValueError(f"hard_constraints[{index}] has no property")
        source = _entry_source(entry)
        allow_implicit_pass = (
            source == "molecular"
            and MOLECULAR_PROPERTY_ALIASES.get(property_name)
            in {"smiles_validity", "pains_filter", "structural_alerts"}
        )
        spec = _operator_spec(entry, allow_implicit_pass=allow_implicit_pass)
        normalized.append(
            {
                **dict(entry),
                **spec,
                "property": property_name,
                "source": source,
                "endpoint": _prediction_column(entry) if source == "admet" else None,
            }
        )
    return normalized


def _section_entries(contract: Mapping[str, Any], *names: str) -> list[Mapping[str, Any]]:
    for name in names:
        entries = contract.get(name)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"{name} must be a list")
        invalid = [index for index, entry in enumerate(entries) if not isinstance(entry, Mapping)]
        if invalid:
            raise ValueError(f"{name} entries must be objects; invalid indexes: {invalid}")
        return list(entries)
    return []


def validate_task_contract(contract: Mapping[str, Any]) -> None:
    _normalized_hard_constraints(contract)
    for kind, entries in (
        ("objective", _section_entries(contract, "optimization_objectives", "objectives")),
        ("hold", _section_entries(contract, "hold_constant", "holds")),
    ):
        for entry in entries:
            property_name = _property_name(entry)
            if not property_name:
                raise ValueError(f"{kind} entry has no property")
            source = _entry_source(entry)
            if source != "activity" and not _prediction_column(entry):
                raise ValueError(f"Unsupported {kind} property {property_name!r}")
            if kind == "objective":
                direction = str(entry.get("direction") or "").casefold()
                if direction not in {"minimize", "maximize"}:
                    raise ValueError(
                        f"Objective {property_name!r} direction must be minimize or maximize"
                    )
                _number(entry.get("threshold", 0.0), "threshold")
            else:
                direction = str(entry.get("direction") or "change").casefold()
                if direction not in {"change", "increase", "decrease"}:
                    raise ValueError(
                        f"Hold {property_name!r} direction must be change, increase, or decrease"
                    )
                _number(entry.get("tolerance", 0.0), "tolerance")


def required_admet_columns(contract: Mapping[str, Any]) -> list[str]:
    validate_task_contract(contract)
    columns: list[str] = []
    for entry in [
        *_section_entries(contract, "optimization_objectives", "objectives"),
        *_section_entries(contract, "hold_constant", "holds"),
        *[
            item
            for item in _normalized_hard_constraints(contract)
            if item["source"] == "admet"
        ],
    ]:
        column = _prediction_column(entry)
        if column and column not in columns:
            columns.append(column)
    return columns


def _reference_fingerprint(reference_smiles: str):
    text = str(reference_smiles or "").strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError("reference_smiles is invalid")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def _constraint_result(value: Any, constraint: Mapping[str, Any]) -> tuple[bool, float | None]:
    operator = str(constraint["operator"])
    if operator == "pass":
        return bool(value), None
    if operator == "fail":
        return not bool(value), None
    numeric = float(value)
    if operator in {"lt", "lte"}:
        target = float(constraint["value"])
        return (numeric < target if operator == "lt" else numeric <= target), target - numeric
    if operator in {"gt", "gte"}:
        target = float(constraint["value"])
        return (numeric > target if operator == "gt" else numeric >= target), numeric - target
    if operator == "eq":
        target = float(constraint["value"])
        margin = -abs(numeric - target)
        return numeric == target, margin
    minimum, maximum = float(constraint["min"]), float(constraint["max"])
    inside = minimum <= numeric <= maximum
    margin = min(numeric - minimum, maximum - numeric)
    return (inside, margin) if operator == "between" else (not inside, -margin)


def _molecular_value(
    property_name: str,
    molecule: Chem.Mol,
    canonical: str,
    reference_fingerprint: Any,
) -> Any:
    property_name = MOLECULAR_PROPERTY_ALIASES[property_name]
    if property_name == "smiles_validity":
        return "." not in canonical and not any(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
    if property_name == "molecular_weight":
        return float(Descriptors.MolWt(molecule))
    if property_name == "logp":
        return float(Crippen.MolLogP(molecule))
    if property_name == "tpsa":
        return float(Descriptors.TPSA(molecule))
    if property_name == "hbd":
        return int(Lipinski.NumHDonors(molecule))
    if property_name == "hba":
        return int(Lipinski.NumHAcceptors(molecule))
    if property_name == "rotatable_bonds":
        return int(Lipinski.NumRotatableBonds(molecule))
    if property_name == "formal_charge":
        return int(Chem.GetFormalCharge(molecule))
    if property_name == "synthetic_accessibility":
        return float(sascorer.calculateScore(molecule))
    if property_name == "pains_filter":
        return not PAINS_CATALOG.HasMatch(molecule)
    if property_name == "structural_alerts":
        return not BRENK_CATALOG.HasMatch(molecule)
    if property_name == "tanimoto_similarity":
        if reference_fingerprint is None:
            raise ValueError("tanimoto_similarity requires reference_smiles")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fingerprint, reference_fingerprint))
    raise ValueError(f"Unsupported molecular property {property_name!r}")


def _molecular_constraints(
    smiles: str,
    *,
    constraints: Sequence[Mapping[str, Any]],
    reference_fingerprint: Any,
) -> tuple[bool, dict[str, Any], str | None]:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return False, {"smiles_validity": {"value": False, "passed": False}}, None
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        return False, {"smiles_validity": {"value": False, "passed": False}}, None
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return False, {"smiles_validity": {"value": False, "passed": False}}, None

    details: dict[str, Any] = {}
    passed = True
    for index, constraint in enumerate(constraints):
        property_name = str(constraint["property"])
        value = _molecular_value(
            property_name,
            molecule,
            canonical,
            reference_fingerprint,
        )
        constraint_passed, margin = _constraint_result(value, constraint)
        key = property_name if property_name not in details else f"{property_name}_{index}"
        details[key] = {
            "value": value,
            "operator": constraint["operator"],
            "target": constraint.get("value"),
            "min": constraint.get("min"),
            "max": constraint.get("max"),
            "margin": margin,
            "passed": constraint_passed,
        }
        passed = passed and constraint_passed
    return passed, details, canonical


def _casefold_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in values.items()}


def _prediction_value(prediction: Mapping[str, Any], column: str) -> Any:
    if column in prediction:
        return prediction[column]
    return _casefold_mapping(prediction).get(column.casefold())


def _baseline(entry: Mapping[str, Any], baselines: Mapping[str, Any]) -> float:
    property_name = _property_name(entry)
    baseline_lookup = _casefold_mapping(baselines)
    value = entry.get("baseline", baseline_lookup.get(property_name))
    if value is None:
        endpoint = str(entry.get("endpoint") or "").casefold()
        value = baseline_lookup.get(endpoint)
    if value is None:
        raise ValueError(f"Missing baseline for ADMET property {property_name!r}")
    return float(value)


def _admet_constraints(
    prediction: Mapping[str, Any],
    contract: Mapping[str, Any],
    hard_constraints: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool, list[dict[str, Any]], float]:
    baselines = contract.get("baseline_values")
    if not isinstance(baselines, Mapping):
        baselines = {}
    objective_entries = _section_entries(contract, "optimization_objectives", "objectives")
    hold_entries = _section_entries(contract, "hold_constant", "holds")

    details: list[dict[str, Any]] = []
    objective_pass = True
    hold_pass = True
    hard_pass = True
    margins: list[float] = []

    for entry in objective_entries:
        if _entry_source(entry) == "activity":
            continue
        property_name = _property_name(entry)
        column = _prediction_column(entry)
        value = _prediction_value(prediction, str(column)) if column else None
        if value is None or pd.isna(value):
            objective_pass = False
            details.append({"kind": "objective", "property": property_name, "endpoint": column, "passed": False, "reason": "missing_prediction"})
            continue
        baseline = _baseline(entry, baselines)
        threshold = float(entry.get("threshold", 0.0))
        direction = str(entry.get("direction") or "").casefold()
        numeric = float(value)
        margin = baseline - numeric - threshold if direction == "minimize" else numeric - baseline - threshold
        passed = margin >= 0.0
        objective_pass = objective_pass and passed
        margins.append(margin)
        details.append({"kind": "objective", "property": property_name, "endpoint": column, "value": numeric, "baseline": baseline, "direction": direction, "threshold": threshold, "margin": margin, "passed": passed})

    for entry in hold_entries:
        if _entry_source(entry) == "activity":
            continue
        property_name = _property_name(entry)
        column = _prediction_column(entry)
        value = _prediction_value(prediction, str(column)) if column else None
        if value is None or pd.isna(value):
            hold_pass = False
            details.append({"kind": "hold", "property": property_name, "endpoint": column, "passed": False, "reason": "missing_prediction"})
            continue
        baseline = _baseline(entry, baselines)
        tolerance = float(entry.get("tolerance", 0.0))
        direction = str(entry.get("direction") or "change").casefold()
        numeric = float(value)
        delta = numeric - baseline
        if direction == "increase":
            margin = tolerance - delta
        elif direction == "decrease":
            margin = tolerance + delta
        else:
            margin = tolerance - abs(delta)
        passed = margin >= 0.0
        hold_pass = hold_pass and passed
        margins.append(margin)
        details.append({"kind": "hold", "property": property_name, "endpoint": column, "value": numeric, "baseline": baseline, "direction": direction, "tolerance": tolerance, "margin": margin, "passed": passed})

    for constraint in hard_constraints:
        column = str(constraint["endpoint"])
        value = _prediction_value(prediction, column)
        property_name = str(constraint["property"])
        if value is None or pd.isna(value):
            hard_pass = False
            details.append({"kind": "hard", "property": property_name, "endpoint": column, "passed": False, "reason": "missing_prediction"})
            continue
        passed, margin = _constraint_result(value, constraint)
        hard_pass = hard_pass and passed
        if margin is not None:
            margins.append(margin)
        details.append({"kind": "hard", "property": property_name, "endpoint": column, "value": float(value) if not isinstance(value, bool) else value, "operator": constraint["operator"], "target": constraint.get("value"), "min": constraint.get("min"), "max": constraint.get("max"), "margin": margin, "passed": passed})

    return objective_pass, hold_pass, hard_pass, details, min(margins, default=0.0)


def assess_candidate_frame(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    reference_smiles: str,
) -> pd.DataFrame:
    validate_task_contract(contract)
    columns = [str(column) for column in source.columns]
    smiles_column = next(
        (column for column in ("SMILES", "smiles") if column in columns),
        columns[0] if columns else None,
    )
    if smiles_column is None:
        raise ValueError("Candidate CSV has no columns")

    prediction_by_smiles = {
        str(row["SMILES"]): row.to_dict()
        for _, row in predictions.iterrows()
    }
    normalized_hard = _normalized_hard_constraints(contract)
    molecular_hard = [item for item in normalized_hard if item["source"] == "molecular"]
    admet_hard = [item for item in normalized_hard if item["source"] == "admet"]
    deferred_activity = [item for item in normalized_hard if item["source"] == "activity"]
    reference_fingerprint = _reference_fingerprint(reference_smiles)
    required_columns = required_admet_columns(contract)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _, source_row in source.iterrows():
        raw_smiles = str(source_row.get(smiles_column) or "").strip()
        molecular_pass, molecular_details, canonical = _molecular_constraints(
            raw_smiles,
            constraints=molecular_hard,
            reference_fingerprint=reference_fingerprint,
        )
        dedup_key = canonical or raw_smiles
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        record = source_row.to_dict()
        record["SMILES"] = canonical or raw_smiles
        prediction = prediction_by_smiles.get(canonical or "", {})
        for column in required_columns:
            record[column] = _prediction_value(prediction, column)
        objective_pass, hold_pass, admet_hard_pass, admet_details, minimum_margin = _admet_constraints(
            prediction,
            contract,
            admet_hard,
        )
        record["deterministic_hard_pass"] = bool(molecular_pass)
        record["admet_hard_pass"] = bool(admet_hard_pass)
        record["admet_objectives_pass"] = bool(objective_pass)
        record["admet_holds_pass"] = bool(hold_pass)
        record["admet_constraints_pass"] = bool(admet_hard_pass and objective_pass and hold_pass)
        record["leadopt_constraints_pass"] = bool(molecular_pass and admet_hard_pass and objective_pass and hold_pass)
        record["admet_minimum_margin"] = float(minimum_margin)
        record["hard_constraint_details"] = json.dumps(molecular_details, ensure_ascii=False)
        record["admet_constraint_details"] = json.dumps(admet_details, ensure_ascii=False)
        record["deferred_activity_constraints"] = json.dumps(deferred_activity, ensure_ascii=False)
        records.append(record)

    return pd.DataFrame(records)


def write_constraint_results(
    assessed: pd.DataFrame,
    *,
    scored_csv: Path,
    eligible_csv: Path,
) -> pd.DataFrame:
    scored_csv.parent.mkdir(parents=True, exist_ok=True)
    assessed.to_csv(scored_csv, index=False)
    eligible = assessed.loc[assessed["leadopt_constraints_pass"] == True].copy()  # noqa: E712
    eligible = eligible.sort_values("admet_minimum_margin", ascending=False, kind="stable")
    eligible.to_csv(eligible_csv, index=False)
    return eligible
