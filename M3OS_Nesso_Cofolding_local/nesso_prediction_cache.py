"""Persistent JSON cache for protein-aware Nesso activity predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_SCHEMA_VERSION = "nesso-prediction-cache-v2"


def _json_cache_path(path: Path) -> Path:
    if path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}:
        return path.with_suffix(".json")
    return path


def default_cache_path() -> Path:
    configured = os.environ.get("NESSO_PREDICTION_CACHE_PATH", "").strip()
    if configured:
        return _json_cache_path(Path(configured).expanduser().resolve())
    return PROJECT_DIR / "cache" / "nesso_predictions.json"


def default_model_namespace() -> str:
    configured = os.environ.get("NESSO_PREDICTION_CACHE_NAMESPACE", "").strip()
    if configured:
        return configured
    return "nesso-prediction-cache-v1:nesso-local:seed-42"


def normalize_protein_sequence(protein_sequence: str) -> str:
    residues = []
    for line in str(protein_sequence or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(">"):
            residues.append("".join(stripped.split()))
    normalized = "".join(residues).upper()
    if not normalized:
        raise ValueError("Nesso cache requires a non-empty protein sequence")
    return normalized


def protein_sequence_hash(protein_sequence: str) -> str:
    normalized = normalize_protein_sequence(protein_sequence)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _record_key(model_namespace: str, protein_hash: str, canonical_smiles: str) -> str:
    identity = "\u001f".join((model_namespace, protein_hash, canonical_smiles))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class NessoPredictionCache:
    """Process-resident JSON cache keyed by model, protein, and canonical SMILES."""

    def __init__(
        self,
        path: Path | str | None = None,
        model_namespace: str | None = None,
    ) -> None:
        requested_path = Path(path or default_cache_path()).expanduser().resolve()
        self.path = _json_cache_path(requested_path)
        self.legacy_sqlite_path = (
            requested_path
            if requested_path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}
            else requested_path.with_suffix(".sqlite")
        )
        self.model_namespace = str(
            model_namespace or default_model_namespace()
        ).strip()
        if not self.model_namespace:
            raise ValueError("Nesso cache model namespace cannot be empty")
        self._lock = threading.RLock()
        self._loaded = False
        self._records: dict[str, dict[str, Any]] = {}

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            migrated = False
            if self.path.is_file():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid Nesso JSON cache: {self.path}")
                records = payload.get("records", {})
                if not isinstance(records, dict):
                    raise ValueError(f"Invalid Nesso JSON cache records: {self.path}")
                self._records = {
                    str(key): dict(value)
                    for key, value in records.items()
                    if isinstance(value, dict)
                }
            elif self.legacy_sqlite_path.is_file():
                self._records = self._read_legacy_sqlite(self.legacy_sqlite_path)
                migrated = bool(self._records)
            self._loaded = True
            if migrated:
                self._persist_locked()

    @staticmethod
    def _read_legacy_sqlite(path: Path) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            rows = connection.execute(
                """
                SELECT model_namespace,
                       protein_sha256,
                       canonical_smiles,
                       affinity_pred_value,
                       affinity_probability_binary,
                       source_tool,
                       created_at_utc,
                       updated_at_utc
                FROM nesso_predictions
                """
            ).fetchall()
        finally:
            connection.close()
        for (
            model_namespace,
            protein_hash,
            canonical_smiles,
            pred_value,
            probability,
            source_tool,
            created_at,
            updated_at,
        ) in rows:
            key = _record_key(
                str(model_namespace), str(protein_hash), str(canonical_smiles)
            )
            records[key] = {
                "model_namespace": str(model_namespace),
                "protein_sha256": str(protein_hash),
                "canonical_smiles": str(canonical_smiles),
                "affinity_pred_value": float(pred_value),
                "affinity_probability_binary": float(probability),
                "source_tool": str(source_tool),
                "created_at_utc": str(created_at),
                "updated_at_utc": str(updated_at),
            }
        return records

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "records": self._records,
        }
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def get_many(
        self,
        protein_sequence: str,
        canonical_smiles: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        keys = list(dict.fromkeys(str(value) for value in canonical_smiles if value))
        if not keys:
            return {}
        protein_hash = protein_sequence_hash(protein_sequence)
        self._load()
        found: dict[str, dict[str, Any]] = {}
        with self._lock:
            for smiles in keys:
                record = self._records.get(
                    _record_key(self.model_namespace, protein_hash, smiles)
                )
                if record is None:
                    continue
                found[smiles] = {
                    "SMILES": smiles,
                    "Nesso_affinity_pred_value": float(record["affinity_pred_value"]),
                    "Nesso_affinity_probability_binary": float(
                        record["affinity_probability_binary"]
                    ),
                }
        return found

    def upsert_many(
        self,
        protein_sequence: str,
        records: Iterable[dict[str, Any]],
        *,
        source_tool: str,
    ) -> int:
        protein_hash = protein_sequence_hash(protein_sequence)
        timestamp = datetime.now(timezone.utc).isoformat()
        prepared: list[tuple[str, float, float]] = []
        for record in records:
            canonical_smiles = str(record.get("SMILES") or "").strip()
            if not canonical_smiles:
                continue
            try:
                pred_value = float(record["Nesso_affinity_pred_value"])
                probability = float(record["Nesso_affinity_probability_binary"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(pred_value) or not math.isfinite(probability):
                continue
            prepared.append((canonical_smiles, pred_value, probability))
        if not prepared:
            return 0

        self._load()
        with self._lock:
            for canonical_smiles, pred_value, probability in prepared:
                key = _record_key(
                    self.model_namespace, protein_hash, canonical_smiles
                )
                existing = self._records.get(key, {})
                self._records[key] = {
                    "model_namespace": self.model_namespace,
                    "protein_sha256": protein_hash,
                    "canonical_smiles": canonical_smiles,
                    "affinity_pred_value": pred_value,
                    "affinity_probability_binary": probability,
                    "source_tool": str(source_tool or "unknown"),
                    "created_at_utc": existing.get("created_at_utc", timestamp),
                    "updated_at_utc": timestamp,
                }
            self._persist_locked()
        return len(prepared)
