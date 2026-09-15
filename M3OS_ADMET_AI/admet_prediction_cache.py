"""Persistent cache for raw ADMET-AI prediction rows."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_SCHEMA_VERSION = "admet-prediction-cache-v1"


def default_cache_path() -> Path:
    configured = os.environ.get("ADMET_PREDICTION_CACHE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_DIR / "cache" / "admet_predictions.sqlite"


def default_model_namespace() -> str:
    configured = os.environ.get("ADMET_PREDICTION_CACHE_NAMESPACE", "").strip()
    if configured:
        return configured
    try:
        package_version = metadata.version("admet-ai")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    return f"{CACHE_SCHEMA_VERSION}:admet-ai-{package_version}"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    return str(value)


class AdmetPredictionCache:
    """SQLite-backed cache keyed by model namespace and canonical SMILES."""

    def __init__(
        self,
        path: Path | str | None = None,
        model_namespace: str | None = None,
    ) -> None:
        self.path = Path(path or default_cache_path()).expanduser().resolve()
        self.model_namespace = str(
            model_namespace or default_model_namespace()
        ).strip()
        if not self.model_namespace:
            raise ValueError("ADMET cache model namespace cannot be empty")
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _ensure_schema(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS admet_predictions (
                        model_namespace TEXT NOT NULL,
                        canonical_smiles TEXT NOT NULL,
                        values_json TEXT NOT NULL,
                        source_tool TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        PRIMARY KEY (model_namespace, canonical_smiles)
                    )
                    """
                )
            self._initialized = True

    def get_many(self, canonical_smiles: Iterable[str]) -> dict[str, dict[str, Any]]:
        keys = list(dict.fromkeys(str(value) for value in canonical_smiles if value))
        if not keys:
            return {}
        self._ensure_schema()
        found: dict[str, dict[str, Any]] = {}
        with self._lock, self._connect() as connection:
            chunk_size = 500
            for offset in range(0, len(keys), chunk_size):
                chunk = keys[offset : offset + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT canonical_smiles, values_json
                    FROM admet_predictions
                    WHERE model_namespace = ?
                      AND canonical_smiles IN ({placeholders})
                    """,
                    [self.model_namespace, *chunk],
                ).fetchall()
                for smiles, values_json in rows:
                    payload = json.loads(values_json)
                    if isinstance(payload, dict):
                        found[str(smiles)] = payload
        return found

    def upsert_many(
        self,
        records: Iterable[dict[str, Any]],
        *,
        source_tool: str,
    ) -> int:
        prepared = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for record in records:
            canonical_smiles = str(record.get("SMILES") or "").strip()
            if not canonical_smiles:
                continue
            payload = {
                str(key): _json_value(value)
                for key, value in record.items()
            }
            payload["SMILES"] = canonical_smiles
            prepared.append(
                (
                    self.model_namespace,
                    canonical_smiles,
                    json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    str(source_tool or "unknown"),
                    timestamp,
                    timestamp,
                )
            )
        if not prepared:
            return 0
        self._ensure_schema()
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO admet_predictions (
                    model_namespace,
                    canonical_smiles,
                    values_json,
                    source_tool,
                    created_at_utc,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_namespace, canonical_smiles) DO UPDATE SET
                    values_json = excluded.values_json,
                    source_tool = excluded.source_tool,
                    updated_at_utc = excluded.updated_at_utc
                """,
                prepared,
            )
        return len(prepared)
