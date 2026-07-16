"""
app/audit/logger.py
--------------------
Immutable audit trail for every compliance query.

Storage backends:
  1. JSON file  — human-readable, easy to download from UI
  2. SQLite DB  — structured, queryable, lightweight

Both backends are written on every ``log_answer()`` call so that
even if one fails, the other preserves the record.

All timestamps are UTC ISO-8601.
Records are append-only — no update or delete operations are exposed.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger as _log

from app.utils.config import get_settings
from app.utils.models import AuditRecord, ComplianceAnswer


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# SQLite backend
# ──────────────────────────────────────────────────────────────

_DB_PATH: Optional[Path] = None
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL,   -- JSON array of filenames
    confidence  REAL NOT NULL,
    topic       TEXT NOT NULL,
    risk_level  TEXT NOT NULL,
    owner       TEXT NOT NULL,
    escalated   INTEGER NOT NULL,
    refused     INTEGER NOT NULL,
    session_id  TEXT
);
"""


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        cfg = get_settings()
        _DB_PATH = cfg.audit_log_path.parent / "audit.db"
    return _DB_PATH


def _get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    _ensure_dir(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _write_sqlite(record: AuditRecord) -> None:
    """Insert one audit record into the SQLite database."""
    try:
        conn = _get_connection()
        conn.execute(
            """
            INSERT INTO audit_log
                (id, timestamp, question, answer, sources, confidence,
                 topic, risk_level, owner, escalated, refused, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.timestamp,
                record.question,
                record.answer,
                json.dumps(record.sources),
                record.confidence,
                record.topic,
                record.risk_level,
                record.owner,
                int(record.escalated),
                int(record.refused),
                record.session_id,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        _log.error(f"SQLite write failed: {exc}")


# ──────────────────────────────────────────────────────────────
# JSON file backend
# ──────────────────────────────────────────────────────────────

def _get_json_path() -> Path:
    cfg = get_settings()
    return cfg.audit_log_path


def _write_json(record: AuditRecord) -> None:
    """Append one audit record to the JSON log file."""
    json_path = _get_json_path()
    _ensure_dir(json_path)

    # Load existing records
    records: List[dict] = []
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                content = fh.read().strip()
                if content:
                    records = json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning(f"Could not read existing audit log: {exc}")

    records.append(record.model_dump())

    try:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
    except OSError as exc:
        _log.error(f"JSON write failed: {exc}")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def log_answer(
    answer: ComplianceAnswer,
    session_id: str = "",
) -> AuditRecord:
    """
    Persist a compliance answer to both the JSON file and SQLite DB.

    Parameters
    ----------
    answer:
        The ``ComplianceAnswer`` produced by the compliance agent.
    session_id:
        Optional Streamlit session identifier for filtering.

    Returns
    -------
    AuditRecord
        The persisted record (useful for immediate UI display).
    """
    record = AuditRecord(
        id=str(uuid.uuid4()),
        timestamp=_utc_now(),
        question=answer.question,
        answer=answer.answer,
        sources=answer.source_names,
        confidence=round(answer.confidence, 4),
        topic=answer.topic.value,
        risk_level=answer.risk_level.value,
        owner=answer.owner.value,
        escalated=answer.escalated,
        refused=answer.refused,
        session_id=session_id,
    )

    _write_json(record)
    _write_sqlite(record)

    _log.info(
        f"[audit] Logged: risk={record.risk_level} | "
        f"conf={record.confidence:.2f} | escalated={record.escalated}"
    )
    return record


def load_audit_log(limit: int = 100) -> List[AuditRecord]:
    """
    Load recent audit records from the JSON file.

    Parameters
    ----------
    limit:
        Maximum number of records to return (most recent first).

    Returns
    -------
    List[AuditRecord]
    """
    json_path = _get_json_path()
    if not json_path.exists():
        return []

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if not content:
                return []
            raw_records: List[dict] = json.loads(content)
    except (json.JSONDecodeError, OSError) as exc:
        _log.error(f"Failed to load audit log: {exc}")
        return []

    records = [AuditRecord(**r) for r in raw_records]
    # Return most recent first
    return list(reversed(records))[:limit]


def get_audit_stats() -> dict:
    """
    Compute summary statistics from the audit log for the dashboard.

    Returns
    -------
    dict with keys: total, escalated, refused, high_risk, avg_confidence
    """
    records = load_audit_log(limit=10_000)

    if not records:
        return {
            "total": 0,
            "escalated": 0,
            "refused": 0,
            "high_risk": 0,
            "avg_confidence": 0.0,
        }

    return {
        "total": len(records),
        "escalated": sum(1 for r in records if r.escalated),
        "refused": sum(1 for r in records if r.refused),
        "high_risk": sum(1 for r in records if r.risk_level == "HIGH"),
        "avg_confidence": round(
            sum(r.confidence for r in records) / len(records), 3
        ),
    }


def get_json_for_download() -> str:
    """Return the full audit log as a pretty-printed JSON string."""
    json_path = _get_json_path()
    if not json_path.exists():
        return "[]"
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return "[]"
