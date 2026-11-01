"""SQLite storage: applications, decisions (with reasons + model version) and repayment outcomes."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    application_id TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    payload_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    application_id TEXT PRIMARY KEY REFERENCES applications(application_id),
    decided_at     TEXT NOT NULL,
    pd             REAL NOT NULL,
    credit_score   INTEGER NOT NULL,
    risk_band      TEXT NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('APPROVE','REFER','DECLINE')),
    reasons_json   TEXT NOT NULL,
    model_version  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    application_id TEXT PRIMARY KEY REFERENCES applications(application_id),
    recorded_at    TEXT NOT NULL,
    defaulted      INTEGER NOT NULL CHECK (defaulted IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions(decision);
"""


class DuplicateApplicationError(Exception):
    """An application with this id already exists."""


class NotFoundError(Exception):
    """No application with this id."""


class OutcomeExistsError(Exception):
    """An outcome was already recorded for this application."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Short-lived connection (safe with FastAPI's thread pool). Commits on success, rolls back on error."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:  # transaction
            yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_assessment(db_path: str | Path, application_id: str, payload: dict[str, Any], pd_value: float,
                    credit_score: int, band: str, decision: str, reasons: list[str], model_version: str) -> None:
    """Insert the application AND its decision in ONE transaction (both or neither)."""
    try:
        with connect(db_path) as conn:
            conn.execute("INSERT INTO applications VALUES (?,?,?)", (application_id, _now(), json.dumps(payload)))
            conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                         (application_id, _now(), pd_value, credit_score, band, decision, json.dumps(reasons),
                          model_version))
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
            raise DuplicateApplicationError(application_id) from exc
        raise  # e.g. CHECK / FOREIGN KEY violation: a real error, not a duplicate


def application_exists(db_path: str | Path, application_id: str) -> bool:
    with connect(db_path) as conn:
        return conn.execute("SELECT 1 FROM applications WHERE application_id=?", (application_id,)).fetchone() is not None


def _decision_row(row: sqlite3.Row) -> dict[str, Any]:
    return {"application_id": row["application_id"], "decided_at": row["decided_at"], "pd": row["pd"],
            "credit_score": row["credit_score"], "risk_band": row["risk_band"], "decision": row["decision"],
            "reasons": json.loads(row["reasons_json"]), "model_version": row["model_version"]}


def get_application(db_path: str | Path, application_id: str) -> dict[str, Any]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT d.*, a.payload_json, o.defaulted FROM decisions d JOIN applications a USING(application_id) "
            "LEFT JOIN outcomes o USING(application_id) WHERE d.application_id=?", (application_id,)).fetchone()
    if row is None:
        raise NotFoundError(application_id)
    out = _decision_row(row)
    out["applicant"] = json.loads(row["payload_json"])
    out["outcome_defaulted"] = None if row["defaulted"] is None else bool(row["defaulted"])
    return out


def referral_queue(db_path: str | Path) -> list[dict[str, Any]]:
    """REFER decisions that are still awaiting a final outcome, oldest first."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT d.* FROM decisions d LEFT JOIN outcomes o USING(application_id) "
            "WHERE d.decision='REFER' AND o.application_id IS NULL ORDER BY d.decided_at, d.application_id").fetchall()
    return [_decision_row(r) for r in rows]


def record_outcome(db_path: str | Path, application_id: str, defaulted: bool) -> None:
    try:
        with connect(db_path) as conn:
            if conn.execute("SELECT 1 FROM applications WHERE application_id=?", (application_id,)).fetchone() is None:
                raise NotFoundError(application_id)
            conn.execute("INSERT INTO outcomes VALUES (?,?,?)", (application_id, _now(), int(defaulted)))
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
            raise OutcomeExistsError(application_id) from exc
        raise


def all_application_payloads(db_path: str | Path) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [json.loads(r["payload_json"]) for r in conn.execute("SELECT payload_json FROM applications")]


def feedback_rows(db_path: str | Path) -> list[tuple[dict[str, Any], int]]:
    """(applicant payload, defaulted) for every application with a recorded outcome."""
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT a.payload_json, o.defaulted FROM outcomes o JOIN applications a "
                            "USING(application_id)").fetchall()
    return [(json.loads(r["payload_json"]), int(r["defaulted"])) for r in rows]
