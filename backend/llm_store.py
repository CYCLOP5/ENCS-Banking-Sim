"""llm_store.py — Persistent store for simulation runs and bank profiles."""
from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
@dataclass
class StoredRun:
    run_id: str
    run_type: str
    created_at: str
    request_json: Dict[str, Any]
    result_json: Dict[str, Any]
    bank_snapshot_json: Optional[Dict[str, Any]]
    summary_json: Optional[Dict[str, Any]]
class LlmStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        base = Path(__file__).parent / "data" / "output"
        default_path = str(base / "llm_store.db")
        self.db_path = db_path or os.environ.get("ENCS_LLM_DB_PATH", default_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    bank_snapshot_json TEXT,
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bank_profiles (
                    bank_id TEXT PRIMARY KEY,
                    name TEXT,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)"
            )
    def save_run(
        self,
        run_id: str,
        run_type: str,
        request_json: Dict[str, Any],
        result_json: Dict[str, Any],
        bank_snapshot_json: Optional[Dict[str, Any]] = None,
        summary_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "run_type": run_type,
            "created_at": _utc_now(),
            "request_json": json.dumps(request_json),
            "result_json": json.dumps(result_json),
            "bank_snapshot_json": json.dumps(bank_snapshot_json) if bank_snapshot_json else None,
            "summary_json": json.dumps(summary_json) if summary_json else None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, run_type, created_at, request_json, result_json, bank_snapshot_json, summary_json)
                VALUES (:run_id, :run_type, :created_at, :request_json, :result_json, :bank_snapshot_json, :summary_json)
                """,
                payload,
            )
    def get_run(self, run_id: str) -> Optional[StoredRun]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return StoredRun(
            run_id=row["run_id"],
            run_type=row["run_type"],
            created_at=row["created_at"],
            request_json=json.loads(row["request_json"]),
            result_json=json.loads(row["result_json"]),
            bank_snapshot_json=json.loads(row["bank_snapshot_json"]) if row["bank_snapshot_json"] else None,
            summary_json=json.loads(row["summary_json"]) if row["summary_json"] else None,
        )
    def get_latest_run(self, run_type: Optional[str] = None) -> Optional[StoredRun]:
        with self._connect() as conn:
            if run_type:
                row = conn.execute(
                    "SELECT * FROM runs WHERE run_type = ? ORDER BY created_at DESC LIMIT 1",
                    (run_type,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
        if not row:
            return None
        return StoredRun(
            run_id=row["run_id"],
            run_type=row["run_type"],
            created_at=row["created_at"],
            request_json=json.loads(row["request_json"]),
            result_json=json.loads(row["result_json"]),
            bank_snapshot_json=json.loads(row["bank_snapshot_json"]) if row["bank_snapshot_json"] else None,
            summary_json=json.loads(row["summary_json"]) if row["summary_json"] else None,
        )
    def upsert_bank_profiles(self, banks: list[Dict[str, Any]]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            for b in banks:
                bank_id = str(b.get("bank_id", ""))
                if not bank_id:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bank_profiles (bank_id, name, profile_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (bank_id, str(b.get("name", "")), json.dumps(b), now),
                )
    def get_bank_profile(self, bank_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM bank_profiles WHERE bank_id = ?",
                (bank_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["profile_json"])
    def find_bank_profile_by_name(self, name_query: str) -> Optional[Dict[str, Any]]:
        if not name_query:
            return None
        pattern = f"%{name_query.strip().lower()}%"
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json FROM bank_profiles
                WHERE LOWER(name) LIKE ?
                ORDER BY LENGTH(name) ASC
                LIMIT 1
                """,
                (pattern,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["profile_json"])
    def get_top_bank_profiles(self, limit: int = 10) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT profile_json FROM bank_profiles LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["profile_json"]) for r in rows]
    def get_bank_profile_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM bank_profiles").fetchone()
        return int(row["cnt"] if row else 0)
    def get_all_bank_profiles(self) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT profile_json FROM bank_profiles").fetchall()
        return [json.loads(r["profile_json"]) for r in rows]