import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def db_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with db_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                qa_id TEXT UNIQUE NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                context TEXT,
                evaluation_json TEXT,
                similarity_json TEXT,
                rating INTEGER,
                trace_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)"
        )


def ensure_session(db_path: Path, session_id: str, title: Optional[str] = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db_conn(db_path) as conn:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title or "New chat", now, now),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )


def store_message(
    *,
    db_path: Path,
    session_id: str,
    qa_id: str,
    question: str,
    answer: str,
    context: str,
    evaluation: dict,
    similarity: dict,
    trace_id: Optional[str],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages(
                session_id, qa_id, question, answer, context,
                evaluation_json, similarity_json, trace_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(qa_id) DO UPDATE SET
                answer=excluded.answer,
                context=excluded.context,
                evaluation_json=excluded.evaluation_json,
                similarity_json=excluded.similarity_json,
                trace_id=excluded.trace_id,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                qa_id,
                question,
                answer,
                context,
                json.dumps(evaluation or {}),
                json.dumps(similarity or {}),
                trace_id,
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def update_message_rating(db_path: Path, qa_id: str, rating: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db_conn(db_path) as conn:
        conn.execute(
            "UPDATE messages SET rating = ?, updated_at = ? WHERE qa_id = ?",
            (rating, now, qa_id),
        )


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}
