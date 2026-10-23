"""SQLite-backed feedback storage for pypath-shiny."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STATUSES = ("open", "addressed", "wontfix", "duplicate")

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "feedback.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT,
    message TEXT NOT NULL,
    steps TEXT,
    rating INTEGER,
    page TEXT,
    app_version TEXT,
    browser TEXT,
    session_id TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_at TEXT,
    resolved_note TEXT,
    fix_reference TEXT,
    duplicate_of INTEGER
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve DB path using explicit arg, env var, then package default."""
    if db_path is not None:
        return Path(db_path)
    env = os.environ.get("PYPATH_SHINY_FEEDBACK_DB")
    return Path(env) if env else _DEFAULT_DB


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fall back to /tmp if the default location is not writable
        # This handles deployment scenarios where app user cannot write to app directory
        path = Path("/tmp") / "pypath-shiny-feedback" / "feedback.db"
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    return conn


def add_feedback(
    *,
    category: str,
    title: str,
    message: str,
    steps: str | None,
    rating: int | None,
    page: str | None,
    app_version: str | None,
    browser: str | None,
    session_id: str | None,
    db_path: str | Path | None = None,
) -> int:
    """Insert a feedback row and return its id."""
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            """
            INSERT INTO feedback (
                created_at, category, title, message, steps, rating, page,
                app_version, browser, session_id, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            (
                _now_iso(),
                category,
                title,
                message,
                steps,
                rating,
                page,
                app_version,
                browser,
                session_id,
            ),
        )
        return int(cur.lastrowid)


def list_feedback(
    *,
    statuses: tuple[str, ...] | list[str] | None = None,
    category: str | None = None,
    limit: int = 500,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List newest feedback rows with optional status/category filters."""
    clauses: list[str] = []
    params: list[Any] = []
    if statuses:
        marks = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({marks})")
        params.extend(statuses)
    if category and category != "all":
        clauses.append("category = ?")
        params.append(category)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT * FROM feedback {where} ORDER BY id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]


def get_feedback(
    feedback_id: int,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM feedback WHERE id = ?",
            (feedback_id,),
        ).fetchone()
        return dict(row) if row else None


def resolve_feedback(
    *,
    feedback_id: int,
    status: str,
    note: str,
    fix_reference: str | None = None,
    duplicate_of: int | None = None,
    db_path: str | Path | None = None,
) -> bool:
    """Resolve an open feedback item."""
    if status not in {"addressed", "wontfix", "duplicate"}:
        return False
    if status == "duplicate":
        if duplicate_of is None or duplicate_of == feedback_id:
            return False
        target = get_feedback(duplicate_of, db_path=db_path)
        if target is None or target.get("status") == "duplicate":
            return False

    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            """
            UPDATE feedback
               SET status = ?, resolved_at = ?, resolved_note = ?,
                   fix_reference = ?, duplicate_of = ?
             WHERE id = ? AND status = 'open'
            """,
            (
                status,
                _now_iso(),
                note,
                fix_reference if status == "addressed" else None,
                duplicate_of if status == "duplicate" else None,
                feedback_id,
            ),
        )
        return cur.rowcount == 1


def _parse_ts(value: str) -> datetime | None:
    try:
        ts = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def feedback_stats(*, db_path: str | Path | None = None) -> dict[str, int]:
    with closing(_connect(db_path)) as conn:
        total, open_items, bugs = conn.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(status='open'), 0),
                   COALESCE(SUM(category='bug'), 0)
              FROM feedback
            """
        ).fetchone()
        stamps = [r[0] for r in conn.execute("SELECT created_at FROM feedback")]

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    last_7_days = sum(
        1 for stamp in stamps if (ts := _parse_ts(stamp)) is not None and ts >= cutoff
    )

    return {
        "total": int(total),
        "open": int(open_items),
        "resolved": int(total - open_items),
        "bugs": int(bugs),
        "last_7_days": int(last_7_days),
    }


def weekly_feedback_counts(
    *,
    weeks: int = 8,
    db_path: str | Path | None = None,
) -> list[tuple[str, int]]:
    """Return weekly counts as ``[(YYYY-Www, n), ...]`` for recent weeks."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=max(1, int(weeks)))
    counts: dict[str, int] = {}

    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT created_at FROM feedback").fetchall()

    for row in rows:
        ts = _parse_ts(row[0])
        if ts is None or ts < start or ts > now:
            continue
        y, w, _ = ts.isocalendar()
        key = f"{y}-W{w:02d}"
        counts[key] = counts.get(key, 0) + 1

    return sorted(counts.items(), key=lambda x: x[0])
