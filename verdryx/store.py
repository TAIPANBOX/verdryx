"""SQLite persistence for eval runs, scores, and baselines.

Mirrors engram/store.py's style: a thin wrapper around a single sqlite3
connection, an embedded DDL string run through executescript() on open, and
explicit conn.commit() calls after each write (no ORM, no migrations
framework -- CREATE TABLE IF NOT EXISTS is the whole migration story, same
as Engram's schema.py).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verdryx.models import Baseline, EvalRun, Score

_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id            TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES eval_runs(id),
    case_id       TEXT NOT NULL,
    value         REAL NOT NULL,
    tokens        INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_scores_run_id ON scores(run_id);

CREATE TABLE IF NOT EXISTS baselines (
    id            TEXT PRIMARY KEY,
    eval_run_id   TEXT NOT NULL REFERENCES eval_runs(id),
    mean_score    REAL NOT NULL,
    created_at    TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT ''
);
"""


def _iso(dt: datetime) -> str:
    """Serialise a datetime to a canonical UTC isoformat for storage."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def migrate(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes. Idempotent, safe to call repeatedly."""
    conn.executescript(_DDL)
    conn.commit()


def _configure_connection(conn: sqlite3.Connection, path: str) -> None:
    """Apply performance PRAGMAs, mirroring engram.core._configure_connection."""
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")


class Store:
    """Thin wrapper around a sqlite3 connection providing eval-run persistence."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, path: str | Path = ":memory:") -> Store:
        """Open (creating if needed) a SQLite store at `path`, running migrations."""
        path_str = str(path)
        conn = sqlite3.connect(path_str, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _configure_connection(conn, path_str)
        migrate(conn)
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Eval runs
    # ------------------------------------------------------------------

    def save_run(self, run: EvalRun) -> None:
        """Insert or replace an eval run and all of its scores."""
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_runs (id, model, started_at, finished_at) "
            "VALUES (?, ?, ?, ?)",
            (
                run.id,
                run.model,
                _iso(run.started_at),
                _iso(run.finished_at) if run.finished_at else None,
            ),
        )
        self._conn.execute("DELETE FROM scores WHERE run_id = ?", (run.id,))
        self._conn.executemany(
            "INSERT INTO scores (run_id, case_id, value, tokens, cost_usd) VALUES (?, ?, ?, ?, ?)",
            [(run.id, s.case_id, s.value, s.tokens, s.cost_usd) for s in run.scores],
        )
        self._conn.commit()

    def load_run(self, run_id: str) -> EvalRun | None:
        """Fetch a single eval run by id, with its scores, or None."""
        row: Any = self._conn.execute(
            "SELECT id, model, started_at, finished_at FROM eval_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate_run(row)

    def list_runs(self, model: str | None = None, limit: int | None = None) -> list[EvalRun]:
        """List eval runs, most recently started first.

        Args:
            model: If set, restrict to runs of this model.
            limit: If set, return at most this many runs.
        """
        sql = "SELECT id, model, started_at, finished_at FROM eval_runs"
        params: tuple[Any, ...] = ()
        if model is not None:
            sql += " WHERE model = ?"
            params = (model,)
        sql += " ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        rows: list[Any] = self._conn.execute(sql, params).fetchall()
        return [self._hydrate_run(row) for row in rows]

    def _hydrate_run(self, row: Any) -> EvalRun:
        score_rows: list[Any] = self._conn.execute(
            "SELECT case_id, value, tokens, cost_usd FROM scores WHERE run_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        scores = [
            Score(
                case_id=r["case_id"], value=r["value"], tokens=r["tokens"], cost_usd=r["cost_usd"]
            )
            for r in score_rows
        ]
        return EvalRun(
            id=row["id"],
            model=row["model"],
            started_at=_parse_iso(row["started_at"]),
            finished_at=_parse_iso(row["finished_at"]) if row["finished_at"] else None,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def set_baseline(self, baseline: Baseline) -> None:
        """Insert or replace a baseline."""
        self._conn.execute(
            "INSERT OR REPLACE INTO baselines (id, eval_run_id, mean_score, created_at, label) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                baseline.id,
                baseline.eval_run_id,
                baseline.mean_score,
                _iso(baseline.created_at),
                baseline.label,
            ),
        )
        self._conn.commit()

    def get_baseline(self, baseline_id: str) -> Baseline | None:
        row: Any = self._conn.execute(
            "SELECT id, eval_run_id, mean_score, created_at, label FROM baselines WHERE id = ?",
            (baseline_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate_baseline(row)

    def list_baselines(self) -> list[Baseline]:
        """List all baselines, most recently created first."""
        rows: list[Any] = self._conn.execute(
            "SELECT id, eval_run_id, mean_score, created_at, label "
            "FROM baselines ORDER BY created_at DESC"
        ).fetchall()
        return [self._hydrate_baseline(row) for row in rows]

    @staticmethod
    def _hydrate_baseline(row: Any) -> Baseline:
        return Baseline(
            id=row["id"],
            eval_run_id=row["eval_run_id"],
            mean_score=row["mean_score"],
            created_at=_parse_iso(row["created_at"]),
            label=row["label"] or "",
        )
