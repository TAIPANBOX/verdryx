"""SQLite persistence for eval runs, scores, unanswered cases, and baselines.

Mirrors engram/store.py's style: a thin wrapper around a single sqlite3
connection, an embedded DDL string run through executescript() on open, and
explicit conn.commit() calls after each write (no ORM, no migrations
framework -- CREATE TABLE IF NOT EXISTS is the whole migration story, same
as Engram's schema.py).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verdryx.models import (
    STATUS_FIX_PENDING,
    STATUS_OPEN,
    Baseline,
    EvalRun,
    Finding,
    Fix,
    Score,
    Unanswered,
)

_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id            TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    agent_id      TEXT,
    agent_version TEXT
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

-- A typed case typryx could not answer (models.Unanswered): never a Score,
-- counted apart with its own reason instead of failing the whole run --
-- @decided 2026-09-25, see features/typed-grader.feature. A NEW table, not
-- a widened column, so CREATE TABLE IF NOT EXISTS is the whole migration
-- story here exactly as it is for every other table above: an older store
-- gains this table empty the next time anything opens it, and a build that
-- predates this table simply never selects from it. See the SCHEMA_VERSION
-- comment below for why that column-vs-table distinction is what decides
-- whether the version stamp has to move, and for the one hazard this
-- leaves un-gated.
CREATE TABLE IF NOT EXISTS unanswered (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES eval_runs(id),
    case_id       TEXT NOT NULL,
    answer_id     TEXT NOT NULL,
    reason        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_unanswered_run_id ON unanswered(run_id);

-- A recorded quality failure (models.Finding). New tables, not widened
-- columns, for the reason the SCHEMA_VERSION comment gives for `unanswered`:
-- a build that predates them never names them. `evidence`, `detail` and a
-- fix's `result` are JSON text; `dedupe_key` is what makes the same failure
-- seen again one finding instead of a second row.
CREATE TABLE IF NOT EXISTS findings (
    id                TEXT PRIMARY KEY,
    dedupe_key        TEXT NOT NULL,
    agent_id          TEXT,
    kind              TEXT NOT NULL,
    sli               TEXT,
    window            TEXT NOT NULL,
    status            TEXT NOT NULL,
    opened_at         TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    seen_count        INTEGER NOT NULL DEFAULT 1,
    closed_at         TEXT,
    evidence          TEXT NOT NULL,
    detail            TEXT NOT NULL,
    suspect_model     TEXT,
    suspect_version   TEXT,
    suspect_since_run TEXT,
    suspect_note      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_dedupe_key ON findings(dedupe_key);

CREATE TABLE IF NOT EXISTS fixes (
    id            TEXT PRIMARY KEY,
    finding_id    TEXT NOT NULL REFERENCES findings(id),
    note          TEXT NOT NULL,
    version       TEXT,
    registered_at TEXT NOT NULL,
    status        TEXT NOT NULL,
    evaluated_at  TEXT,
    result        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fixes_finding_id ON fixes(finding_id);
"""


def _iso(dt: datetime) -> str:
    """Serialise a datetime to a canonical UTC isoformat for storage."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


#: Schema version stamped into `PRAGMA user_version`. Bump this only
#: together with a real change to `_DDL` that an older reader would
#: misinterpret.
#:
#: It did NOT move when `eval_runs.agent_id` arrived on 2026-08-26, and the
#: rule above is the reason rather than an exception to it. Nothing in this
#: module selects `*`: every read names its columns, so a build that predates
#: the column reads exactly the values it always read and cannot misinterpret
#: one it never asks for. Bumping would have made every such build refuse a
#: store the moment a single new run was written into it, and the Genaryx
#: console opens this file directly.
#:
#: The residual hazard is real and is named rather than hidden: an OLDER build
#: re-saving a run would erase its subject, because `save_run` is
#: `INSERT OR REPLACE` over a column it does not know about. It stays a
#: hazard and not a defect because nothing re-saves a run: `cli.run_eval`
#: mints `uuid.uuid4()` per run, so the only writer never presents an id that
#: already exists. If a rewrite path is ever added, that is the change that
#: earns the version bump, not this one.
#:
#: @claude, 2026-09-25: it did NOT move for the `unanswered` table either,
#: and for a different reason than the `agent_id` column above -- a new
#: TABLE, not a widened one. Nothing in this module selects `*` from
#: `eval_runs`, and an older build's own SQL never names `unanswered` at
#: all, so meeting that table on a newer-written store cannot be
#: misinterpreted the way meeting an unexpected COLUMN could be: the older
#: build simply never looks at it, exactly as it already never looks at any
#: table this module might add in the future. `CREATE TABLE IF NOT EXISTS`
#: is the whole migration story for a table the way it already is for the
#: three above it.
#:
#: The hazard this leaves is real and belongs beside the one above rather
#: than instead of it: the Genaryx console opens this file directly as its
#: quality plane (components.json), and a Genaryx build that predates this
#: change reads `eval_runs`/`scores` exactly as before and shows a mean with
#: no unanswered count at all -- not wrong, since that mean is still the
#: mean over scored cases, but silently partial for any run that had
#: unanswered cases. Nothing here can gate that; it is fixed only by
#: rebuilding the console image against this version of verdryx.
SCHEMA_VERSION = 1

#: Columns added to `eval_runs` after the table first shipped, in the order
#: they arrived, as `(name, type)`. `CREATE TABLE IF NOT EXISTS` is still the
#: whole migration story for a table, and it says nothing at all about a table
#: that already exists: an older store reaches this build one column short and
#: every read of the new name is an OperationalError on a file that was
#: perfectly good yesterday. So each entry here is applied with `ALTER TABLE`
#: when, and only when, the column is missing.
#:
#: Every column here must be nullable with no default. That is what makes a
#: row written before it read back as "no subject" rather than as a fabricated
#: one: SQLite fills existing rows with NULL, and NULL is the honest answer for
#: a run whose subject nobody recorded. A `DEFAULT` here would invent an
#: attribution for every historical run in one statement.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (("agent_id", "TEXT"), ("agent_version", "TEXT"))


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Widen `eval_runs` in place for a store written before a column existed.

    `ALTER TABLE ... ADD COLUMN` rather than a rebuild, and the difference
    matters more than the line count: rebuilding the table would drop and
    recreate it, and `baselines.eval_run_id` and `scores.run_id` both
    REFERENCE `eval_runs(id)`. With `PRAGMA foreign_keys=ON` (see
    `_configure_connection`) that is either a refusal or, worse, an orphaning,
    on a file whose only fault was being written last month.

    Guarded by `PRAGMA table_info` rather than by a caught exception, because
    "the column is already there" and "the ALTER failed for some other reason"
    are different facts and `sqlite3.OperationalError` reports them the same
    way.
    """
    present = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    if not present:
        # No such table yet, so `_DDL` is about to create it complete. Nothing
        # to widen, and an ALTER here would fail rather than measure nothing.
        return
    for name, sql_type in _ADDED_COLUMNS:
        if name not in present:
            # Not parameterisable: DDL does not take bound values. Both halves
            # are this module's own constants, never caller input.
            conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {name} {sql_type}")


def migrate(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes. Idempotent, safe to call repeatedly.

    Also stamps `PRAGMA user_version`, which was previously left at SQLite's
    default of 0. This does NOT introduce a migration framework: "CREATE
    TABLE IF NOT EXISTS is the whole migration story" still holds. What it
    introduces is the ability to have one later. A version stamp is the only
    part that cannot be retrofitted: stores already written without it are
    indistinguishable from fresh ones, so every day this is missing produces
    more files a future migration could not reason about.

    It also lets a reader fail closed. This file is read directly by other
    processes (the Genaryx console opens it as its quality plane), and a
    reader that meets a shape it does not know should say so rather than
    quietly return wrong rows, which is why a store stamped NEWER than this
    build understands is refused outright rather than opened.

    `_add_missing_columns` runs first, and the order is not arbitrary: it has
    to see the table as the older build left it, before `_DDL` has had a
    chance to be a no-op over it. On a fresh file it finds no table and does
    nothing, and `_DDL` then creates the complete shape in one statement.
    """
    _add_missing_columns(conn)
    conn.executescript(_DDL)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"this store's schema version is {current}, newer than this "
            f"verdryx understands ({SCHEMA_VERSION}); upgrade verdryx rather "
            "than risk misreading it"
        )
    if current != SCHEMA_VERSION:
        # Not parameterisable: PRAGMA does not take bound values. The
        # interpolated value is this module's own int constant.
        conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
    conn.commit()


def _configure_connection(conn: sqlite3.Connection, path: str) -> None:
    """Apply performance PRAGMAs, mirroring engram.core._configure_connection.

    Also turns on foreign-key enforcement (off by default per-connection in
    SQLite): baselines.eval_run_id and scores.run_id both REFERENCE
    eval_runs(id), so with this on, a baseline (or a score) can no longer be
    inserted pointing at a run id that does not exist, and -- were a
    delete/rename of an eval_runs row ever added -- it would be rejected
    while a baseline still references it. Without this, a baseline whose
    source run was never actually saved (e.g. a typo'd run id, or a
    hand-edited database) silently turns verdryx.cli's `drift` command's
    model filter into None, which then pools eval runs across every model
    instead of just the baseline's own -- see `_cmd_drift`.
    """
    conn.execute("PRAGMA foreign_keys=ON")
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
        """Insert or replace an eval run and all of its scores.

        `agent_id` is written as NULL when the run carries no subject, and
        `""` is normalised to the same NULL on the way in. One spelling of
        absence, deliberately: two of them in one column is how a later
        `WHERE agent_id IS NULL` silently misses half the rows it was written
        for. The consequence is visible and is tested: a run saved with
        `agent_id=""` loads back as `agent_id=None`.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_runs "
            "(id, model, started_at, finished_at, agent_id, agent_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.model,
                _iso(run.started_at),
                _iso(run.finished_at) if run.finished_at else None,
                run.agent_id or None,
                run.agent_version or None,
            ),
        )
        self._conn.execute("DELETE FROM scores WHERE run_id = ?", (run.id,))
        self._conn.executemany(
            "INSERT INTO scores (run_id, case_id, value, tokens, cost_usd) VALUES (?, ?, ?, ?, ?)",
            [(run.id, s.case_id, s.value, s.tokens, s.cost_usd) for s in run.scores],
        )
        self._conn.execute("DELETE FROM unanswered WHERE run_id = ?", (run.id,))
        self._conn.executemany(
            "INSERT INTO unanswered (run_id, case_id, answer_id, reason) VALUES (?, ?, ?, ?)",
            [(run.id, u.case_id, u.answer_id, u.reason) for u in run.unanswered],
        )
        self._conn.commit()

    def load_run(self, run_id: str) -> EvalRun | None:
        """Fetch a single eval run by id, with its scores, or None."""
        row: Any = self._conn.execute(
            "SELECT id, model, started_at, finished_at, agent_id, agent_version "
            "FROM eval_runs WHERE id = ?",
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
        sql = "SELECT id, model, started_at, finished_at, agent_id, agent_version FROM eval_runs"
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
        unanswered_rows: list[Any] = self._conn.execute(
            "SELECT case_id, answer_id, reason FROM unanswered WHERE run_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        unanswered = [
            Unanswered(case_id=r["case_id"], answer_id=r["answer_id"], reason=r["reason"])
            for r in unanswered_rows
        ]
        return EvalRun(
            id=row["id"],
            model=row["model"],
            started_at=_parse_iso(row["started_at"]),
            finished_at=_parse_iso(row["finished_at"]) if row["finished_at"] else None,
            scores=scores,
            # NULL stays None. A row written before the column existed cannot
            # know whose run it was, and anything filled in here (the model
            # name, a default agent, the first subject the store holds) would
            # attribute somebody's runs to an agent that never made them.
            agent_id=row["agent_id"],
            # NULL stays None here too: a run nobody labelled has no version,
            # and a made-up one would let a finding blame a build that never
            # ran.
            agent_version=row["agent_version"],
            # [] for a run that predates the `unanswered` table too -- a
            # store that old never had a case counted apart from a Score in
            # the first place, so an empty list is the honest read, not a
            # guess (same shape as agent_id=None above).
            unanswered=unanswered,
        )

    def score_records(self) -> list[dict[str, Any]]:
        """Every stored score, under the subject of the run that produced it.

        Shaped for `slo.compute_slo(scores=...)`, which is the only consumer,
        and the shape is a documented coupling rather than an accident: it is
        one SQL join, and doing it in the CLI would mean hydrating every run
        object to throw all of it away but two fields.

        **One record per CASE, not per run**, and that is the load-bearing
        choice here. A case is one thing a person asked the agent to do, which
        is the unit `slo` counts, the same way a tokenfuse run is one task
        rather than one call. Thresholding a run's `mean_score` instead would
        hide exactly the distribution the floor exists to price: `slo`'s own
        module docstring makes the argument, that a fleet scoring 0.9 every
        time and a fleet alternating 1.0 and 0.8 have the same mean and very
        different reliability, and only one of them embarrasses somebody.

        The subject is `""` when the run carries none, never absent and never
        guessed, because that is the value `slo.compute_slo` counts as
        unattributed and puts in nobody's numbers.

        `last_ts_millis` is the run's finish time, falling back to its start
        for a run that never finished. It is here so the quality burn rate can
        be measured at all: without a timestamp `slo._recent_burn` returns an
        honest zero, the report calls the window blind, and the indicator can
        compute a ratio it can never warn about. Note this is the EVAL clock,
        not the gateway's, so a report that mixes trace scores and stored ones
        is asking two sources what "recently" means. Both are wall-clock UTC
        milliseconds, which is what makes that answerable at all.
        """
        rows: list[Any] = self._conn.execute(
            "SELECT r.agent_id AS agent_id, s.value AS value, "
            "       r.id AS run_id, s.case_id AS case_id, "
            "       r.finished_at AS finished_at, r.started_at AS started_at "
            "FROM scores s JOIN eval_runs r ON s.run_id = r.id "
            "ORDER BY r.started_at, s.id"
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            when = _parse_iso(row["finished_at"] or row["started_at"])
            records.append(
                {
                    "agent_id": row["agent_id"] or "",
                    "score": row["value"],
                    # Which stored score this is, so a finding built on the
                    # quality floor can point at the cases that failed it.
                    "eval_run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "last_ts_millis": int(when.timestamp() * 1000),
                }
            )
        return records

    # ------------------------------------------------------------------
    # Findings and fixes
    # ------------------------------------------------------------------

    def add_finding(self, finding: Finding) -> None:
        self._conn.execute(
            "INSERT INTO findings (id, dedupe_key, agent_id, kind, sli, window, status, "
            "opened_at, last_seen_at, seen_count, closed_at, evidence, detail, "
            "suspect_model, suspect_version, suspect_since_run, suspect_note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._finding_row(finding),
        )
        self._conn.commit()

    def update_finding(self, finding: Finding) -> None:
        """Rewrite every column of an existing finding except its id."""
        row = self._finding_row(finding)
        self._conn.execute(
            "UPDATE findings SET dedupe_key=?, agent_id=?, kind=?, sli=?, window=?, status=?, "
            "opened_at=?, last_seen_at=?, seen_count=?, closed_at=?, evidence=?, detail=?, "
            "suspect_model=?, suspect_version=?, suspect_since_run=?, suspect_note=? "
            "WHERE id = ?",
            (*row[1:], finding.id),
        )
        self._conn.commit()

    @staticmethod
    def _finding_row(f: Finding) -> tuple[Any, ...]:
        return (
            f.id,
            f.dedupe_key,
            f.agent_id or None,
            f.kind,
            f.sli,
            f.window,
            f.status,
            _iso(f.opened_at),
            _iso(f.last_seen_at),
            f.seen_count,
            _iso(f.closed_at) if f.closed_at else None,
            json.dumps(f.evidence),
            json.dumps(f.detail, sort_keys=True),
            f.suspect_model,
            f.suspect_version,
            f.suspect_since_run,
            f.suspect_note,
        )

    def get_finding(self, finding_id: str) -> Finding | None:
        row: Any = self._conn.execute(
            "SELECT * FROM findings WHERE id = ?", (finding_id,)
        ).fetchone()
        return self._hydrate_finding(row) if row is not None else None

    def resolve_finding_id(self, prefix: str) -> str:
        """The full id a unique prefix names, or LookupError saying why not.

        Matched in Python rather than with LIKE, so a `%` or `_` typed by an
        operator is a character and not a wildcard.
        """
        ids = [r["id"] for r in self._conn.execute("SELECT id FROM findings ORDER BY id")]
        matches = [i for i in ids if prefix and i.startswith(prefix)]
        if not matches:
            raise LookupError(f"no finding with an id starting {prefix!r}")
        if len(matches) > 1:
            raise LookupError(f"{len(matches)} findings start with {prefix!r}; give more of the id")
        return str(matches[0])

    def list_findings(
        self,
        statuses: tuple[str, ...] | list[str] | None = (STATUS_OPEN, STATUS_FIX_PENDING),
        agent_id: str | None = None,
    ) -> list[Finding]:
        """Findings, most recently seen first. `statuses=None` means every status."""
        sql = "SELECT * FROM findings"
        clauses: list[str] = []
        params: list[Any] = []
        if statuses is not None:
            clauses.append(f"status IN ({', '.join('?' for _ in statuses)})")
            params.extend(statuses)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY last_seen_at DESC, id"
        rows: list[Any] = self._conn.execute(sql, params).fetchall()
        return [self._hydrate_finding(r) for r in rows]

    def find_active_by_key(self, dedupe_key: str) -> Finding | None:
        """The open or fix-pending finding for this key, if there is one.

        A closed finding is history, not a match: the same failure after a
        verified fix is a new finding.
        """
        row: Any = self._conn.execute(
            "SELECT * FROM findings WHERE dedupe_key = ? AND status IN (?, ?) "
            "ORDER BY opened_at DESC LIMIT 1",
            (dedupe_key, STATUS_OPEN, STATUS_FIX_PENDING),
        ).fetchone()
        return self._hydrate_finding(row) if row is not None else None

    def fix_horizon(self, dedupe_key: str) -> datetime | None:
        """When the newest fix for this key was registered, or None for never.

        Runs from before that moment are the ones the fix was written against.
        They stay in every pooled window for a while, and without this line a
        verified fix would be undone by the runs it was verified against.
        """
        rows: list[Any] = self._conn.execute(
            "SELECT x.registered_at AS at FROM fixes x JOIN findings f ON x.finding_id = f.id "
            "WHERE f.dedupe_key = ?",
            (dedupe_key,),
        ).fetchall()
        times = [_parse_iso(r["at"]) for r in rows]
        return max(times) if times else None

    def add_fix(self, fix: Fix) -> None:
        self._conn.execute(
            "INSERT INTO fixes (id, finding_id, note, version, registered_at, status, "
            "evaluated_at, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fix.id,
                fix.finding_id,
                fix.note,
                fix.version,
                _iso(fix.registered_at),
                fix.status,
                _iso(fix.evaluated_at) if fix.evaluated_at else None,
                json.dumps(fix.result, sort_keys=True),
            ),
        )
        self._conn.commit()

    def update_fix(self, fix: Fix) -> None:
        self._conn.execute(
            "UPDATE fixes SET status = ?, evaluated_at = ?, result = ? WHERE id = ?",
            (
                fix.status,
                _iso(fix.evaluated_at) if fix.evaluated_at else None,
                json.dumps(fix.result, sort_keys=True),
                fix.id,
            ),
        )
        self._conn.commit()

    def _hydrate_finding(self, row: Any) -> Finding:
        fix_rows: list[Any] = self._conn.execute(
            "SELECT * FROM fixes WHERE finding_id = ? ORDER BY registered_at, id", (row["id"],)
        ).fetchall()
        fixes = [
            Fix(
                id=r["id"],
                finding_id=r["finding_id"],
                note=r["note"],
                registered_at=_parse_iso(r["registered_at"]),
                version=r["version"],
                status=r["status"],
                evaluated_at=_parse_iso(r["evaluated_at"]) if r["evaluated_at"] else None,
                result=json.loads(r["result"]),
            )
            for r in fix_rows
        ]
        return Finding(
            id=row["id"],
            dedupe_key=row["dedupe_key"],
            agent_id=row["agent_id"],
            kind=row["kind"],
            sli=row["sli"],
            window=row["window"],
            status=row["status"],
            opened_at=_parse_iso(row["opened_at"]),
            last_seen_at=_parse_iso(row["last_seen_at"]),
            seen_count=row["seen_count"],
            evidence=json.loads(row["evidence"]),
            detail=json.loads(row["detail"]),
            suspect_model=row["suspect_model"],
            suspect_version=row["suspect_version"],
            suspect_since_run=row["suspect_since_run"],
            suspect_note=row["suspect_note"],
            closed_at=_parse_iso(row["closed_at"]) if row["closed_at"] else None,
            fixes=fixes,
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
