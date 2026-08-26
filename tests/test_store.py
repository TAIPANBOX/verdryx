"""Tests for verdryx.store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from verdryx.models import Baseline, EvalRun, Score
from verdryx.store import SCHEMA_VERSION, Store


def _run(run_id: str = "r1", model: str = "stub") -> EvalRun:
    return EvalRun(
        id=run_id,
        model=model,
        started_at=datetime(2026, 7, 1, tzinfo=UTC),
        finished_at=datetime(2026, 7, 1, 0, 1, tzinfo=UTC),
        scores=[
            Score(case_id="c1", value=1.0, tokens=10, cost_usd=0.01),
            Score(case_id="c2", value=0.5, tokens=20, cost_usd=0.02),
        ],
    )


# ------------------------------------------------------------------
# open / lifecycle
# ------------------------------------------------------------------


def test_open_in_memory_creates_empty_tables() -> None:
    with Store.open(":memory:") as store:
        assert store.list_runs() == []
        assert store.list_baselines() == []


def test_open_default_path_is_in_memory() -> None:
    with Store.open() as store:
        assert store.list_runs() == []


def test_open_is_idempotent_on_an_existing_file(tmp_path) -> None:
    db_path = tmp_path / "store.db"
    with Store.open(db_path) as store:
        store.save_run(_run())
    # Re-opening the same file must not fail or wipe existing data
    # (CREATE TABLE IF NOT EXISTS, not CREATE TABLE).
    with Store.open(db_path) as store:
        assert len(store.list_runs()) == 1


# ------------------------------------------------------------------
# Eval runs
# ------------------------------------------------------------------


def test_save_and_load_run_round_trips(tmp_path) -> None:
    db_path = tmp_path / "store.db"
    run = _run()
    with Store.open(db_path) as store:
        store.save_run(run)
    with Store.open(db_path) as store:
        loaded = store.load_run("r1")
    assert loaded == run


def test_load_run_missing_returns_none() -> None:
    with Store.open(":memory:") as store:
        assert store.load_run("does-not-exist") is None


def test_save_run_replaces_existing_scores() -> None:
    with Store.open(":memory:") as store:
        run = _run()
        store.save_run(run)
        updated = EvalRun(
            id="r1",
            model="stub",
            started_at=run.started_at,
            finished_at=run.finished_at,
            scores=[Score(case_id="only-one", value=0.9)],
        )
        store.save_run(updated)
        loaded = store.load_run("r1")
    assert loaded is not None
    assert [s.case_id for s in loaded.scores] == ["only-one"]


def test_save_run_with_no_finished_at_round_trips_as_none() -> None:
    with Store.open(":memory:") as store:
        run = EvalRun(id="r1", model="stub", started_at=datetime(2026, 7, 1, tzinfo=UTC))
        store.save_run(run)
        loaded = store.load_run("r1")
    assert loaded is not None
    assert loaded.finished_at is None


def test_list_runs_orders_most_recent_first_and_filters_by_model() -> None:
    with Store.open(":memory:") as store:
        store.save_run(EvalRun(id="r1", model="a", started_at=datetime(2026, 1, 1, tzinfo=UTC)))
        store.save_run(EvalRun(id="r2", model="a", started_at=datetime(2026, 1, 2, tzinfo=UTC)))
        store.save_run(EvalRun(id="r3", model="b", started_at=datetime(2026, 1, 3, tzinfo=UTC)))

        all_runs = store.list_runs()
        assert [r.id for r in all_runs] == ["r3", "r2", "r1"]

        model_a = store.list_runs(model="a")
        assert [r.id for r in model_a] == ["r2", "r1"]

        limited = store.list_runs(limit=1)
        assert [r.id for r in limited] == ["r3"]

        none_match = store.list_runs(model="does-not-exist")
        assert none_match == []


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------


def test_set_and_get_baseline_round_trips() -> None:
    with Store.open(":memory:") as store:
        store.save_run(_run())
        baseline = Baseline(
            id="b1",
            eval_run_id="r1",
            mean_score=0.75,
            created_at=datetime(2026, 7, 2, tzinfo=UTC),
            label="v1",
        )
        store.set_baseline(baseline)
        loaded = store.get_baseline("b1")
    assert loaded == baseline


def test_get_baseline_missing_returns_none() -> None:
    with Store.open(":memory:") as store:
        assert store.get_baseline("nope") is None


def test_set_baseline_replaces_existing() -> None:
    with Store.open(":memory:") as store:
        store.save_run(_run())
        store.set_baseline(
            Baseline(
                id="b1",
                eval_run_id="r1",
                mean_score=0.5,
                created_at=datetime(2026, 7, 2, tzinfo=UTC),
            )
        )
        store.set_baseline(
            Baseline(
                id="b1",
                eval_run_id="r1",
                mean_score=0.9,
                created_at=datetime(2026, 7, 3, tzinfo=UTC),
                label="updated",
            )
        )
        loaded = store.get_baseline("b1")
    assert loaded is not None
    # SQLite's REAL column is an 8-byte IEEE-754 double, same as Python's
    # float, so this literal round-trips exactly -- no pytest.approx needed.
    assert loaded.mean_score == 0.9
    assert loaded.label == "updated"


def test_list_baselines_most_recent_first() -> None:
    with Store.open(":memory:") as store:
        store.save_run(_run())
        store.set_baseline(
            Baseline(
                id="b1",
                eval_run_id="r1",
                mean_score=0.5,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        store.set_baseline(
            Baseline(
                id="b2",
                eval_run_id="r1",
                mean_score=0.6,
                created_at=datetime(2026, 1, 2, tzinfo=UTC),
            )
        )
        baselines = store.list_baselines()
    assert [b.id for b in baselines] == ["b2", "b1"]


def test_baseline_survives_reopen(tmp_path) -> None:
    db_path = tmp_path / "store.db"
    with Store.open(db_path) as store:
        store.save_run(_run())
        store.set_baseline(
            Baseline(
                id="b1",
                eval_run_id="r1",
                mean_score=0.42,
                created_at=datetime(2026, 7, 2, tzinfo=UTC),
            )
        )
    with Store.open(db_path) as store:
        loaded = store.get_baseline("b1")
    assert loaded is not None
    assert loaded.mean_score == 0.42


# ------------------------------------------------------------------
# schema version
# ------------------------------------------------------------------


def _user_version(db_path) -> int:
    raw = sqlite3.connect(str(db_path))
    try:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])
    finally:
        raw.close()


def test_new_store_is_stamped_with_the_schema_version(tmp_path) -> None:
    db_path = tmp_path / "store.db"
    with Store.open(db_path):
        pass
    assert _user_version(db_path) == SCHEMA_VERSION


def test_a_store_written_before_stamping_is_upgraded_in_place(tmp_path) -> None:
    """Version 0 is what every store written before this existed looks like.

    It is indistinguishable from a fresh one (same tables), so it is simply
    stamped on the next open rather than treated as foreign.
    """
    db_path = tmp_path / "store.db"
    with Store.open(db_path) as store:
        store.save_run(_run())
    raw = sqlite3.connect(str(db_path))
    raw.execute("PRAGMA user_version = 0")
    raw.commit()
    raw.close()

    with Store.open(db_path) as store:
        assert store.load_run("r1") is not None
    assert _user_version(db_path) == SCHEMA_VERSION


def test_a_newer_store_is_refused_rather_than_misread(tmp_path) -> None:
    """Fail closed: this file is read by other processes too.

    The Genaryx console opens this store directly as its quality plane, so a
    build meeting a shape it does not know must say so instead of quietly
    returning rows it may be reading wrong.
    """
    db_path = tmp_path / "store.db"
    with Store.open(db_path):
        pass
    raw = sqlite3.connect(str(db_path))
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()

    with pytest.raises(RuntimeError, match="newer than this verdryx"):
        Store.open(db_path)


# ------------------------------------------------------------------
# The subject of a run
# ------------------------------------------------------------------

#: `eval_runs` exactly as it stood before a run had a subject. Every store
#: written by a build before 2026-08-26 has this shape on somebody's disk, and
#: the only honest way to test the migration is to produce one rather than to
#: describe it.
_PRE_SUBJECT_DDL = """
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

CREATE TABLE IF NOT EXISTS baselines (
    id            TEXT PRIMARY KEY,
    eval_run_id   TEXT NOT NULL REFERENCES eval_runs(id),
    mean_score    REAL NOT NULL,
    created_at    TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT ''
);
"""


def _write_pre_subject_store(db_path) -> None:
    """A store as an older verdryx left it: no `agent_id`, one run, two scores."""
    raw = sqlite3.connect(str(db_path))
    try:
        raw.executescript(_PRE_SUBJECT_DDL)
        raw.execute(
            "INSERT INTO eval_runs (id, model, started_at, finished_at) VALUES (?, ?, ?, ?)",
            ("old-run", "stub", "2026-07-01T00:00:00+00:00", "2026-07-01T00:01:00+00:00"),
        )
        raw.executemany(
            "INSERT INTO scores (run_id, case_id, value, tokens, cost_usd) VALUES (?, ?, ?, ?, ?)",
            [("old-run", "c1", 1.0, 10, 0.01), ("old-run", "c2", 0.5, 20, 0.02)],
        )
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()


def test_an_eval_run_round_trips_its_subject(tmp_path) -> None:
    """The subject is the whole point of the column: it has to survive a write."""
    db_path = tmp_path / "store.db"
    run = _run()
    run.agent_id = "agent://acme.example/support/bot"
    with Store.open(db_path) as store:
        store.save_run(run)
    with Store.open(db_path) as store:
        loaded = store.load_run("r1")
    assert loaded is not None
    assert loaded.agent_id == "agent://acme.example/support/bot"


def test_a_store_written_before_the_subject_existed_still_opens(tmp_path) -> None:
    """CREATE TABLE IF NOT EXISTS does not add a column to a table that exists.

    An old store on somebody's disk therefore reaches the new build one column
    short, and if nothing widens it every read of `agent_id` is an
    OperationalError on a file that was perfectly good yesterday.
    """
    db_path = tmp_path / "store.db"
    _write_pre_subject_store(db_path)
    with Store.open(db_path) as store:
        loaded = store.load_run("old-run")
    assert loaded is not None
    assert [s.case_id for s in loaded.scores] == ["c1", "c2"]


def test_a_row_written_before_the_column_reads_back_as_no_subject(tmp_path) -> None:
    """Not as a fabricated one.

    The other half of the migration, and the half that fails quietly. A row
    that predates the column cannot know whose run it was, and anything that
    filled it in (the model name, a default agent, the first subject seen)
    would attribute somebody's runs to an agent that never made them.
    """
    db_path = tmp_path / "store.db"
    _write_pre_subject_store(db_path)
    with Store.open(db_path) as store:
        loaded = store.load_run("old-run")
    assert loaded is not None
    assert loaded.agent_id is None


def test_the_migration_does_not_lose_the_baselines_either(tmp_path) -> None:
    """A migration that rebuilt `eval_runs` would take its foreign keys with it."""
    db_path = tmp_path / "store.db"
    _write_pre_subject_store(db_path)
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "INSERT INTO baselines (id, eval_run_id, mean_score, created_at, label) "
        "VALUES (?, ?, ?, ?, ?)",
        ("b-old", "old-run", 0.75, "2026-07-02T00:00:00+00:00", "v1"),
    )
    raw.commit()
    raw.close()

    with Store.open(db_path) as store:
        baseline = store.get_baseline("b-old")
        assert baseline is not None
        assert baseline.eval_run_id == "old-run"
        assert store.load_run("old-run") is not None


def test_the_migration_runs_once_and_re_opening_is_still_idempotent(tmp_path) -> None:
    """A second open must not try to add the column again, which SQLite refuses."""
    db_path = tmp_path / "store.db"
    _write_pre_subject_store(db_path)
    for _ in range(3):
        with Store.open(db_path) as store:
            assert store.load_run("old-run") is not None


def test_an_empty_subject_is_stored_as_the_one_spelling_of_absence(tmp_path) -> None:
    """`""` and `None` both mean "no subject", and the column keeps one of them.

    Two spellings of absence in one column is how a later `WHERE agent_id IS
    NULL` silently misses half the rows it was written for.
    """
    db_path = tmp_path / "store.db"
    run = _run()
    run.agent_id = ""
    with Store.open(db_path) as store:
        store.save_run(run)
    raw = sqlite3.connect(str(db_path))
    try:
        stored = raw.execute("SELECT agent_id FROM eval_runs WHERE id = 'r1'").fetchone()[0]
    finally:
        raw.close()
    assert stored is None
    with Store.open(db_path) as store:
        loaded = store.load_run("r1")
    assert loaded is not None
    assert loaded.agent_id is None


def test_list_runs_carries_the_subject_too() -> None:
    """`load_run` and `list_runs` read the same table and must agree about it."""
    with Store.open(":memory:") as store:
        run = _run()
        run.agent_id = "agent://acme.example/support/bot"
        store.save_run(run)
        listed = store.list_runs()
    assert [r.agent_id for r in listed] == ["agent://acme.example/support/bot"]


# ------------------------------------------------------------------
# Score records, the join the SLO plane could not make
# ------------------------------------------------------------------


def test_score_records_carry_one_record_per_case_under_the_run_s_subject() -> None:
    """One record per CASE, not per run.

    A case is the unit a person asked the agent to do, which is the unit
    `slo` counts. Thresholding a run's mean would hide exactly the
    distribution the floor exists to price.
    """
    with Store.open(":memory:") as store:
        run = _run()
        run.agent_id = "agent://acme.example/support/bot"
        store.save_run(run)
        records = store.score_records()
    assert [r["score"] for r in records] == [1.0, 0.5]
    assert {r["agent_id"] for r in records} == {"agent://acme.example/support/bot"}


def test_score_records_from_an_unattributed_run_carry_an_empty_subject() -> None:
    """Empty, never absent and never guessed: `slo` counts these and buckets
    none of them."""
    with Store.open(":memory:") as store:
        store.save_run(_run())
        records = store.score_records()
    assert records
    assert {r["agent_id"] for r in records} == {""}


def test_score_records_carry_the_run_s_clock() -> None:
    """Without a timestamp the quality burn rate can never fire, and a warning
    that can never fire reports exactly like one with nothing to report."""
    with Store.open(":memory:") as store:
        run = _run()
        run.agent_id = "agent://acme.example/support/bot"
        store.save_run(run)
        records = store.score_records()
    finished_millis = int(datetime(2026, 7, 1, 0, 1, tzinfo=UTC).timestamp() * 1000)
    assert {r["last_ts_millis"] for r in records} == {finished_millis}


def test_score_records_fall_back_to_the_start_when_a_run_never_finished() -> None:
    """An unfinished run still happened at a knowable time."""
    with Store.open(":memory:") as store:
        store.save_run(
            EvalRun(
                id="r1",
                model="stub",
                started_at=datetime(2026, 7, 1, tzinfo=UTC),
                agent_id="agent://acme.example/support/bot",
                scores=[Score(case_id="c1", value=1.0)],
            )
        )
        records = store.score_records()
    assert records[0]["last_ts_millis"] == int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1000)
