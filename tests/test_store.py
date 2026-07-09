"""Tests for verdryx.store."""

from __future__ import annotations

from datetime import UTC, datetime

from verdryx.models import Baseline, EvalRun, Score
from verdryx.store import Store


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
