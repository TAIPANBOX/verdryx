"""Human grades: a person's grade outweighs the model's, and carries its author.

Two roles are registered in the store: a `reviewer` can grade, an `uploader`
cannot. `eval` needs no credential and has no path to a human grade at all.
"""

from __future__ import annotations

import math
import os
import stat
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.findings_support import (
    AGENT,
    BEFORE,
    good,
    make_run,
    ms,
    regressed_store,
    seed,
    trace_file,
)
from verdryx.cli import main
from verdryx.models import EvalRun, Unanswered
from verdryx.store import Store


def _principal(tmp_path: Path, db: Path, name: str, role: str = "reviewer") -> Path:
    key = tmp_path / f"{name}.key"
    main(["principal", "add", name, "--role", role, "--key-out", str(key), "--db", str(db)])
    return key


def _grade(db: Path, run_id: str, case_id: str, value: str, author: str, key: Path, *extra: str):
    main(
        [
            "grade",
            run_id,
            case_id,
            value,
            "--author",
            author,
            "--key-file",
            str(key),
            "--db",
            str(db),
            *extra,
        ]
    )


def _store_with_run(tmp_path: Path, values: list[float] | None = None) -> tuple[Path, EvalRun]:
    db = tmp_path / "store.db"
    run = make_run(values or good(4), started=BEFORE, run_id="run-1")
    seed(db, [run])
    return db, run


def _load(db: Path, run_id: str) -> EvalRun:
    with Store.open(db) as store:
        run = store.load_run(run_id)
    assert run is not None
    return run


def _grades(db: Path, run_id: str, case_id: str | None = None) -> list[Any]:
    with Store.open(db) as store:
        return store.human_grades(run_id, case_id)


def test_a_reviewer_grade_is_stored_with_its_author(tmp_path) -> None:
    db, run = _store_with_run(tmp_path)
    key = _principal(tmp_path, db, "alice")

    _grade(db, run.id, "c0", "0.0", "alice", key, "--note", "refunded the wrong order")

    (grade,) = _grades(db, run.id)
    assert (grade.author, grade.case_id, grade.value) == ("alice", "c0", 0.0)
    assert grade.note == "refunded the wrong order"
    assert grade.created_at is not None


def test_a_human_grade_outweighs_the_model_in_the_run_mean(tmp_path) -> None:
    db, run = _store_with_run(tmp_path, good(4))
    key = _principal(tmp_path, db, "alice")
    assert _load(db, run.id).mean_score == pytest.approx(1.0)

    _grade(db, run.id, "c0", "0.0", "alice", key)
    _grade(db, run.id, "c1", "0.0", "alice", key)

    loaded = _load(db, run.id)
    assert loaded.mean_score == pytest.approx(0.5)
    assert loaded.human_graded == 2
    graded = {s.case_id: s for s in loaded.scores}
    assert (graded["c0"].value, graded["c0"].model_value, graded["c0"].graded_by) == (
        0.0,
        1.0,
        "alice",
    )
    assert (graded["c2"].graded_by, graded["c2"].model_value) == (None, None)


def test_a_human_grade_outweighs_the_model_in_drift(tmp_path, capsys) -> None:
    db = tmp_path / "store.db"
    base = make_run(good(20), started=BEFORE, run_id="base")
    newer = make_run(good(20), started=BEFORE + timedelta(hours=1), run_id="newer")
    seed(db, [base, newer], baseline_run=base)
    main(["drift", "--baseline", "b1", "--db", str(db)])
    assert "on-track" in capsys.readouterr().out
    key = _principal(tmp_path, db, "alice")
    for i in range(10):
        _grade(db, "newer", f"c{i}", "0.0", "alice", key)
    capsys.readouterr()

    main(["drift", "--baseline", "b1", "--db", str(db)])

    out = capsys.readouterr().out
    assert "regressed" in out
    assert "10 of 20" in out and "graded by a person" in out
    with Store.open(db) as store:
        (finding,) = store.list_findings(statuses=None)
    assert finding.detail["human_graded"] == 10


def test_a_human_grade_outweighs_the_model_in_the_quality_floor(tmp_path) -> None:
    db = tmp_path / "store.db"
    run = make_run(good(40), started=BEFORE, run_id="scored", agent=AGENT)
    seed(db, [run])
    trace = trace_file(tmp_path, ["case_resolved"] * 20, start_ms=ms(BEFORE))
    args = [
        "slo",
        "--input",
        str(trace),
        "--scores-db",
        str(db),
        "--min-events",
        "10",
        "--db",
        str(db),
    ]
    main(args)
    with Store.open(db) as store:
        assert store.list_findings(statuses=None) == []
    key = _principal(tmp_path, db, "alice")
    for i in range(20):
        _grade(db, "scored", f"c{i}", "0.0", "alice", key)

    main(args)

    with Store.open(db) as store:
        floor = [f for f in store.list_findings(statuses=None) if f.sli == "quality_floor"]
    (finding,) = floor
    assert finding.agent_id == AGENT
    assert finding.detail["good"] == 20 and finding.detail["events"] == 40
    assert set(finding.evidence) == {f"eval:scored:c{i}" for i in range(20)}


def test_a_principal_who_only_uploads_cannot_grade(tmp_path, capsys) -> None:
    db, run = _store_with_run(tmp_path)
    key = _principal(tmp_path, db, "ingest-bot", role="uploader")
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        _grade(db, run.id, "c0", "0.0", "ingest-bot", key)

    assert exc.value.code == 1
    assert "only a reviewer can grade" in capsys.readouterr().err
    assert _grades(db, run.id) == []
    assert _load(db, run.id).mean_score == pytest.approx(1.0)


def test_a_grade_with_no_credential_or_a_wrong_one_is_refused(
    tmp_path, capsys, monkeypatch
) -> None:
    db, run = _store_with_run(tmp_path)
    right = _principal(tmp_path, db, "alice")
    wrong = tmp_path / "wrong.key"
    wrong.write_text("not-alices-key\n")
    monkeypatch.delenv("VERDRYX_REVIEWER_KEY_FILE", raising=False)
    capsys.readouterr()

    with pytest.raises(SystemExit):
        main(["grade", run.id, "c0", "0.0", "--author", "alice", "--db", str(db)])
    assert "credential" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _grade(db, run.id, "c0", "0.0", "alice", wrong)
    assert "not recognised" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _grade(db, run.id, "c0", "0.0", "mallory", right)
    assert "not recognised" in capsys.readouterr().err

    assert _grades(db, run.id) == []


def test_the_key_file_can_come_from_the_environment(tmp_path, monkeypatch) -> None:
    db, run = _store_with_run(tmp_path)
    key = _principal(tmp_path, db, "alice")
    monkeypatch.setenv("VERDRYX_REVIEWER_KEY_FILE", str(key))

    main(["grade", run.id, "c0", "0.25", "--author", "alice", "--db", str(db)])

    assert [g.value for g in _grades(db, run.id)] == [0.25]


def test_a_reviewer_cannot_sign_with_another_reviewers_name(tmp_path, capsys) -> None:
    db, run = _store_with_run(tmp_path)
    alice_key = _principal(tmp_path, db, "alice")
    bob_key = _principal(tmp_path, db, "bob")
    assert alice_key != bob_key
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _grade(db, run.id, "c0", "0.0", "alice", bob_key)

    assert "not recognised" in capsys.readouterr().err
    assert _grades(db, run.id) == []


def test_a_later_human_grade_supersedes_the_earlier_one_and_both_stay_on_record(tmp_path) -> None:
    db, run = _store_with_run(tmp_path)
    alice = _principal(tmp_path, db, "alice")
    bob = _principal(tmp_path, db, "bob")
    _grade(db, run.id, "c0", "0.0", "alice", alice)

    _grade(db, run.id, "c0", "0.75", "bob", bob)

    graded = {s.case_id: s for s in _load(db, run.id).scores}["c0"]
    assert (graded.value, graded.graded_by, graded.model_value) == (0.75, "bob", 1.0)
    history = _grades(db, run.id, "c0")
    assert [(g.author, g.value) for g in history] == [("alice", 0.0), ("bob", 0.75)]


@pytest.mark.parametrize(
    ("case_id", "value", "why"),
    [
        ("never-asked", "0.5", "no such case"),
        ("unanswered-1", "0.5", "no answer"),
        ("c0", "1.5", "between 0 and 1"),
        ("c0", "-0.1", "between 0 and 1"),
        ("c0", "nan", "between 0 and 1"),
    ],
)
def test_a_grade_that_cannot_be_given_is_refused(tmp_path, capsys, case_id, value, why) -> None:
    db = tmp_path / "store.db"
    run = make_run(good(2), started=BEFORE, run_id="run-1")
    run.unanswered.append(Unanswered(case_id="unanswered-1", answer_id="a1", reason="mass"))
    seed(db, [run])
    key = _principal(tmp_path, db, "alice")
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _grade(db, run.id, case_id, value, "alice", key)

    assert why in capsys.readouterr().err
    assert _grades(db, run.id) == []
    assert not math.isnan(_load(db, run.id).mean_score)


def test_principal_add_writes_the_key_once_and_the_store_keeps_only_its_hash(
    tmp_path, capsys
) -> None:
    db = tmp_path / "store.db"
    key_path = tmp_path / "alice.key"

    main(
        [
            "principal",
            "add",
            "alice",
            "--role",
            "reviewer",
            "--key-out",
            str(key_path),
            "--db",
            str(db),
        ]
    )

    key = key_path.read_text().strip()
    assert len(key) >= 40
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    stored = b"".join(p.read_bytes() for p in tmp_path.glob("store.db*"))
    assert key.encode() not in stored
    assert key not in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(
            [
                "principal",
                "add",
                "alice",
                "--role",
                "reviewer",
                "--key-out",
                str(tmp_path / "b.key"),
                "--db",
                str(db),
            ]
        )
    assert not (tmp_path / "b.key").exists()

    with pytest.raises(SystemExit):
        main(
            [
                "principal",
                "add",
                "carol",
                "--role",
                "reviewer",
                "--key-out",
                str(key_path),
                "--db",
                str(db),
            ]
        )
    assert key_path.read_text().strip() == key

    main(["principal", "list", "--db", str(db)])
    listed = capsys.readouterr().out
    assert "alice" in listed and "reviewer" in listed and key not in listed


def test_uploading_results_needs_no_credential_and_writes_no_human_grade(
    sample_evalset_path, tmp_path
) -> None:
    db = tmp_path / "store.db"

    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])

    with Store.open(db) as store:
        (run,) = store.list_runs()
        assert store.human_grades(run.id) == []
        assert run.human_graded == 0


def test_drift_says_when_the_baseline_run_was_regraded_after_the_baseline_was_set(
    tmp_path, capsys
) -> None:
    db, baseline, _newer = regressed_store(tmp_path)
    key = _principal(tmp_path, db, "alice")
    _grade(db, baseline.id, "c0", "0.0", "alice", key)
    capsys.readouterr()

    main(["drift", "--baseline", "b1", "--db", str(db)])

    err = capsys.readouterr().err
    assert "set a new baseline" in err.lower() and "b1" in err


def test_drift_is_quiet_about_the_baseline_when_nobody_regraded_it(tmp_path, capsys) -> None:
    db, _baseline, _newer = regressed_store(tmp_path)

    main(["drift", "--baseline", "b1", "--db", str(db)])

    assert "set a new baseline" not in capsys.readouterr().err.lower()
