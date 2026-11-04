"""Fix verification: a registered fix is judged by the runs that come after it.

`BEFORE` is two days ago and `AFTER` two days ahead, so a fix registered at the
real clock sits between them and "a run after the fix" is simply a run stamped
`AFTER`.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.findings_support import (
    AFTER,
    BEFORE,
    bad,
    good,
    make_run,
    ms,
    regressed_store,
    trace_file,
)
from verdryx.cli import main
from verdryx.store import Store


def _all(db: Path) -> list[Any]:
    with Store.open(db) as store:
        return store.list_findings(statuses=None)


def _only(db: Path) -> Any:
    (finding,) = _all(db)
    return finding


def _register(db: Path, finding_id: str, *extra: str, note: str = "tightened the prompt") -> None:
    main(["findings", "fix", finding_id[:8], "--note", note, "--db", str(db), *extra])


def _drift(db: Path, *extra: str) -> None:
    main(["drift", "--baseline", "b1", "--db", str(db), *extra])


def _with_open_finding(tmp_path: Path, **kw: Any) -> tuple[Path, Any]:
    db, _baseline, _newer = regressed_store(tmp_path, **kw)
    _drift(db)
    return db, _only(db)


def _save(db: Path, run: Any) -> None:
    with Store.open(db) as store:
        store.save_run(run)


def test_registering_a_fix_marks_the_finding_as_waiting_for_its_run(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path)

    _register(db, finding.id)

    after = _only(db)
    assert after.status == "fix_pending"
    assert [f.status for f in after.fixes] == ["pending"]
    assert after.fixes[0].note == "tightened the prompt"
    out = capsys.readouterr().out
    assert "registered" in out
    main(["findings", "list", "--db", str(db)])
    assert "fix_pending" in capsys.readouterr().out


def test_a_second_fix_is_refused_while_one_is_pending(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    pending = _only(db).fixes[0]
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        _register(db, finding.id, note="another idea")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "already has fix" in err and pending.id[:8] in err
    assert len(_only(db).fixes) == 1


def test_a_fix_for_an_unknown_finding_or_with_an_empty_note_is_refused(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        main(["findings", "fix", "zzzzzz", "--note", "x", "--db", str(db)])
    assert exc.value.code == 1
    assert "no finding" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        _register(db, finding.id, note="   ")
    assert exc.value.code == 1
    assert "needs a note" in capsys.readouterr().err
    assert _only(db).fixes == []


def test_a_run_after_the_fix_that_is_on_track_verifies_it_and_closes_the_finding(
    tmp_path, capsys
) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    _save(db, make_run(good(), started=AFTER, run_id="after-good"))
    capsys.readouterr()

    _drift(db)

    out = capsys.readouterr().out
    after = _only(db)
    assert after.status == "closed"
    assert after.closed_at is not None
    assert after.fixes[0].status == "verified"
    assert after.fixes[0].evaluated_at is not None
    assert after.fixes[0].result["runs"] == ["after-good"]
    assert "verified" in out


def test_a_run_after_the_fix_that_still_regresses_fails_it_and_reopens_the_finding_with_the_attempt(
    tmp_path, capsys
) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id, note="raised the temperature")
    _save(db, make_run(bad(), started=AFTER, run_id="after-bad"))
    capsys.readouterr()

    _drift(db)

    out = capsys.readouterr().out
    after = _only(db)
    assert after.status == "open"
    assert after.closed_at is None
    assert after.id == finding.id
    assert after.seen_count == 2
    assert "eval:after-bad" in after.evidence
    (attempt,) = after.fixes
    assert attempt.status == "failed"
    assert attempt.result["verdict"] == "regressed"
    assert "failed" in out

    main(["findings", "show", finding.id[:8], "--db", str(db)])
    shown = capsys.readouterr().out
    assert "[failed]" in shown and "raised the temperature" in shown


def test_a_drift_check_with_no_run_since_the_fix_leaves_it_pending(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    capsys.readouterr()

    _drift(db)

    out = capsys.readouterr().out
    after = _only(db)
    assert after.status == "fix_pending"
    assert after.fixes[0].status == "pending"
    assert "still pending" in out and "no eval run has started since" in out


def test_runs_from_before_the_fix_do_not_verify_it(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    _save(db, make_run(good(), started=AFTER, run_id="after-good"))
    capsys.readouterr()

    _drift(db, "--window", "2")

    out = capsys.readouterr().out
    # The pooled window (one bad run from before, one good from after) is still
    # regressed on its own terms; the fix is judged by the run after it alone.
    assert "verdict:    regressed" in out
    after = _only(db)
    assert after.fixes[0].status == "verified"
    assert after.fixes[0].result["runs"] == ["after-good"]
    assert after.status == "closed"


def test_a_closed_finding_is_not_reopened_by_the_runs_it_was_closed_over(tmp_path) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    _save(db, make_run(good(), started=AFTER, run_id="after-good"))
    _drift(db, "--window", "2")
    assert _only(db).status == "closed"

    _drift(db, "--window", "2")

    after = _only(db)
    assert after.status == "closed"
    assert after.seen_count == finding.seen_count


def test_a_regression_after_a_verified_fix_is_a_new_finding(tmp_path) -> None:
    db, finding = _with_open_finding(tmp_path)
    _register(db, finding.id)
    _save(db, make_run(good(), started=AFTER, run_id="after-good"))
    _drift(db)
    assert _only(db).status == "closed"
    _save(db, make_run(bad(), started=AFTER + timedelta(hours=1), run_id="later-bad"))

    _drift(db)

    findings = _all(db)
    assert sorted(f.status for f in findings) == ["closed", "open"]
    reopened = next(f for f in findings if f.status == "open")
    assert reopened.id != finding.id
    assert "eval:later-bad" in reopened.evidence
    assert "eval:bad-run" not in reopened.evidence


def test_a_run_of_another_version_does_not_test_a_fix_that_names_one(tmp_path, capsys) -> None:
    db, finding = _with_open_finding(tmp_path, version_before="v1", version_after="v1")
    _register(db, finding.id, "--version", "v2")
    _save(db, make_run(bad(), started=AFTER, run_id="old-build", version="v1"))
    capsys.readouterr()

    _drift(db)

    out = capsys.readouterr().out
    assert _only(db).fixes[0].status == "pending"
    assert "none of version 'v2'" in out

    _save(
        db, make_run(good(), started=AFTER + timedelta(hours=1), run_id="new-build", version="v2")
    )
    _drift(db)
    after = _only(db)
    assert after.fixes[0].status == "verified"
    assert after.fixes[0].result["runs"] == ["new-build"]


def test_registering_a_version_on_an_slo_finding_is_refused(tmp_path, capsys) -> None:
    trace = trace_file(tmp_path, ["case_resolved"] * 20 + ["escalated"] * 20, start_ms=ms(BEFORE))
    db = tmp_path / "store.db"
    main(["slo", "--input", str(trace), "--min-events", "10", "--db", str(db)])
    finding = _only(db)
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _register(db, finding.id, "--version", "v2")

    assert "carries no version" in capsys.readouterr().err


# ------------------------------------------------------------------
# Error budgets
# ------------------------------------------------------------------


def _slo(db: Path, *files: Path) -> None:
    combined = files[0].parent / "combined.ndjson"
    combined.write_text("".join(f.read_text() for f in files), encoding="utf-8")
    main(["slo", "--input", str(combined), "--min-events", "10", "--db", str(db)])


def _slo_finding_with_fix(tmp_path: Path) -> tuple[Path, Path, Any]:
    old = trace_file(
        tmp_path,
        ["case_resolved"] * 20 + ["escalated"] * 20,
        start_ms=ms(BEFORE),
        name="old.ndjson",
    )
    db = tmp_path / "store.db"
    _slo(db, old)
    finding = _only(db)
    main(["findings", "fix", finding.id[:8], "--note", "added a retry", "--db", str(db)])
    return db, old, finding


def test_an_slo_fix_with_too_few_runs_since_stays_pending_and_says_not_measured(
    tmp_path, capsys
) -> None:
    db, old, _finding = _slo_finding_with_fix(tmp_path)
    few = trace_file(
        tmp_path, ["case_resolved"] * 5, start_ms=ms(AFTER), name="few.ndjson", prefix="n"
    )
    capsys.readouterr()

    _slo(db, old, few)

    out = capsys.readouterr().out
    after = _only(db)
    assert after.status == "fix_pending"
    assert after.fixes[0].status == "pending"
    assert "not measured since the fix" in out


def test_an_slo_fix_is_verified_by_runs_after_it_that_meet_the_objective(tmp_path) -> None:
    db, old, finding = _slo_finding_with_fix(tmp_path)
    new = trace_file(
        tmp_path, ["case_resolved"] * 30, start_ms=ms(AFTER), name="new.ndjson", prefix="n"
    )

    _slo(db, old, new)

    after = _only(db)
    assert after.id == finding.id
    assert after.status == "closed"
    assert after.fixes[0].status == "verified"
    assert after.fixes[0].result["events"] == 30

    # The pooled window still holds twenty failures, so it still breaches; the
    # runs the fix was written against open nothing new.
    _slo(db, old, new)
    assert len(_all(db)) == 1


def test_an_slo_fix_that_still_breaches_is_failed(tmp_path, capsys) -> None:
    db, old, finding = _slo_finding_with_fix(tmp_path)
    new = trace_file(
        tmp_path,
        ["case_resolved"] * 15 + ["escalated"] * 15,
        start_ms=ms(AFTER),
        name="new.ndjson",
        prefix="n",
    )
    capsys.readouterr()

    _slo(db, old, new)

    after = _only(db)
    assert after.id == finding.id
    assert after.status == "open"
    assert after.fixes[0].status == "failed"
    assert after.seen_count == 2
    assert {e for e in after.evidence if e.startswith("trace:n")} == {
        f"trace:n{i}" for i in range(15, 30)
    }
    assert "failed" in capsys.readouterr().out
