"""The edges of findings, fixes and grades: the refusals, the empty cases and the
messages an operator reads when something is not there."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.findings_support import (
    AFTER,
    BEFORE,
    NOW,
    good,
    make_run,
    ms,
    regressed_store,
    seed,
    trace_file,
)
from verdryx.cli import main
from verdryx.findings import FixError, _num, evaluate_drift_fix, evaluate_slo_fix, register_fix
from verdryx.models import Baseline, EvalRun, Finding, Fix, Score
from verdryx.store import GradeError, PrincipalError, Store


def _finding(finding_id: str) -> Finding:
    return Finding(
        id=finding_id,
        dedupe_key="drift|x|b",
        agent_id=None,
        kind="drift",
        sli=None,
        window="1 run(s)",
        status="open",
        opened_at=NOW,
        last_seen_at=NOW,
        seen_count=1,
        evidence=[],
        detail={},
        suspect_model=None,
        suspect_version=None,
        suspect_since_run=None,
        suspect_note="n/a",
    )


def test_a_number_that_json_cannot_carry_is_written_as_text_never_dropped() -> None:
    assert _num(math.inf) == "inf"
    assert _num(-math.inf) == "-inf"
    assert _num(math.nan) == "nan"
    assert _num(0.0) == 0.0 and _num(None) is None


def test_registering_a_fix_refuses_an_unknown_and_a_closed_finding(tmp_path) -> None:
    db = tmp_path / "s.db"
    with Store.open(db) as store:
        with pytest.raises(FixError, match="no such finding"):
            register_fix(store, "nope", "x", version=None, now=NOW)
        closed = _finding("c1")
        closed.status = "closed"
        store.add_finding(closed)
        with pytest.raises(FixError, match="is closed"):
            register_fix(store, "c1", "x", version=None, now=NOW)


def test_a_fix_over_runs_with_no_scored_case_is_pending_not_verified() -> None:
    empty = EvalRun(id="r", model="m", started_at=AFTER, scores=[])
    baseline = Baseline(id="b", eval_run_id="x", mean_score=1.0, created_at=BEFORE)
    fix = Fix(id="f", finding_id="c", note="n", registered_at=NOW)

    status, result = evaluate_drift_fix(
        [empty], since=1, fix=fix, baseline=baseline, baseline_run=None, threshold=0.05
    )

    assert status == "pending"
    assert "no scored case" in result["reason"]


def test_an_slo_fix_with_no_measurement_at_all_is_pending() -> None:
    status, result = evaluate_slo_fix(None)
    assert status == "pending" and "no run for this agent" in result["reason"]


def test_an_ambiguous_finding_prefix_is_refused_and_a_filter_by_agent_works(
    tmp_path, capsys
) -> None:
    db = tmp_path / "s.db"
    with Store.open(db) as store:
        store.add_finding(_finding("abc111"))
        second = _finding("abc222")
        second.agent_id = "agent://acme.example/a"
        second.dedupe_key = "drift|a|b"
        store.add_finding(second)
        with pytest.raises(LookupError, match="2 findings start with"):
            store.resolve_finding_id("abc")
        assert store.resolve_finding_id("abc2") == "abc222"
        assert [f.id for f in store.list_findings(agent_id="agent://acme.example/a")] == ["abc222"]

    main(["findings", "list", "--agent-id", "agent://acme.example/a", "--db", str(db)])
    out = capsys.readouterr().out
    assert "abc222" in out and "abc111" not in out


def test_the_listings_say_when_there_is_nothing_and_a_closed_finding_shows_its_closing(
    tmp_path, capsys
) -> None:
    db = tmp_path / "s.db"
    main(["findings", "list", "--db", str(db)])
    assert "(none)" in capsys.readouterr().out

    with Store.open(db) as store:
        done = _finding("done01")
        done.status = "closed"
        done.closed_at = NOW
        store.add_finding(done)
    main(["findings", "show", "done01", "--db", str(db)])
    assert "closed:" in capsys.readouterr().out
    main(["findings", "list", "--status", "closed", "--db", str(db)])
    assert "done01" in capsys.readouterr().out


@pytest.mark.parametrize("group", ["findings", "principal"])
def test_a_group_command_with_nothing_after_it_says_how_to_use_it(group, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([group])
    assert exc.value.code == 1
    assert f"verdryx {group}" in capsys.readouterr().err


def test_principal_names_and_roles_that_cannot_be_registered_are_refused(tmp_path) -> None:
    with Store.open(tmp_path / "s.db") as store:
        with pytest.raises(PrincipalError, match="not a usable name"):
            store.add_principal("has space", "reviewer", "k")
        with pytest.raises(PrincipalError, match="not a role"):
            store.add_principal("alice", "admin", "k")


def test_a_grade_for_a_run_that_does_not_exist_is_refused(tmp_path) -> None:
    with Store.open(tmp_path / "s.db") as store:
        store.add_principal("alice", "reviewer", "k")
        with pytest.raises(GradeError, match="no such eval run"):
            store.add_human_grade("ghost", "c0", 0.5, author="alice", key="k")


def test_an_unreadable_key_file_and_an_unwritable_key_path_are_said_plainly(
    tmp_path, capsys
) -> None:
    db = tmp_path / "s.db"
    seed(db, [make_run(good(2), run_id="r1")])
    with pytest.raises(SystemExit):
        main(
            [
                "grade",
                "r1",
                "c0",
                "0.5",
                "--author",
                "a",
                "--key-file",
                str(tmp_path / "no"),
                "--db",
                str(db),
            ]
        )
    assert "cannot read" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        main(
            [
                "principal",
                "add",
                "a",
                "--role",
                "reviewer",
                "--key-out",
                str(tmp_path / "no" / "dir" / "k"),
                "--db",
                str(db),
            ]
        )
    assert "cannot write the key" in capsys.readouterr().err


def test_eval_prints_the_version_it_was_given(sample_evalset_path, tmp_path, capsys) -> None:
    main(
        [
            "eval",
            str(sample_evalset_path),
            "--model",
            "stub",
            "--db",
            str(tmp_path / "s.db"),
            "--agent-version",
            "v9",
        ]
    )
    assert "version=v9" in capsys.readouterr().out
    with Store.open(tmp_path / "s.db") as store:
        assert store.list_runs()[0].agent_version == "v9"


def test_a_run_saved_without_a_version_reads_back_without_one(tmp_path) -> None:
    with Store.open(tmp_path / "s.db") as store:
        store.save_run(make_run(good(1), run_id="r1", version=""))
        run = store.load_run("r1")
    assert run is not None and run.agent_version is None
    assert Score("c", 1.0).graded_by is None


def test_an_slo_breach_after_a_verified_fix_is_a_new_finding(tmp_path) -> None:
    old = trace_file(
        tmp_path, ["case_resolved"] * 20 + ["escalated"] * 20, start_ms=ms(BEFORE), name="o.ndjson"
    )
    db = tmp_path / "s.db"
    combined = tmp_path / "all.ndjson"

    def slo(*files: Path) -> None:
        combined.write_text("".join(f.read_text() for f in files), encoding="utf-8")
        main(["slo", "--input", str(combined), "--min-events", "10", "--db", str(db)])

    slo(old)
    with Store.open(db) as store:
        (first,) = store.list_findings(statuses=None)
    main(["findings", "fix", first.id[:8], "--note", "retry", "--db", str(db)])
    fixed = trace_file(
        tmp_path, ["case_resolved"] * 30, start_ms=ms(AFTER), name="f.ndjson", prefix="f"
    )
    slo(old, fixed)
    later = trace_file(
        tmp_path,
        ["case_resolved"] * 10 + ["escalated"] * 30,
        start_ms=ms(AFTER + timedelta(hours=1)),
        name="l.ndjson",
        prefix="l",
    )

    slo(old, fixed, later)

    with Store.open(db) as store:
        found = store.list_findings(statuses=None)
    assert sorted(f.status for f in found) == ["closed", "open"]
    new = next(f for f in found if f.status == "open")
    assert new.id != first.id
    assert all(e.startswith("trace:l") for e in new.evidence)


def test_an_slo_run_leaves_a_pending_drift_fix_to_drift(tmp_path) -> None:
    db, _baseline, _newer = regressed_store(tmp_path)
    main(["drift", "--baseline", "b1", "--db", str(db)])
    with Store.open(db) as store:
        (drift_finding,) = store.list_findings(statuses=None)
    main(["findings", "fix", drift_finding.id[:8], "--note", "x", "--db", str(db)])
    trace = trace_file(tmp_path, ["case_resolved"] * 30, start_ms=ms(AFTER))

    main(["slo", "--input", str(trace), "--min-events", "10", "--db", str(db)])

    with Store.open(db) as store:
        (after,) = store.list_findings(statuses=None)
    assert after.status == "fix_pending" and after.fixes[0].status == "pending"


def test_a_breach_with_only_old_runs_after_a_fix_opens_nothing_and_says_why(
    tmp_path, capsys
) -> None:
    old = trace_file(
        tmp_path, ["case_resolved"] * 20 + ["escalated"] * 20, start_ms=ms(BEFORE), name="o.ndjson"
    )
    db = tmp_path / "s.db"
    main(["slo", "--input", str(old), "--min-events", "10", "--db", str(db)])
    with Store.open(db) as store:
        (finding,) = store.list_findings(statuses=None)
    main(["findings", "fix", finding.id[:8], "--note", "x", "--db", str(db)])
    fixed = trace_file(
        tmp_path, ["case_resolved"] * 30, start_ms=ms(AFTER), name="f.ndjson", prefix="f"
    )
    both = tmp_path / "both.ndjson"
    both.write_text(old.read_text() + fixed.read_text())
    main(["slo", "--input", str(both), "--min-events", "10", "--db", str(db)])
    capsys.readouterr()

    main(["slo", "--input", str(both), "--min-events", "10", "--db", str(db)])

    out: Any = capsys.readouterr().out
    assert "no new finding" in out
    with Store.open(db) as store:
        assert [f.status for f in store.list_findings(statuses=None)] == ["closed"]
