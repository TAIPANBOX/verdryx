"""Findings: what `drift` and `slo` write down when they see a failure.

Every test here runs against stores and trace files only. No model is called
and none is reachable (see the no-network test), because a finding has to be
as useful with the model gone as with it.
"""

from __future__ import annotations

import socket
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.findings_support import (
    AGENT,
    BEFORE,
    bad,
    good,
    make_run,
    ms,
    regressed_store,
    seed,
    trace_file,
)
from verdryx.cli import main
from verdryx.findings import find_suspect
from verdryx.store import Store


def _findings(db: Path) -> list[Any]:
    with Store.open(db) as store:
        return store.list_findings(statuses=None)


def test_a_regressed_drift_check_records_a_finding_with_agent_window_kind_and_evidence(
    tmp_path, capsys
) -> None:
    db, _baseline, newer = regressed_store(tmp_path)

    main(["drift", "--baseline", "b1", "--db", str(db)])

    out = capsys.readouterr().out
    assert "regressed" in out
    (finding,) = _findings(db)
    assert finding.agent_id == AGENT
    assert finding.kind == "drift"
    assert finding.status == "open"
    assert finding.seen_count == 1
    assert "1 run" in finding.window
    assert f"eval:{newer.id}" in finding.evidence
    worst = [e for e in finding.evidence if e.startswith(f"eval:{newer.id}:")]
    assert 1 <= len(worst) <= 5
    assert finding.detail["delta"] == pytest.approx(-1.0)
    assert finding.detail["baseline_id"] == "b1"
    assert finding.id[:8] in out


def test_an_on_track_drift_check_records_no_finding(tmp_path, capsys) -> None:
    db = tmp_path / "store.db"
    base = make_run(good(), started=BEFORE, run_id="base")
    same = make_run(good(), started=BEFORE + timedelta(hours=1), run_id="same")
    seed(db, [base, same], baseline_run=base)

    main(["drift", "--baseline", "b1", "--db", str(db)])

    assert "on-track" in capsys.readouterr().out
    assert _findings(db) == []


def test_the_same_regression_seen_twice_is_one_finding_seen_twice(tmp_path) -> None:
    db, _baseline, first = regressed_store(tmp_path)
    main(["drift", "--baseline", "b1", "--db", str(db)])
    second = make_run(bad(), started=BEFORE + timedelta(hours=2), run_id="bad-run-2")
    with Store.open(db) as store:
        store.save_run(second)

    main(["drift", "--baseline", "b1", "--db", str(db)])

    (finding,) = _findings(db)
    assert finding.seen_count == 2
    assert f"eval:{second.id}" in finding.evidence
    assert f"eval:{first.id}" in finding.evidence


def test_the_finding_names_the_version_that_changed_since_the_baseline(tmp_path) -> None:
    db, _baseline, newer = regressed_store(tmp_path, version_before="v1", version_after="v2")

    main(["drift", "--baseline", "b1", "--db", str(db)])

    (finding,) = _findings(db)
    assert finding.suspect_version == "v2"
    assert finding.suspect_since_run == newer.id
    assert "v1" in finding.suspect_note and "v2" in finding.suspect_note


def test_a_finding_with_no_change_in_model_or_version_says_none_changed(tmp_path) -> None:
    db, _baseline, _newer = regressed_store(tmp_path, version_before="v1", version_after="v1")

    main(["drift", "--baseline", "b1", "--db", str(db)])

    (finding,) = _findings(db)
    assert finding.suspect_model is None
    assert finding.suspect_version is None
    assert finding.suspect_since_run is None
    assert "did not change" in finding.suspect_note


def test_find_suspect_over_a_single_run_says_there_is_nothing_to_compare() -> None:
    suspect = find_suspect([make_run(good(), version="v1")])
    assert suspect.model is None and suspect.version is None and suspect.since_run is None
    assert "nothing to compare" in suspect.note
    assert "did not change" not in suspect.note


def test_find_suspect_reports_a_model_change_over_a_version_label_that_did_not_move() -> None:
    first = make_run(good(), started=BEFORE, run_id="r1", model="m1", version="v1")
    second = make_run(good(), started=BEFORE + timedelta(hours=1), run_id="r2", model="m2")
    third = make_run(good(), started=BEFORE + timedelta(hours=2), run_id="r3", model="m2")

    suspect = find_suspect([third, first, second])

    assert suspect.model == "m2"
    assert suspect.since_run == "r2"
    assert "m1" in suspect.note and "m2" in suspect.note


def test_a_regression_nobody_can_attribute_is_still_a_finding_without_an_agent(
    tmp_path, capsys
) -> None:
    db, _baseline, _newer = regressed_store(tmp_path, agent=None)

    main(["drift", "--baseline", "b1", "--db", str(db)])
    capsys.readouterr()

    (finding,) = _findings(db)
    assert finding.agent_id is None
    main(["findings", "list", "--db", str(db)])
    assert "(no agent id)" in capsys.readouterr().out


def test_a_finding_is_recorded_with_no_model_and_no_network(tmp_path, monkeypatch) -> None:
    db, _baseline, _newer = regressed_store(tmp_path)

    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a finding must not reach for the network or a model")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr("verdryx.graders.AnthropicAdapter.__init__", refuse)
    monkeypatch.setattr("urllib.request.urlopen", refuse)

    main(["drift", "--baseline", "b1", "--db", str(db)])

    assert len(_findings(db)) == 1


# ------------------------------------------------------------------
# The error budget
# ------------------------------------------------------------------


def _breach(tmp_path: Path, **kw: Any) -> Path:
    return trace_file(
        tmp_path, ["case_resolved"] * 20 + ["escalated"] * 20, start_ms=ms(BEFORE), **kw
    )


def test_an_established_slo_breach_records_a_finding_with_the_failing_runs(
    tmp_path, capsys
) -> None:
    trace = _breach(tmp_path)
    db = tmp_path / "store.db"

    main(["slo", "--input", str(trace), "--min-events", "10", "--db", str(db)])

    out = capsys.readouterr().out
    (finding,) = _findings(db)
    assert finding.kind == "slo_exhausted"
    assert finding.sli == "task_success"
    assert finding.agent_id == AGENT
    assert finding.status == "open"
    assert finding.window == "28d"
    failing = {f"trace:t{i}" for i in range(20, 40)}
    assert set(finding.evidence) == failing
    assert finding.detail["events"] == 40
    assert finding.detail["good"] == 20
    assert finding.id[:8] in out


def test_a_slo_breach_the_interval_does_not_establish_records_no_finding(tmp_path, capsys) -> None:
    trace = trace_file(tmp_path, ["case_resolved"] * 56 + ["escalated"] * 4, start_ms=ms(BEFORE))
    db = tmp_path / "store.db"

    main(["slo", "--input", str(trace), "--min-events", "10", "--db", str(db)])

    assert _findings(db) == []
    assert "not recorded" not in capsys.readouterr().out


def test_an_slo_breach_with_no_store_says_it_was_not_recorded(tmp_path, capsys) -> None:
    trace = _breach(tmp_path)

    main(["slo", "--input", str(trace), "--min-events", "10"])

    out = capsys.readouterr().out
    assert "not recorded as a finding" in out
    assert "--db" in out


def test_an_slo_breach_of_a_key_is_reported_and_never_recorded_as_a_finding(
    tmp_path, capsys
) -> None:
    trace = _breach(tmp_path, key="k-ops")
    db = tmp_path / "store.db"

    main(
        [
            "slo",
            "--input",
            str(trace),
            "--min-events",
            "10",
            "--identity-field",
            "key_id",
            "--db",
            str(db),
        ]
    )

    assert _findings(db) == []
    out = capsys.readouterr().out
    assert "key_id" in out and "not recorded as a finding" in out


def test_findings_list_and_show_print_what_was_recorded(tmp_path, capsys) -> None:
    db, _baseline, newer = regressed_store(tmp_path, version_before="v1", version_after="v2")
    main(["drift", "--baseline", "b1", "--db", str(db)])
    capsys.readouterr()
    (finding,) = _findings(db)

    main(["findings", "list", "--db", str(db)])
    listing = capsys.readouterr().out
    assert finding.id[:8] in listing
    assert "open" in listing and AGENT in listing and "drift" in listing

    main(["findings", "show", finding.id[:8], "--db", str(db)])
    shown = capsys.readouterr().out
    for needle in ("window", "evidence", "suspect", "delta", f"eval:{newer.id}", "v2"):
        assert needle in shown, needle


def test_findings_show_refuses_an_unknown_or_ambiguous_id(tmp_path, capsys) -> None:
    db, _baseline, _newer = regressed_store(tmp_path)
    main(["drift", "--baseline", "b1", "--db", str(db)])

    with pytest.raises(SystemExit) as exc:
        main(["findings", "show", "zzzzzz", "--db", str(db)])
    assert exc.value.code == 1
    assert "no finding" in capsys.readouterr().err
