"""Tests for verdryx.slo.

Every test here was run against a broken implementation before it was allowed
to pass, and which mutation made it red is named in the PR. The ones worth
knowing about, because they are the failures this module could plausibly ship
with and nobody would see:

  - a burn event sent for a subject whose interval still covers the target
    (paging on noise, which teaches an operator to filter the sender);
  - an unattributed run bucketed under a placeholder subject instead of
    counted (a fleet then scores better the less of it is identified);
  - the cost reference computed per subject instead of over the fleet (every
    agent compared to itself, so all of them pass);
  - `refused_calls` read off the nine Breaker reasons alone, so a fleet the
    policy plane is refusing reads as unrestrained.
"""

from __future__ import annotations

import math

import pytest

from verdryx import slo
from verdryx.costper import UNTAGGED, _CallRow, _reduce_call_rows, is_refused_decision
from verdryx.models import OUTCOME_ESCALATED, OUTCOME_RESOLVED


def _run(**kw):
    """One per-run record with sane defaults, overridden per test."""
    base = {
        "outcome": OUTCOME_RESOLVED,
        "cost_usd": 0.0005,
        "run_id": "r",
        "agent_id": "agent://acme.example/support/bot",
        "key_id": "k1",
        "calls": 2,
        "refused_calls": 0,
    }
    base.update(kw)
    return base


def _runs(n, prefix="r", **kw):
    return [_run(run_id=f"{prefix}{i}", **kw) for i in range(n)]


# ---------------------------------------------------------------- the maths


def test_wilson_widens_on_a_small_sample():
    """Three of four says nothing; three hundred of four hundred says a lot.

    The whole reason the interval is computed rather than only the ratio.
    """
    small_low, small_high = slo.wilson_interval(3, 4, 0.95)
    big_low, big_high = slo.wilson_interval(300, 400, 0.95)
    assert (small_high - small_low) > (big_high - big_low) * 3
    assert 0.0 <= small_low < small_high <= 1.0


def test_wilson_on_an_empty_sample_is_the_whole_range():
    """Not a point estimate of zero, which would read as total failure."""
    assert slo.wilson_interval(0, 0, 0.95) == (0.0, 1.0)


def test_wilson_never_leaves_the_unit_interval_at_the_edges():
    """Where the textbook normal interval is wrong and Wilson is not."""
    for good, total in ((20, 20), (0, 20), (1, 200), (199, 200)):
        low, high = slo.wilson_interval(good, total, 0.95)
        assert 0.0 <= low <= high <= 1.0


def test_the_error_budget_goes_negative_rather_than_clamping():
    """`-1.0` is twice the allowed failures, and an operator must be able to
    tell that from a budget exactly used up."""
    assert slo.error_budget_remaining(0.95, 0.95) == pytest.approx(0.0)
    assert slo.error_budget_remaining(0.90, 0.95) == pytest.approx(-1.0)
    assert slo.error_budget_remaining(1.0, 0.95) == pytest.approx(1.0)


def test_burn_rate_of_one_spends_the_budget_exactly():
    assert slo.burn_rate(0.95, 0.95) == pytest.approx(1.0)
    assert slo.burn_rate(1.0, 0.95) == pytest.approx(0.0)
    assert slo.burn_rate(0.70, 0.95) == pytest.approx(6.0)


def test_a_target_of_one_permits_no_failures_and_says_so():
    """A perfect objective has no budget to hold a fraction of, and the code
    must not divide by the zero allowance."""
    assert slo.error_budget_remaining(1.0, 1.0) == 1.0
    assert slo.error_budget_remaining(0.99, 1.0) == -math.inf
    assert slo.burn_rate(0.99, 1.0) == math.inf


def test_exhaustion_outranks_rate():
    """Reporting fast_burn for a budget already gone tells an operator to
    watch something that has finished happening."""
    assert slo.trigger_for(-0.5, 99.0) == slo.TRIGGER_EXHAUSTED
    assert slo.trigger_for(0.5, 7.0) == slo.TRIGGER_FAST_BURN
    assert slo.trigger_for(0.5, 3.0) == slo.TRIGGER_SLOW_BURN
    assert slo.trigger_for(0.9, 1.0) is None


# ------------------------------------------------------------- the indicators


def test_an_untagged_run_is_not_a_failed_run():
    """It leaves the denominator too. Counting it as a failure would make an
    unconfigured fleet look broken rather than unmeasured."""
    cfg = {"good_outcomes": frozenset({OUTCOME_RESOLVED})}
    assert slo._sli_task_success({"outcome": UNTAGGED}, cfg) is None
    assert slo._sli_task_success({"outcome": ""}, cfg) is None
    assert slo._sli_task_success({"outcome": OUTCOME_RESOLVED}, cfg) is True
    assert slo._sli_task_success({"outcome": OUTCOME_ESCALATED}, cfg) is False


def test_escalation_is_a_failure_of_task_success_by_default():
    """The default most likely to be argued with, so it is pinned. The budget
    prices autonomy, and an escalation is a person being pulled in."""
    assert OUTCOME_ESCALATED not in slo.DEFAULT_GOOD_OUTCOMES
    report = slo.compute_slo(_runs(40, outcome=OUTCOME_ESCALATED))
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_TASK_SUCCESS)
    assert m.observed == 0.0


def test_containment_counts_a_wardryx_refusal_and_not_only_a_breaker_one():
    """The two the nine do not cover. Without them a fleet the policy plane is
    refusing reads as perfectly contained."""
    assert is_refused_decision("wardryx_deny")
    assert is_refused_decision("wardryx_hold")
    assert is_refused_decision("budget_exceeded")
    assert not is_refused_decision("allow")
    assert not is_refused_decision("cache_hit")


def test_the_reduction_counts_a_wardryx_deny_as_refused():
    """End to end through the shared reduction, not only the predicate."""
    rows = [
        _CallRow("run-1", 0, "", 0.0, "wardryx_deny", "a", "k", 1),
        _CallRow("run-1", 1, OUTCOME_RESOLVED, 0.001, "allow", "a", "k", 2),
    ]
    (record,) = _reduce_call_rows(rows)
    assert record["refused_calls"] == 1
    assert record["calls"] == 2
    assert record["agent_id"] == "a"


def test_cost_discipline_is_measured_against_the_fleet_not_the_agent():
    """An agent expensive since the day it shipped must not be graded against
    its own habit, which would pass it forever.

    The expensive agent is a MINORITY of the fleet on purpose. Half a fleet
    being expensive moves the median to the middle and the test then proves
    nothing about which reference was used, which is how the first version of
    this test passed against both implementations.
    """
    cheap = _runs(60, prefix="c", agent_id="cheap", cost_usd=0.001)
    dear = _runs(20, prefix="d", agent_id="dear", cost_usd=0.100)
    report = slo.compute_slo(cheap + dear, min_events=10)
    assert report.cost_reference_usd == pytest.approx(0.001), "the fleet median"
    m = _find(report, "dear", slo.SLI_COST_DISCIPLINE)
    assert m.observed == 0.0, "every one of its runs is 100x the fleet median"
    assert _find(report, "cheap", slo.SLI_COST_DISCIPLINE).observed == 1.0


def test_the_cost_reference_is_a_median_not_a_mean():
    """The mean is dragged by exactly the runaway runs this catches."""
    runs = [_run(cost_usd=c) for c in [0.001] * 9 + [10.0]]
    assert slo.cost_reference(runs) == pytest.approx(0.001)


# -------------------------------------------------------------- the report


def test_an_unattributed_run_is_counted_and_never_bucketed():
    """A fleet scores better the less of it is identified either way, and only
    one of the two says so."""
    runs = _runs(20) + _runs(5, prefix="u", agent_id="")
    report = slo.compute_slo(runs, min_events=5)
    assert report.unattributed_runs == 5
    assert report.total_runs == 25
    assert "" not in report.subjects
    assert len(report.subjects) == 1


def test_an_indicator_below_min_events_reports_as_unmeasured():
    report = slo.compute_slo(_runs(4), min_events=20)
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_TASK_SUCCESS)
    assert m.measured is False
    assert "below the 20" in m.unmeasured_reason
    assert m.observed == 0.0, "an unmeasured indicator must not present a ratio"


def test_quality_floor_says_why_it_could_not_be_measured():
    """An operator needs to be sent to the right place, and the right place
    changed on 2026-08-26.

    The reason used to be structural: scores lived in an eval store whose
    `eval_runs` table had no subject on it, so nothing could be joined to a
    fleet at all. That is no longer true, and a reason that has stopped being
    true is worse than none: it sends the reader to close a gap that is
    already closed. It now names the two commands that carry a score from the
    eval store to this report.
    """
    report = slo.compute_slo(_runs(40))
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.measured is False
    assert "--scores-db" in m.unmeasured_reason
    assert "eval --agent-id" in m.unmeasured_reason
    assert "carries no agent_id" not in m.unmeasured_reason


def test_quality_floor_is_measured_when_scores_are_supplied():
    runs = _runs(20, score=0.9) + _runs(
        20,
        prefix="bad",
        score=0.1,
    )
    report = slo.compute_slo(runs, min_events=10)
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.measured is True
    assert m.observed == pytest.approx(0.5)


def test_blind_spots_name_what_was_not_computed():
    """A burn alert that can never fire reports exactly like one with nothing
    to report."""
    report = slo.compute_slo(_runs(40))
    assert any("quality_floor" in line for line in report.blind_spots)


def test_an_unknown_identity_field_is_refused():
    with pytest.raises(slo.SloInputError, match="identity_field"):
        slo.compute_slo(_runs(5), identity_field="run_id")


def test_a_target_outside_the_unit_interval_is_refused():
    with pytest.raises(slo.SloInputError, match=r"within 0\.\.1"):
        slo.compute_slo(_runs(5), targets={slo.SLI_TASK_SUCCESS: 1.5})


def test_an_unknown_sli_in_targets_is_refused():
    with pytest.raises(slo.SloInputError, match="unknown SLI"):
        slo.compute_slo(_runs(5), targets={"latency": 0.9})


# --------------------------------------------------------------- the bus


def _timeline(good_then, bad_now, span_ms=28 * 24 * 3600 * 1000):
    """Runs across a window, with the failures concentrated at the END.

    This is what makes a burn rate mean anything: the budget looks at the
    whole window and the rate looks at the recent slice, so a fleet that was
    fine for a month and broke this morning has budget left and a high rate.
    A fixture with its failures spread evenly cannot tell the two windows
    apart and would pass against the implementation this module had before
    2026-08-26, where the rate was the budget with its sign flipped.
    """
    out = []
    total = good_then + bad_now
    for i in range(good_then):
        out.append(_run(run_id=f"old{i}", last_ts_millis=int(i * span_ms / total)))
    for i in range(bad_now):
        out.append(
            _run(
                run_id=f"new{i}",
                outcome=OUTCOME_ESCALATED,
                last_ts_millis=int((good_then + i) * span_ms / total),
            )
        )
    return out


def test_a_slow_burn_never_reaches_the_bus():
    """Reported, never alerted. One type is one paging band."""
    # Healthy across the window, three of the last eight runs escalating: the
    # budget survives and the recent slice burns at a multiple that warns.
    runs = _timeline(good_then=225, bad_now=15)
    report = slo.compute_slo(
        runs,
        min_events=10,
        min_burn_events=5,
        targets={slo.SLI_TASK_SUCCESS: 0.90},
    )
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_TASK_SUCCESS)
    assert m.remaining > 0.0, "the budget must still have room, or this is exhaustion"
    assert m.burn_events_seen > 0, "the rate must actually have been measured"
    assert m.trigger in (slo.TRIGGER_SLOW_BURN, slo.TRIGGER_FAST_BURN)
    if m.trigger == slo.TRIGGER_SLOW_BURN:
        assert not [e for e in slo.burn_events(report) if e["sli"] == slo.SLI_TASK_SUCCESS]


def test_a_burn_rate_with_no_clock_is_unmeasured_and_not_zero():
    """The two look identical in the float and are opposite facts, so the
    report has to carry the count beside the rate."""
    report = slo.compute_slo(_runs(40), min_events=10)
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_TASK_SUCCESS)
    assert m.burn_events_seen == 0
    assert m.burn_rate == 0.0
    assert any("burn rate not measurable" in line for line in report.blind_spots)


def test_a_rate_trigger_cannot_fire_on_an_unmeasured_rate():
    """The defect this module shipped with for one afternoon: burn and budget
    computed over the same runs makes the rate `1 - remaining`, so a rate
    trigger is unreachable and a zero rate is indistinguishable from no clock.
    """
    assert slo.trigger_for(0.5, 0.0) is None
    assert slo.trigger_for(0.5, 3.0) == slo.TRIGGER_SLOW_BURN


def test_a_breach_the_interval_does_not_establish_never_reaches_the_bus():
    """56 good of 60 against 0.95 is a third over budget by the point estimate
    and an interval that comfortably contains the target. Paging on it teaches
    an operator to filter the sender."""
    runs = _runs(56) + _runs(4, prefix="bad", outcome=OUTCOME_ESCALATED)
    report = slo.compute_slo(runs, min_events=10)
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_TASK_SUCCESS)
    assert m.trigger == slo.TRIGGER_EXHAUSTED, "the report still says it is over budget"
    assert m.ci_high > m.target, "and the interval still covers the target"
    assert slo.breach_is_established(m) is False
    assert not [e for e in slo.burn_events(report) if e["sli"] == slo.SLI_TASK_SUCCESS]


def test_an_established_breach_does_reach_the_bus_with_its_evidence():
    """The positive control: without it, every assertion above is satisfied by
    an emitter that sends nothing at all."""
    runs = _runs(20) + _runs(20, prefix="bad", outcome=OUTCOME_ESCALATED)
    report = slo.compute_slo(runs, min_events=10)
    events = [e for e in slo.burn_events(report) if e["sli"] == slo.SLI_TASK_SUCCESS]
    assert len(events) == 1
    e = events[0]
    assert e["trigger"] == slo.TRIGGER_EXHAUSTED
    assert e["identity_field"] == slo.IDENTITY_AGENT_ID
    assert e["events"] == 40
    assert e["budget_remaining"] < 0
    assert e["ci_high"] < e["target"]
    assert set(e) == {
        "sli",
        "target",
        "observed",
        "ci_low",
        "ci_high",
        "events",
        "window",
        "budget_remaining",
        "burn_rate",
        "trigger",
        "identity_field",
        "_subject",
    }


def test_an_unmeasured_indicator_emits_nothing():
    """There is no honest event to send about a number that was not computed."""
    report = slo.compute_slo(_runs(40))
    assert not [e for e in slo.burn_events(report) if e["sli"] == slo.SLI_QUALITY_FLOOR]


def test_slo_burn_is_registered_at_the_high_band():
    """Fixed per type, so no emission site can choose it."""
    from verdryx.events import EVENT_SEVERITY

    assert EVENT_SEVERITY["slo_burn"] == "high"


def _find(report, subject, sli):
    return next(m for m in report.subjects[subject] if m.sli == sli)


def test_a_key_keyed_subject_is_reported_and_never_emitted():
    """`key_id` is the sound key for anything enforced and is not an agent id.

    The envelope has one subject field and SPEC 3.1 fixes its grammar, so a
    measurement about a credential has nowhere to go on the bus. Reported,
    not emitted, and not written malformed either.
    """
    assert slo.subject_is_emittable("agent://acme.example/support/bot")
    assert not slo.subject_is_emittable("k-ops")
    assert not slo.subject_is_emittable("")

    runs = _runs(20, agent_id="", key_id="k-ops") + _runs(
        20, prefix="bad", agent_id="", key_id="k-ops", outcome=OUTCOME_ESCALATED
    )
    report = slo.compute_slo(runs, identity_field=slo.IDENTITY_KEY_ID, min_events=10)
    payloads = slo.burn_events(report)
    assert payloads, "the breach is real and belongs in the report"
    assert not [p for p in payloads if slo.subject_is_emittable(p["_subject"])]


# ------------------------------------------- scores joined to a fleet subject


def _score(subject="agent://acme.example/support/bot", value=1.0, **kw):
    """One score record as `Store.score_records` produces it."""
    base = {"agent_id": subject, "score": value}
    base.update(kw)
    return base


def _scores(n, subject="agent://acme.example/support/bot", value=1.0, **kw):
    return [_score(subject, value, **kw) for _ in range(n)]


def test_quality_floor_computes_from_scores_supplied_beside_the_trace():
    """The indicator the module shipped dark.

    The trace carries no score and the eval store carries no tokenfuse run id,
    so the join is on the SUBJECT and nowhere else. Half the scores under the
    floor is a ratio of one half.
    """
    report = slo.compute_slo(
        _runs(40),
        scores=_scores(20, value=0.9) + _scores(20, value=0.1),
        min_events=10,
    )
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.measured is True
    assert m.events == 40
    assert m.observed == pytest.approx(0.5)


def test_a_score_with_no_subject_is_counted_and_never_bucketed():
    """The rule `unattributed_runs` already holds, applied to the other input.

    A fleet must not score better for being less identified, and it does
    exactly that if an unattributed score is dropped OR if it is filed under a
    placeholder subject.
    """
    report = slo.compute_slo(
        _runs(40),
        scores=_scores(20, value=1.0) + _scores(10, subject="", value=0.0),
        min_events=10,
    )
    assert report.unattributed_scores == 10
    assert report.total_scores == 30
    assert "" not in report.subjects
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.events == 20, "the ten with no subject are in nobody's denominator"
    assert m.observed == pytest.approx(1.0)


def test_a_subject_known_only_from_its_scores_still_appears():
    """An agent evaluated offline whose gateway traffic is not tagged.

    Dropping its scores would be a silent loss, and filing them under a trace
    subject would be a fabrication. It gets its own row, with the three
    trace-fed indicators reported as not computed.
    """
    report = slo.compute_slo(
        _runs(40),
        scores=_scores(20, subject="agent://acme.example/research/scout", value=1.0),
        min_events=10,
    )
    assert "agent://acme.example/research/scout" in report.subjects
    m = _find(report, "agent://acme.example/research/scout", slo.SLI_QUALITY_FLOOR)
    assert m.measured is True
    assert m.observed == pytest.approx(1.0)
    task = _find(report, "agent://acme.example/research/scout", slo.SLI_TASK_SUCCESS)
    assert task.measured is False
    assert "no run" in task.unmeasured_reason


def test_a_subject_with_no_runs_at_all_says_so_rather_than_naming_a_missing_tag():
    """ "No run carried an outcome tag" is the wrong diagnosis when there were
    no runs. It sends an operator to look at a header they never had a chance
    to send."""
    report = slo.compute_slo(
        [],
        scores=_scores(20, subject="agent://acme.example/research/scout"),
        min_events=10,
    )
    task = _find(report, "agent://acme.example/research/scout", slo.SLI_TASK_SUCCESS)
    assert task.measured is False
    assert "no runs at all" in task.unmeasured_reason


def test_a_score_on_the_trace_and_one_supplied_are_judged_together():
    """A record carrying its own `score` column is a legitimate input and did
    not stop being one. Both streams land in the same denominator."""
    report = slo.compute_slo(
        _runs(20, score=1.0),
        scores=_scores(20, value=0.0),
        min_events=10,
    )
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.events == 40
    assert m.observed == pytest.approx(0.5)


def test_scores_keyed_by_an_agent_cannot_be_attributed_to_a_credential():
    """Grouping by `key_id` is the sound choice for anything enforced, and the
    eval store has no credential on it. Every score is then unattributed, which
    is the honest answer and is counted rather than quietly absent."""
    report = slo.compute_slo(
        _runs(40),
        scores=_scores(20),
        identity_field=slo.IDENTITY_KEY_ID,
        min_events=10,
    )
    assert report.unattributed_scores == 20
    m = _find(report, "k1", slo.SLI_QUALITY_FLOOR)
    assert m.measured is False


def test_the_quality_burn_rate_is_measurable_once_scores_carry_a_clock():
    """Without a timestamp the quality burn rate is a permanent blind spot, so
    the indicator would compute a ratio and never be able to warn about it."""
    span = 28 * 24 * 3600 * 1000
    good = [_score(value=1.0, last_ts_millis=int(i * span / 240)) for i in range(225)]
    bad = [_score(value=0.0, last_ts_millis=int((225 + i) * span / 240)) for i in range(15)]
    report = slo.compute_slo(
        [],
        scores=good + bad,
        min_events=10,
        min_burn_events=5,
        targets={slo.SLI_QUALITY_FLOOR: 0.90},
    )
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.burn_events_seen > 0, "the rate must actually have been measured"
    assert m.burn_rate > 1.0
    assert m.trigger in (slo.TRIGGER_SLOW_BURN, slo.TRIGGER_FAST_BURN)


def test_no_scores_supplied_leaves_the_report_exactly_as_it_was():
    """The argument is optional and its absence changes nothing, including the
    two new counters, which must read zero rather than None."""
    report = slo.compute_slo(_runs(40), min_events=10)
    assert report.total_scores == 0
    assert report.unattributed_scores == 0
    m = _find(report, "agent://acme.example/support/bot", slo.SLI_QUALITY_FLOOR)
    assert m.measured is False
