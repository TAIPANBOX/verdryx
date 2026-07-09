"""Tests for verdryx.models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from verdryx.models import (
    DEFAULT_OUTCOME_SCORES,
    OUTCOME_ABANDONED,
    OUTCOME_ESCALATED,
    OUTCOME_RESOLVED,
    Baseline,
    CostPerOutcomeReport,
    DriftReport,
    EvalCase,
    EvalRun,
    EvalSet,
    GradeResult,
    GraderKind,
    OutcomeCost,
    Score,
)

# ------------------------------------------------------------------
# EvalCase / EvalSet
# ------------------------------------------------------------------


def test_eval_case_defaults() -> None:
    case = EvalCase(id="c1", prompt="hello")
    assert case.expected is None
    assert case.rubric is None
    assert case.grader == GraderKind.EXACT


def test_eval_case_round_trip_dict() -> None:
    case = EvalCase(id="c1", prompt="hello", expected="hi", grader=GraderKind.REGEX)
    restored = EvalCase.from_dict(case.to_dict())
    assert restored == case


def test_eval_case_round_trip_dict_with_rubric() -> None:
    case = EvalCase(id="c1", prompt="hello", rubric="be nice", grader=GraderKind.LLM_JUDGE)
    restored = EvalCase.from_dict(case.to_dict())
    assert restored == case


def test_eval_case_from_dict_defaults_grader_to_exact() -> None:
    case = EvalCase.from_dict({"id": "c1", "prompt": "hello"})
    assert case.grader == GraderKind.EXACT


def test_eval_case_to_dict_omits_unset_optional_fields() -> None:
    case = EvalCase(id="c1", prompt="hello")
    data = case.to_dict()
    assert "expected" not in data
    assert "rubric" not in data


def test_eval_set_round_trip_dict() -> None:
    evalset = EvalSet(id="s1", cases=[EvalCase(id="c1", prompt="hi")])
    restored = EvalSet.from_dict(evalset.to_dict())
    assert restored == evalset


def test_eval_set_defaults_to_no_cases() -> None:
    evalset = EvalSet(id="s1")
    assert evalset.cases == []


def test_eval_set_save_and_load_round_trips(tmp_path) -> None:
    evalset = EvalSet(
        id="s1",
        cases=[
            EvalCase(id="c1", prompt="hi", expected="hi", grader=GraderKind.EXACT),
            EvalCase(id="c2", prompt="pattern here", expected="pat.*", grader=GraderKind.REGEX),
            EvalCase(id="c3", prompt="case_resolved", grader=GraderKind.OUTCOME_TAG),
        ],
    )
    path = tmp_path / "evalset.json"
    evalset.save(path)
    loaded = EvalSet.load(path)
    assert loaded == evalset


def test_eval_set_load_reads_plain_json_shape(tmp_path) -> None:
    """The JSON shape documented in README.md, not round-tripped through
    to_dict()/save(), still loads correctly."""
    path = tmp_path / "evalset.json"
    path.write_text(
        '{"id": "support-tier1-v1", "cases": '
        '[{"id": "c1", "prompt": "hi", "expected": "hi", "grader": "exact"}]}'
    )
    evalset = EvalSet.load(path)
    assert evalset.id == "support-tier1-v1"
    assert evalset.cases == [EvalCase(id="c1", prompt="hi", expected="hi", grader=GraderKind.EXACT)]


# ------------------------------------------------------------------
# GradeResult / Score
# ------------------------------------------------------------------


def test_grade_result_to_score_attaches_case_id() -> None:
    result = GradeResult(value=0.75, tokens=42, cost_usd=0.01)
    score = result.to_score("case-1")
    assert score == Score(case_id="case-1", value=0.75, tokens=42, cost_usd=0.01)


def test_grade_result_defaults() -> None:
    result = GradeResult(value=1.0)
    assert result.tokens == 0
    assert result.cost_usd == 0.0


# ------------------------------------------------------------------
# EvalRun
# ------------------------------------------------------------------


def test_eval_run_mean_score_and_totals_are_zero_when_empty() -> None:
    run = EvalRun(id="r1", model="stub", started_at=datetime.now(tz=UTC))
    assert run.mean_score == 0.0
    assert run.total_tokens == 0
    assert run.total_cost_usd == 0.0


def test_eval_run_mean_score_and_totals() -> None:
    run = EvalRun(
        id="r1",
        model="stub",
        started_at=datetime.now(tz=UTC),
        scores=[
            Score(case_id="a", value=1.0, tokens=10, cost_usd=0.1),
            Score(case_id="b", value=0.0, tokens=20, cost_usd=0.2),
        ],
    )
    assert run.mean_score == pytest.approx(0.5)
    assert run.total_tokens == 30
    assert run.total_cost_usd == pytest.approx(0.3)


def test_eval_run_finished_at_defaults_to_none() -> None:
    run = EvalRun(id="r1", model="stub", started_at=datetime.now(tz=UTC))
    assert run.finished_at is None


# ------------------------------------------------------------------
# Baseline / DriftReport
# ------------------------------------------------------------------


def test_baseline_label_defaults_to_empty_string() -> None:
    baseline = Baseline(id="b1", eval_run_id="r1", mean_score=0.9, created_at=datetime.now(tz=UTC))
    assert baseline.label == ""


def test_drift_report_construction() -> None:
    report = DriftReport(
        baseline_id="b1", window=3, mean_score=0.8, delta=-0.1, verdict="regressed"
    )
    assert report.verdict == "regressed"
    assert report.window == 3


# ------------------------------------------------------------------
# Outcome-tag vocabulary and CostPerOutcomeReport
# ------------------------------------------------------------------


def test_default_outcome_scores_map() -> None:
    assert DEFAULT_OUTCOME_SCORES == {
        OUTCOME_RESOLVED: 1.0,
        OUTCOME_ESCALATED: 0.5,
        OUTCOME_ABANDONED: 0.0,
    }


def test_cost_per_outcome_report_named_accessors() -> None:
    report = CostPerOutcomeReport(
        by_outcome={OUTCOME_RESOLVED: OutcomeCost(OUTCOME_RESOLVED, 2, 0.3, 0.15)},
        overall=OutcomeCost("overall", 2, 0.3, 0.15),
    )
    assert report.resolved is not None
    assert report.resolved.count == 2
    assert report.escalated is None
    assert report.abandoned is None
    assert report.get(OUTCOME_RESOLVED) is report.by_outcome[OUTCOME_RESOLVED]
    assert report.get("nonexistent-tag") is None


def test_grader_kind_values_round_trip_through_string() -> None:
    for kind in GraderKind:
        assert GraderKind(kind.value) is kind
