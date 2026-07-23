"""Tests for verdryx.models."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from verdryx.models import (
    DEFAULT_OUTCOME_SCORES,
    OUTCOME_ABANDONED,
    OUTCOME_ESCALATED,
    OUTCOME_RESOLVED,
    Baseline,
    Completion,
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

#: A minimal, provider-shape tool definition (Anthropic Messages API `tools`
#: shape), reused across the GraderKind.TOOL_TRACE validation tests below.
_A_TOOL = {
    "name": "lookup_order",
    "description": "Look up an order by id",
    "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}

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
# EvalCase tool_trace fields (GraderKind.TOOL_TRACE): validation matrix
# ------------------------------------------------------------------


def test_eval_case_tools_and_expected_tools_default_to_none() -> None:
    case = EvalCase(id="c1", prompt="hello")
    assert case.tools is None
    assert case.expected_tools is None


def test_eval_case_tool_trace_round_trip_dict() -> None:
    case = EvalCase(
        id="c1",
        prompt="handle it",
        grader=GraderKind.TOOL_TRACE,
        tools=[_A_TOOL],
        expected_tools=["lookup_order"],
    )
    restored = EvalCase.from_dict(case.to_dict())
    assert restored == case


def test_eval_case_tool_trace_legal_empty_expected_tools() -> None:
    """An empty expected_tools is legal: it means the model is expected to
    call no tools at all for that case."""
    case = EvalCase(
        id="c1",
        prompt="just say hi, no tools needed",
        grader=GraderKind.TOOL_TRACE,
        tools=[_A_TOOL],
        expected_tools=[],
    )
    restored = EvalCase.from_dict(case.to_dict())
    assert restored == case
    assert restored.expected_tools == []


def test_eval_case_tool_trace_requires_tools() -> None:
    with pytest.raises(ValueError, match=r"requires the field 'tools'"):
        EvalCase.from_dict(
            {"id": "c1", "prompt": "p", "grader": "tool_trace", "expected_tools": ["a"]}
        )


def test_eval_case_tool_trace_tools_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match=r"'tools' must not be empty"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": [],
                "expected_tools": ["a"],
            }
        )


def test_eval_case_tool_trace_tools_must_be_a_list() -> None:
    with pytest.raises(ValueError, match=r"'tools' must be a list"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": "not-a-list",
                "expected_tools": ["a"],
            }
        )


def test_eval_case_tool_trace_tools_item_must_be_an_object() -> None:
    with pytest.raises(ValueError, match=r"'tools'\[0\] must be an object"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": ["not-a-dict"],
                "expected_tools": ["a"],
            }
        )


def test_eval_case_tool_trace_requires_expected_tools() -> None:
    with pytest.raises(ValueError, match=r"requires the field 'expected_tools'"):
        EvalCase.from_dict({"id": "c1", "prompt": "p", "grader": "tool_trace", "tools": [_A_TOOL]})


def test_eval_case_tool_trace_expected_tools_must_be_a_list() -> None:
    with pytest.raises(ValueError, match=r"'expected_tools' must be a list"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": [_A_TOOL],
                "expected_tools": "lookup_order",
            }
        )


def test_eval_case_tool_trace_expected_tools_item_must_be_a_string() -> None:
    with pytest.raises(ValueError, match=r"'expected_tools'\[0\] must be a non-empty string"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": [_A_TOOL],
                "expected_tools": [123],
            }
        )


def test_eval_case_tool_trace_expected_tools_item_must_not_be_an_empty_string() -> None:
    with pytest.raises(ValueError, match=r"'expected_tools'\[0\] must be a non-empty string"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": [_A_TOOL],
                "expected_tools": [""],
            }
        )


def test_eval_case_tool_trace_expected_tools_second_item_is_named_by_index() -> None:
    with pytest.raises(ValueError, match=r"'expected_tools'\[1\] must be a non-empty string"):
        EvalCase.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "grader": "tool_trace",
                "tools": [_A_TOOL],
                "expected_tools": ["ok", ""],
            }
        )


def test_eval_case_tools_field_on_non_tool_trace_grader_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"'tools' is only valid with grader tool_trace"):
        EvalCase.from_dict({"id": "c1", "prompt": "p", "tools": [_A_TOOL]})


def test_eval_case_expected_tools_field_on_non_tool_trace_grader_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"'expected_tools' is only valid with grader tool_trace"):
        EvalCase.from_dict({"id": "c1", "prompt": "p", "expected_tools": ["a"]})


def test_eval_case_tools_field_on_explicit_exact_grader_is_also_an_error() -> None:
    """The 'wrong kind' check applies to every non-tool_trace grader, not
    just the implicit default -- pin it against an explicitly-named one."""
    with pytest.raises(ValueError, match=r"'tools' is only valid with grader tool_trace"):
        EvalCase.from_dict(
            {"id": "c1", "prompt": "p", "grader": "regex", "expected": "x", "tools": [_A_TOOL]}
        )


def test_eval_set_save_and_load_round_trips_a_tool_trace_case(tmp_path) -> None:
    evalset = EvalSet(
        id="s1",
        cases=[
            EvalCase(
                id="c1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_A_TOOL],
                expected_tools=["lookup_order"],
            )
        ],
    )
    path = tmp_path / "evalset.json"
    evalset.save(path)
    loaded = EvalSet.load(path)
    assert loaded == evalset


# ------------------------------------------------------------------
# Completion
# ------------------------------------------------------------------


def test_completion_fields() -> None:
    completion = Completion(
        text="hi there", tool_names=["lookup_order", "issue_refund"], tokens=42, cost_usd=0.001
    )
    assert completion.text == "hi there"
    assert completion.tool_names == ["lookup_order", "issue_refund"]
    assert completion.tokens == 42
    assert completion.cost_usd == pytest.approx(0.001)


def test_completion_is_frozen() -> None:
    completion = Completion(text="hi", tool_names=[], tokens=0, cost_usd=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        completion.text = "changed"


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
    # Significance fields default to "not computed", not zero/false.
    assert report.baseline_n == 0
    assert report.t_statistic is None
    assert report.ci_low is None
    assert report.ci_high is None


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


# ------------------------------------------------------------------
# eval-set validation: a clear error, not a raw traceback
# ------------------------------------------------------------------


def test_evalcase_missing_id_names_the_field():
    import pytest

    from verdryx.models import EvalCase

    with pytest.raises(ValueError, match="missing the required field 'id'"):
        EvalCase.from_dict({"prompt": "hi"})


def test_evalset_points_at_the_offending_case():
    import pytest

    from verdryx.models import EvalSet

    with pytest.raises(ValueError, match=r"case #2 .*missing the required field 'prompt'"):
        EvalSet.from_dict({"id": "s", "cases": [{"id": "c1", "prompt": "ok"}, {"id": "c2"}]})


def test_evalset_bad_grader_lists_the_choices():
    import pytest

    from verdryx.models import EvalSet

    with pytest.raises(ValueError, match="not one of exact, regex"):
        EvalSet.from_dict({"id": "s", "cases": [{"id": "c1", "prompt": "p", "grader": "nope"}]})


def test_load_names_the_file(tmp_path):
    import pytest

    from verdryx.models import EvalSet

    p = tmp_path / "broken.json"
    p.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.json:.*missing the required field 'id'"):
        EvalSet.load(p)


def test_load_reports_bad_json_with_the_path(tmp_path):
    import pytest

    from verdryx.models import EvalSet

    p = tmp_path / "notjson.json"
    p.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"notjson\.json: not valid JSON"):
        EvalSet.load(p)
