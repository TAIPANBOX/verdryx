"""Tests for the Verdryx CLI (verdryx.cli).

`run_eval` (the non-argparse core of the `eval` command) and `main()` (the
full argparse-driven CLI) are both exercised here, almost always against
StubLLMAdapter or `--model stub`, so this file makes no network call. The one
exception (the completion-pricing regression test below) uses AnthropicAdapter
with its `_get_client` patched out, the same network-free technique
test_graders.py uses throughout.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from verdryx import __version__
from verdryx.cli import main, run_eval
from verdryx.graders import AnthropicAdapter, StubLLMAdapter
from verdryx.models import Baseline, EvalCase, EvalSet, GraderKind
from verdryx.store import Store

#: A minimal, provider-shape tool definition, reused across the
#: GraderKind.TOOL_TRACE tests below.
_LOOKUP_ORDER_TOOL = {
    "name": "lookup_order",
    "description": "Look up an order by id",
    "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}


class _NoToolsAdapter:
    """A minimal third-party LLMAdapter that implements complete()/judge()
    but not complete_with_tools() -- stands in for a hand-rolled adapter
    that predates GraderKind.TOOL_TRACE, to exercise run_eval's
    adapter-missing-method death path."""

    def complete(self, prompt: str) -> tuple[str, int, float]:
        return "output", 0, 0.0

    def judge(self, prompt: str, output: str, rubric: str) -> tuple[float, int, float]:
        return 1.0, 0, 0.0


# ------------------------------------------------------------------
# run_eval: the non-argparse core, exercised directly with a stub adapter
# ------------------------------------------------------------------


def test_run_eval_scores_every_case_and_computes_mean(sample_evalset) -> None:
    adapter = StubLLMAdapter()
    run = run_eval(sample_evalset, adapter, model="stub")
    assert run.model == "stub"
    assert len(run.scores) == len(sample_evalset.cases)
    assert {s.case_id for s in run.scores} == {c.id for c in sample_evalset.cases}
    assert run.mean_score == pytest.approx(1.0)
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at


def test_run_eval_outcome_tag_case_makes_no_completion_call() -> None:
    evalset = EvalSet(
        id="s", cases=[EvalCase(id="c1", prompt="escalated", grader=GraderKind.OUTCOME_TAG)]
    )
    adapter = StubLLMAdapter()
    run = run_eval(evalset, adapter, model="stub")
    assert adapter.completions == []
    assert run.scores[0].value == 0.5


def test_run_eval_non_outcome_tag_case_calls_complete_with_the_prompt() -> None:
    evalset = EvalSet(
        id="s", cases=[EvalCase(id="c1", prompt="hello there", expected="stub output")]
    )
    adapter = StubLLMAdapter()
    run_eval(evalset, adapter, model="stub")
    assert adapter.completions == ["hello there"]


def test_run_eval_raises_for_case_grader_with_no_configured_grader() -> None:
    evalset = EvalSet(
        id="s", cases=[EvalCase(id="c1", prompt="p", rubric="x", grader=GraderKind.LLM_JUDGE)]
    )
    adapter = StubLLMAdapter()
    with pytest.raises(ValueError, match="no grader configured"):
        run_eval(evalset, adapter, model="stub", graders={})


def test_run_eval_folds_completion_and_judge_tokens_into_one_score() -> None:
    evalset = EvalSet(
        id="s", cases=[EvalCase(id="c1", prompt="p", rubric="x", grader=GraderKind.LLM_JUDGE)]
    )
    adapter = StubLLMAdapter(tokens=5)
    run = run_eval(evalset, adapter, model="stub")
    # 5 tokens from complete() to produce the output + 5 from judge() to
    # grade it = 10.
    assert run.scores[0].tokens == 10


def test_run_eval_prices_the_completion_not_just_the_judge() -> None:
    """Regression test: previously Score.cost_usd only ever reflected an
    LLM_JUDGE grader's own judge() cost (0.0 for every other grader kind) --
    the model-under-evaluation's own completion call, real billed LLM usage
    for `verdryx eval --model <real-model>`, was never priced anywhere at
    all. This case uses GraderKind.EXACT (no judge involved whatsoever, so
    ExactGrader's own GradeResult.cost_usd is always 0.0), which makes any
    nonzero Score.cost_usd unambiguously attributable to the completion
    call. AnthropicAdapter's network client is mocked, mirroring
    test_graders.py's own technique -- no real call is made."""
    evalset = EvalSet(id="s", cases=[EvalCase(id="c1", prompt="say hi", expected="hello there")])
    adapter = AnthropicAdapter()  # default model: claude-haiku-4-5-20251001
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="hello there")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        run = run_eval(evalset, adapter, model="claude-haiku-4-5-20251001")

    assert run.scores[0].value == 1.0  # exact match, sanity check on grading itself
    # 10 * $1.00/Mtok + 1 * $5.00/Mtok = 0.00001 + 0.000005 = 0.000015
    assert run.scores[0].cost_usd == pytest.approx(0.000015)
    assert run.scores[0].cost_usd > 0
    assert run.total_cost_usd == pytest.approx(0.000015)


# ------------------------------------------------------------------
# run_eval: GraderKind.TOOL_TRACE cases
# ------------------------------------------------------------------


def test_run_eval_tool_trace_case_calls_complete_with_tools_not_complete() -> None:
    evalset = EvalSet(
        id="s",
        cases=[
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order"],
            )
        ],
    )
    adapter = StubLLMAdapter()
    run = run_eval(evalset, adapter, model="stub")
    assert adapter.tool_completions == [("handle it", [_LOOKUP_ORDER_TOOL])]
    assert adapter.completions == []  # tool_trace never calls plain complete()
    # StubLLMAdapter defaults to calling tools[0]'s name, an exact match
    # against expected_tools=["lookup_order"].
    assert run.scores[0].value == 1.0
    assert run.scores[0].case_id == "tools-1"


def test_run_eval_tool_trace_case_folds_completion_tokens_and_cost_into_score() -> None:
    evalset = EvalSet(
        id="s",
        cases=[
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order"],
            )
        ],
    )
    adapter = StubLLMAdapter(tokens=9)
    run = run_eval(evalset, adapter, model="stub")
    # ToolTraceGrader.grade_trace() reports zero tokens/cost of its own; the
    # completion's own tokens (StubLLMAdapter's cost_usd for
    # complete_with_tools is always 0.0, mirroring complete()) are what
    # land in Score.
    assert run.scores[0].tokens == 9
    assert run.scores[0].cost_usd == 0.0


def test_run_eval_tool_trace_case_partial_credit_scores_between_zero_and_one() -> None:
    evalset = EvalSet(
        id="s",
        cases=[
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order", "issue_refund"],
            )
        ],
    )
    adapter = StubLLMAdapter()  # defaults to ["lookup_order"] only
    run = run_eval(evalset, adapter, model="stub")
    assert 0.0 < run.scores[0].value < 1.0


def test_run_eval_mixed_tool_trace_and_exact_cases(sample_evalset) -> None:
    """A mixed eval set covering tool_trace alongside another grader kind:
    both cases score, independently, over one run."""
    evalset = EvalSet(
        id="s",
        cases=[
            *sample_evalset.cases,  # exact / regex / outcome_tag / llm_judge
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order"],
            ),
        ],
    )
    adapter = StubLLMAdapter()
    run = run_eval(evalset, adapter, model="stub")
    assert len(run.scores) == len(evalset.cases)
    assert run.mean_score == pytest.approx(1.0)
    by_id = {s.case_id: s for s in run.scores}
    assert by_id["tools-1"].value == 1.0


def test_run_eval_dies_clearly_when_adapter_lacks_complete_with_tools(capsys) -> None:
    """A custom third-party LLMAdapter lacking complete_with_tools() dies
    with a message naming both the adapter and the missing method, instead
    of failing deep in the loop with an opaque AttributeError."""
    evalset = EvalSet(
        id="s",
        cases=[
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order"],
            )
        ],
    )
    adapter = _NoToolsAdapter()
    with pytest.raises(SystemExit) as exc_info:
        run_eval(evalset, adapter, model="stub")
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "_NoToolsAdapter" in err
    assert "complete_with_tools" in err
    assert "tools-1" in err


# ------------------------------------------------------------------
# CLI: eval -> baseline -> drift -> cost-per-correct, end to end via main()
# ------------------------------------------------------------------


def test_eval_command_stores_a_run_and_prints_scores(
    sample_evalset_path, sample_evalset, tmp_path, capsys
) -> None:
    db = tmp_path / "store.db"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    out = capsys.readouterr().out
    assert "Eval run" in out
    assert "mean score: 1.000" in out

    with Store.open(db) as store:
        runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].model == "stub"
    assert len(runs[0].scores) == len(sample_evalset.cases)


def test_eval_command_mixed_tool_trace_and_exact_cases_lands_scores_in_store(
    tmp_path, capsys
) -> None:
    """End-to-end via main(): a mixed eval set (tool_trace + exact cases)
    graded with the stub adapter, no network anywhere, scores landing in
    the SQLite store exactly like any other grader kind's."""
    evalset = EvalSet(
        id="mixed-v1",
        cases=[
            EvalCase(
                id="exact-1", prompt="say hi", expected="stub output", grader=GraderKind.EXACT
            ),
            EvalCase(
                id="tools-1",
                prompt="handle it",
                grader=GraderKind.TOOL_TRACE,
                tools=[_LOOKUP_ORDER_TOOL],
                expected_tools=["lookup_order"],
            ),
        ],
    )
    evalset_path = tmp_path / "mixed.json"
    evalset.save(evalset_path)
    db = tmp_path / "store.db"

    main(["eval", str(evalset_path), "--model", "stub", "--db", str(db)])
    out = capsys.readouterr().out
    assert "mean score: 1.000" in out

    with Store.open(db) as store:
        runs = store.list_runs()
    assert len(runs) == 1
    assert len(runs[0].scores) == 2
    by_id = {s.case_id: s.value for s in runs[0].scores}
    assert by_id == {"exact-1": 1.0, "tools-1": 1.0}


def test_eval_command_with_no_cases_prints_no_cases(tmp_path, capsys) -> None:
    evalset_path = tmp_path / "empty.json"
    evalset_path.write_text('{"id": "empty", "cases": []}')
    db = tmp_path / "store.db"
    main(["eval", str(evalset_path), "--model", "stub", "--db", str(db)])
    out = capsys.readouterr().out
    assert "(no cases)" in out


def test_eval_command_with_events_emits_quality_score_and_eval_run(
    sample_evalset_path, sample_evalset, tmp_path, agent_id
) -> None:
    db = tmp_path / "store.db"
    events_path = tmp_path / "events.ndjson"
    main(
        [
            "eval",
            str(sample_evalset_path),
            "--model",
            "stub",
            "--db",
            str(db),
            "--events",
            str(events_path),
            "--agent-id",
            agent_id,
        ]
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    types = [e["type"] for e in events]
    assert types.count("quality_score") == len(sample_evalset.cases)
    assert types.count("eval_run") == 1
    eval_run_event = next(e for e in events if e["type"] == "eval_run")
    assert eval_run_event["data"]["cases"] == len(sample_evalset.cases)
    assert eval_run_event["data"]["mean_score"] == pytest.approx(1.0)
    assert all(e["agent_id"] == agent_id for e in events)


def test_eval_command_without_agent_id_emits_nothing(sample_evalset_path, tmp_path) -> None:
    db = tmp_path / "store.db"
    events_path = tmp_path / "events.ndjson"
    main(
        [
            "eval",
            str(sample_evalset_path),
            "--model",
            "stub",
            "--db",
            str(db),
            "--events",
            str(events_path),
        ]
    )
    assert not events_path.exists()


def test_baseline_and_drift_commands_end_to_end(sample_evalset_path, tmp_path, capsys) -> None:
    db = tmp_path / "store.db"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    capsys.readouterr()  # discard eval's own output

    with Store.open(db) as store:
        run_id = store.list_runs()[0].id

    main(["baseline", run_id, "--db", str(db), "--label", "v1"])
    out = capsys.readouterr().out
    assert "Baseline" in out
    assert "mean_score=1.000" in out

    with Store.open(db) as store:
        baselines = store.list_baselines()
        baseline_id = baselines[0].id
        assert baselines[0].label == "v1"

    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    capsys.readouterr()  # discard

    main(["drift", "--baseline", baseline_id, "--db", str(db), "--window", "5"])
    out = capsys.readouterr().out
    assert "verdict:    on-track" in out


def test_drift_command_prints_significance_fields_when_baseline_run_has_enough_scores(
    sample_evalset_path, sample_evalset, tmp_path, capsys
) -> None:
    # _cmd_drift already loads the baseline's own run to filter recent runs
    # by model; this pins that it's actually threaded into compute_drift
    # (not just used for the model filter), so the CLI prints the
    # significance block instead of silently dropping it.
    db = tmp_path / "store.db"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    capsys.readouterr()

    with Store.open(db) as store:
        run_id = store.list_runs()[0].id
    main(["baseline", run_id, "--db", str(db)])
    capsys.readouterr()
    with Store.open(db) as store:
        baseline_id = store.list_baselines()[0].id

    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    capsys.readouterr()

    main(["drift", "--baseline", baseline_id, "--db", str(db)])
    out = capsys.readouterr().out
    assert f"n={len(sample_evalset.cases)} baseline cases" in out
    assert "CI on delta:" in out
    assert "t-statistic:" in out


def test_drift_command_regressed_emits_quality_drift_event(
    sample_evalset_path, tmp_path, agent_id
) -> None:
    db = tmp_path / "store.db"
    events_path = tmp_path / "events.ndjson"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])

    with Store.open(db) as store:
        run_id = store.list_runs()[0].id
        # Fabricate an artificially high baseline so the real (perfect,
        # mean_score == 1.0) stub run still reads as a regression.
        store.set_baseline(
            Baseline(
                id="too-high", eval_run_id=run_id, mean_score=1.5, created_at=datetime.now(tz=UTC)
            )
        )

    main(
        [
            "drift",
            "--baseline",
            "too-high",
            "--db",
            str(db),
            "--events",
            str(events_path),
            "--agent-id",
            agent_id,
        ]
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "quality_drift"
    assert events[0]["severity"] == "high"
    assert events[0]["data"]["verdict"] == "regressed"


def test_drift_command_on_track_emits_no_event(sample_evalset_path, tmp_path, agent_id) -> None:
    db = tmp_path / "store.db"
    events_path = tmp_path / "events.ndjson"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    with Store.open(db) as store:
        run_id = store.list_runs()[0].id
        store.set_baseline(
            Baseline(id="b1", eval_run_id=run_id, mean_score=1.0, created_at=datetime.now(tz=UTC))
        )

    main(
        [
            "drift",
            "--baseline",
            "b1",
            "--db",
            str(db),
            "--events",
            str(events_path),
            "--agent-id",
            agent_id,
        ]
    )
    assert not events_path.exists()


def test_drift_command_unknown_baseline_dies_cleanly(tmp_path, capsys) -> None:
    db = tmp_path / "store.db"
    with pytest.raises(SystemExit) as exc_info:
        main(["drift", "--baseline", "nonexistent", "--db", str(db)])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "no such baseline" in err


def _insert_dangling_baseline(
    db_path: Path, baseline_id: str, eval_run_id: str, mean_score: float
) -> None:
    """Insert a Baseline row whose eval_run_id does not (or no longer) exist
    in eval_runs, bypassing Store.set_baseline() -- which, now that
    verdryx.store._configure_connection turns PRAGMA foreign_keys on,
    refuses to create this state through the normal API. A raw sqlite3
    connection defaults to foreign_keys OFF, so it can still write this row
    directly, simulating a baseline that outlived its source run (e.g. a
    hand-edited or pre-existing database file predating the FK pragma) --
    exactly the state verdryx.cli._cmd_drift's own defensive check guards
    against, independent of the FK pragma stopping *new* instances of it.
    """
    with Store.open(db_path):
        pass  # create the schema, then close -- nothing to save yet.
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute(
            "INSERT INTO baselines (id, eval_run_id, mean_score, created_at, label) "
            "VALUES (?, ?, ?, ?, ?)",
            (baseline_id, eval_run_id, mean_score, datetime.now(tz=UTC).isoformat(), ""),
        )
        raw.commit()
    finally:
        raw.close()


def test_drift_command_baseline_with_missing_source_run_dies_cleanly(tmp_path, capsys) -> None:
    """A baseline whose eval_run_id points at a run that isn't in the store
    (never saved, or removed out of band) must die with a clear message
    naming the problem, rather than silently falling through to an
    unfiltered query across every model (see the dedicated regression test
    below for the cross-model consequence of that fallback)."""
    db = tmp_path / "store.db"
    _insert_dangling_baseline(db, "b1", "no-such-run", 0.9)

    with pytest.raises(SystemExit) as exc_info:
        main(["drift", "--baseline", "b1", "--db", str(db)])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "no-such-run" in err


def test_drift_command_dangling_baseline_never_pools_an_unrelated_model(
    sample_evalset_path, tmp_path, capsys
) -> None:
    """Regression test for the drift-across-models bug: previously, when a
    baseline's source eval run was gone, `baseline_run` came back None,
    `model_filter` fell back to None, and `list_runs(model=None)` pooled
    EVERY model -- so an unrelated-model run present in the store got
    silently scored as a drift verdict instead of the command dying. Here an
    artificially high baseline mean_score (1.5) stands in for "the baseline
    this dangling row used to represent"; if the buggy fallback fires, the
    unrelated `stub`-model run (mean_score 1.0) gets pooled in and produces a
    bogus 'regressed' verdict with no warning at all."""
    db = tmp_path / "store.db"
    _insert_dangling_baseline(db, "dangling", "no-such-run", 1.5)

    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    capsys.readouterr()  # discard eval's own output

    with pytest.raises(SystemExit) as exc_info:
        main(["drift", "--baseline", "dangling", "--db", str(db)])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "verdict" not in captured.out  # no cross-model verdict was ever printed
    assert "no-such-run" in captured.err


def test_baseline_command_unknown_run_dies_cleanly(tmp_path, capsys) -> None:
    db = tmp_path / "store.db"
    with pytest.raises(SystemExit) as exc_info:
        main(["baseline", "nonexistent-run", "--db", str(db)])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "no such eval run" in err


def test_cost_per_correct_command_prints_report(tmp_path, capsys) -> None:
    input_path = tmp_path / "outcomes.ndjson"
    input_path.write_text(
        '{"outcome": "case_resolved", "cost_usd": 0.1}\n{"outcome": "abandoned", "cost_usd": 0.2}\n'
    )
    main(["cost-per-correct", "--input", str(input_path)])
    out = capsys.readouterr().out
    assert "case_resolved" in out
    assert "overall" in out


def test_cost_per_correct_command_accepts_parquet_file_via_input(
    tmp_path, capsys, pyarrow_and_parquet
) -> None:
    pa, pq = pyarrow_and_parquet
    table = pa.table({"outcome": ["case_resolved"], "cost_microusd": [100_000]})
    path = tmp_path / "trace.parquet"
    pq.write_table(table, path)

    main(["cost-per-correct", "--input", str(path)])
    out = capsys.readouterr().out
    assert "case_resolved" in out
    assert "overall" in out


def test_cost_per_correct_command_accepts_traces_directory(
    tmp_path, capsys, pyarrow_and_parquet
) -> None:
    pa, pq = pyarrow_and_parquet
    table = pa.table(
        {
            "outcome": ["case_resolved", "escalated"],
            "cost_microusd": [100_000, 500_000],
        }
    )
    pq.write_table(table, tmp_path / "calls-00000000.parquet")

    main(["cost-per-correct", "--traces", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Cost per outcome" in out
    assert str(tmp_path) in out
    assert "case_resolved" in out
    assert "escalated" in out


def test_cost_per_correct_command_lists_untagged_bucket_last(
    tmp_path, capsys, pyarrow_and_parquet
) -> None:
    # UNTAGGED ("(untagged)") sorts alphabetically before any real tag on
    # its leading "(" -- the CLI must still print it last, matching
    # tokenfuse-core's own compute_outcomes row order.
    pa, pq = pyarrow_and_parquet
    table = pa.table(
        {
            "run_id": ["r1", "r2", "r3"],
            "step": [0, 0, 0],
            "outcome": ["zzz_last_tag", "", "aaa_first_tag"],
            "cost_microusd": [100_000, 200_000, 300_000],
        }
    )
    pq.write_table(table, tmp_path / "calls-00000000.parquet")

    main(["cost-per-correct", "--traces", str(tmp_path)])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    tags = ("aaa_first_tag", "zzz_last_tag", "(untagged)")
    positions = {tag: i for i, line in enumerate(lines) for tag in tags if tag in line}
    assert positions["aaa_first_tag"] < positions["(untagged)"]
    assert positions["zzz_last_tag"] < positions["(untagged)"]


def test_cost_per_correct_command_requires_input_or_traces() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["cost-per-correct"])
    assert exc_info.value.code == 2  # argparse's own usage-error exit code


def test_cost_per_correct_command_input_and_traces_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["cost-per-correct", "--input", "a.ndjson", "--traces", str(tmp_path)])
    assert exc_info.value.code == 2


def test_version_command_prints_version(capsys) -> None:
    main(["version"])
    out = capsys.readouterr().out.strip()
    assert out == __version__


def test_main_with_no_args_prints_help_and_exits_1(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "usage:" in out


def test_eval_command_requires_model_flag(sample_evalset_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["eval", str(sample_evalset_path)])
    assert exc_info.value.code == 2  # argparse's own usage-error exit code


def test_eval_command_model_stub_selects_stub_adapter_no_network(
    sample_evalset_path, tmp_path
) -> None:
    """Nothing in this test configures ANTHROPIC_API_KEY or mocks the
    anthropic module; if --model stub selected AnthropicAdapter instead of
    StubLLMAdapter this would raise ImportError or a network error."""
    db = tmp_path / "store.db"
    main(["eval", str(sample_evalset_path), "--model", "stub", "--db", str(db)])
    with Store.open(db) as store:
        assert len(store.list_runs()) == 1
