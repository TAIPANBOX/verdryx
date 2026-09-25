"""Tests for examples/bakeoff (the judge bake-off harness): dataset
determinism and truth-by-construction, the metric functions against
hand-computed values, unanswered-never-scored, and the claude spend guard.

Collected by the repository's own pytest (testpaths = ["tests"] in
pyproject.toml); `examples/bakeoff` is importable because pytest's default
"prepend" import mode walks up from this file through tests/__init__.py to
the repository root and puts that root on sys.path -- the same reason
`from verdryx...` imports work in every other file here. `examples/` and
`examples/bakeoff/` each carry an (otherwise-empty) __init__.py so
`import examples.bakeoff...` resolves as an ordinary package import, no
path hacking needed.

Every test below was run red first against a stubbed-out implementation
(functions replaced with ones that raise or return an obviously wrong
value); see the harness report for the transcript. `typryx_fake` is the
shared fixture in tests/conftest.py: a real loopback HTTP server standing
in for typryx, already used by tests/test_cli.py and tests/test_graders.py
for the same TypedGrader/TypryxClient path this file exercises for the
bake-off's own driver.
"""

from __future__ import annotations

import json
import math
import socket

import pytest

from examples.bakeoff import dataset, metrics, run
from verdryx.cli import run_eval
from verdryx.graders import LLMJudgeGrader, StubLLMAdapter, TypedUnanswered
from verdryx.models import EvalCase, GraderKind
from verdryx.pricing import PriceBook

# ------------------------------------------------------------------
# dataset.py: determinism, family sizes, truth-by-construction, balance.
# ------------------------------------------------------------------


def test_generate_is_deterministic_for_the_same_seed() -> None:
    a = dataset.generate(200, seed=7)
    b = dataset.generate(200, seed=7)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_generate_differs_for_a_different_seed() -> None:
    a = dataset.generate(200, seed=7)
    b = dataset.generate(200, seed=8)
    assert [c.to_dict() for c in a] != [c.to_dict() for c in b]


def test_family_sizes_match_the_1000_case_default_mix() -> None:
    assert dataset.family_sizes(1000) == {"arithmetic": 400, "routing": 400, "format": 200}


def test_family_sizes_rejects_odd_n() -> None:
    with pytest.raises(ValueError, match="even"):
        dataset.family_sizes(201)


def test_family_sizes_rejects_n_too_small_for_a_balanced_dataset() -> None:
    with pytest.raises(ValueError, match="too small"):
        dataset.family_sizes(2)


@pytest.mark.parametrize("n", [12, 60, 200, 1000])
def test_family_sizes_sum_to_n_and_are_all_even(n: int) -> None:
    sizes = dataset.family_sizes(n)
    assert sum(sizes.values()) == n
    assert all(v % 2 == 0 for v in sizes.values())


def test_generate_matches_the_default_mix_counts_at_n_1000() -> None:
    cases = dataset.generate(1000, seed=1)
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.family] = counts.get(c.family, 0) + 1
    assert counts == {"arithmetic": 400, "routing": 400, "format": 200}


def test_generate_is_balanced_true_false_within_every_family() -> None:
    cases = dataset.generate(200, seed=3)
    by_family: dict[str, list[bool]] = {}
    for c in cases:
        by_family.setdefault(c.family, []).append(c.truth)
    for family, truths in by_family.items():
        assert sum(truths) == len(truths) - sum(truths), f"{family} is not balanced: {truths}"


def _parse_arithmetic_task(task: str) -> tuple[int, int]:
    prefix, suffix = "What is ", "?"
    assert task.startswith(prefix) and task.endswith(suffix), task
    a_str, b_str = task[len(prefix) : -len(suffix)].split(" times ")
    return int(a_str), int(b_str)


def test_arithmetic_true_cases_really_are_the_right_product() -> None:
    cases = [c for c in dataset.generate(400, seed=11) if c.family == "arithmetic" and c.truth]
    assert cases
    for c in cases:
        a, b = _parse_arithmetic_task(c.task)
        assert int(c.final_answer) == a * b


def test_arithmetic_false_cases_really_are_the_wrong_product() -> None:
    cases = [c for c in dataset.generate(400, seed=11) if c.family == "arithmetic" and not c.truth]
    assert cases
    for c in cases:
        a, b = _parse_arithmetic_task(c.task)
        assert int(c.final_answer) != a * b


def _parse_format_task(task: str) -> tuple[str, str]:
    for prefix, op in (
        ("Uppercase this word: ", "upper"),
        ("Count the letters in this word: ", "count"),
        ("Reverse this word: ", "reverse"),
    ):
        if task.startswith(prefix):
            return op, task[len(prefix) :]
    raise AssertionError(f"unrecognized format task: {task!r}")


def _format_correct_answer(op: str, word: str) -> str:
    if op == "upper":
        return word.upper()
    if op == "count":
        return str(len(word))
    return word[::-1]


def test_format_true_cases_really_are_correct() -> None:
    cases = [c for c in dataset.generate(400, seed=5) if c.family == "format" and c.truth]
    assert cases
    for c in cases:
        op, word = _parse_format_task(c.task)
        assert c.final_answer == _format_correct_answer(op, word)


def test_format_false_cases_really_are_wrong() -> None:
    cases = [c for c in dataset.generate(400, seed=5) if c.family == "format" and not c.truth]
    assert cases
    for c in cases:
        op, word = _parse_format_task(c.task)
        assert c.final_answer != _format_correct_answer(op, word)


def _true_team_for_ticket(task: str) -> str:
    matches = [
        team for team, phrases in dataset._TEAM_PHRASES.items() if any(p in task for p in phrases)
    ]
    assert len(matches) == 1, f"expected exactly one matching team, got {matches} for {task!r}"
    return matches[0]


def test_routing_true_cases_answer_is_the_real_team() -> None:
    cases = [c for c in dataset.generate(400, seed=9) if c.family == "routing" and c.truth]
    assert cases
    for c in cases:
        assert c.final_answer == _true_team_for_ticket(c.task)


def test_routing_false_cases_answer_is_a_different_real_team() -> None:
    cases = [c for c in dataset.generate(400, seed=9) if c.family == "routing" and not c.truth]
    assert cases
    for c in cases:
        assert c.final_answer in dataset._TEAMS
        assert c.final_answer != _true_team_for_ticket(c.task)


def test_write_jsonl_then_run_load_cases_round_trips(tmp_path) -> None:
    cases = dataset.generate(20, seed=2)
    path = tmp_path / "cases.jsonl"
    dataset.write_jsonl(cases, path)
    loaded = run.load_cases(path)
    assert len(loaded) == 20
    assert loaded[0]["id"] == cases[0].id
    assert loaded[0]["truth"] == cases[0].truth


def test_load_cases_names_the_missing_field(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "family": "arithmetic"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task"):
        run.load_cases(path)


def test_dataset_main_writes_the_requested_number_of_cases(tmp_path) -> None:
    out = tmp_path / "d.jsonl"
    rc = dataset.main(["--n", "20", "--seed", "5", "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    row = json.loads(lines[0])
    assert set(row) == {"id", "family", "task", "final_answer", "truth"}


# ------------------------------------------------------------------
# metrics.py: every function against a hand-computed value.
# ------------------------------------------------------------------


def test_accuracy_at_hand_computed() -> None:
    # predicted (p>=0.5): True, False, True, False; truth: True, False, False, True
    pairs = [(0.9, True), (0.2, False), (0.6, False), (0.4, True)]
    result = metrics.accuracy_at(pairs)
    assert result.n == 4
    assert result.accuracy == pytest.approx(0.5)
    assert result.passed_wrong == 1  # (0.6, False): predicted True, truth False
    assert result.failed_right == 1  # (0.4, True): predicted False, truth True


def test_accuracy_at_empty_is_nan_not_zero() -> None:
    result = metrics.accuracy_at([])
    assert result.n == 0
    assert math.isnan(result.accuracy)


def test_brier_hand_computed() -> None:
    pairs = [(1.0, True), (0.0, False), (0.5, True), (0.5, False)]
    # (0)^2 + (0)^2 + (0.25) + (0.25), mean over 4 = 0.125
    assert metrics.brier(pairs) == pytest.approx(0.125)


def test_mean_confidence_hand_computed() -> None:
    pairs = [(0.9, True), (0.1, False), (0.5, True)]
    assert metrics.mean_confidence(pairs) == pytest.approx((0.9 + 0.9 + 0.5) / 3)


def test_ece_hand_computed_two_equal_width_bins() -> None:
    # bin [0, 0.5): (0.1, False), (0.2, True) -> mean_p=0.15, freq_true=0.5, |diff|=0.35
    # bin [0.5, 1.0]: (0.9, True), (0.8, False) -> mean_p=0.85, freq_true=0.5, |diff|=0.35
    # each bin holds half the points, so ECE = 0.5*0.35 + 0.5*0.35 = 0.35
    pairs = [(0.1, False), (0.2, True), (0.9, True), (0.8, False)]
    assert metrics.ece(pairs, bins=2) == pytest.approx(0.35)


def test_ece_perfectly_calibrated_is_zero() -> None:
    pairs = [(1.0, True)] * 5 + [(0.0, False)] * 5
    assert metrics.ece(pairs, bins=10) == pytest.approx(0.0)


def test_percentile_median_hand_computed() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    # rank = 0.5 * 3 = 1.5 -> halfway between values[1]=20 and values[2]=30
    assert metrics.percentile(values, 50) == pytest.approx(25.0)


def test_percentile_endpoints() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert metrics.percentile(values, 0) == pytest.approx(10.0)
    assert metrics.percentile(values, 100) == pytest.approx(40.0)


def test_percentile_p95_hand_computed() -> None:
    values = [float(x) for x in range(1, 101)]  # 1..100
    # rank = 0.95 * 99 = 94.05 -> values[94]=95, values[95]=96, frac 0.05
    assert metrics.percentile(values, 95) == pytest.approx(95.05)


def test_percentile_rejects_out_of_range_pct() -> None:
    with pytest.raises(ValueError):
        metrics.percentile([1.0], 101)


def test_agreement_hand_computed() -> None:
    a = {"c1": True, "c2": False, "c3": True}
    b = {"c1": True, "c2": True, "c4": False}
    # intersection {c1, c2}: c1 matches, c2 does not -> 1/2
    result = metrics.agreement(a, b)
    assert result.n == 2
    assert result.agreement == pytest.approx(0.5)


def test_agreement_empty_intersection_is_nan_not_zero() -> None:
    result = metrics.agreement({"a": True}, {"b": False})
    assert result.n == 0
    assert math.isnan(result.agreement)


def test_cost_per_1000_answered_hand_computed() -> None:
    assert metrics.cost_per_1000_answered(2.0, 500) == pytest.approx(4.0)


def test_cost_per_1000_answered_zero_answered_is_nan_not_zero() -> None:
    assert math.isnan(metrics.cost_per_1000_answered(0.0, 0))


# ------------------------------------------------------------------
# run.py: unanswered is counted apart, never scored (over a real loopback
# typryx_fake server, no mocking of urllib -- see tests/conftest.py).
# ------------------------------------------------------------------


def test_grade_judge_typed_counts_unanswered_and_never_scores_it(typryx_fake) -> None:
    typryx_fake.script(
        "/v1/ask",
        200,
        {
            "answer_id": "a1",
            "template": "eval.outcome_met",
            "template_version": "v1",
            "type": "noul",
            "probabilities": {"true": 0.9, "false": 0.1},
            "backend": "stub",
            "model": "stub-0",
        },
    )
    typryx_fake.script(
        "/v1/ask",
        200,
        {
            "answer_id": "a2",
            "template": "eval.outcome_met",
            "template_version": "v1",
            "type": "noul",
            "unanswered": True,
            "reason": "label_mass_too_low",
            "backend": "stub",
            "model": "stub-0",
        },
    )
    typryx_fake.script("/v1/outcome", 200, {"ok": True})

    cases = [
        {
            "id": "c1", "family": "format", "task": "Uppercase this word: hi",
            "final_answer": "HI", "truth": True,
        },
        {
            "id": "c2", "family": "format", "task": "Uppercase this word: yo",
            "final_answer": "yo", "truth": False,
        },
    ]  # fmt: skip
    judge_cfg = {
        "name": "stubjudge",
        "kind": "typed",
        "typed_url": typryx_fake.url,
        "typed_key": "k1",
    }
    rows = run.grade_judge(judge_cfg, cases)

    assert [r["case_id"] for r in rows] == ["c1", "c2"]
    scored = [r for r in rows if not r["unanswered"]]
    unanswered = [r for r in rows if r["unanswered"]]
    assert [r["case_id"] for r in scored] == ["c1"]
    assert [r["case_id"] for r in unanswered] == ["c2"]
    assert unanswered[0]["reason"] == "label_mass_too_low"
    assert unanswered[0]["value"] is None
    assert unanswered[0]["predicted"] is None

    summary = run.summarize(rows)
    assert summary["n_asked"] == 2
    assert summary["n_answered"] == 1
    assert summary["n_unanswered"] == 1
    assert summary["unanswered_reasons"] == {"label_mass_too_low": 1}
    # Only the answered case (correctly predicted true) feeds accuracy.
    assert summary["accuracy"] == pytest.approx(1.0)


def test_grade_judge_typed_captures_backend_model_template_and_cost(typryx_fake) -> None:
    typryx_fake.script(
        "/v1/ask",
        200,
        {
            "answer_id": "a1",
            "template": "eval.outcome_met",
            "template_version": "v3",
            "type": "noul",
            "probabilities": {"true": 0.8, "false": 0.2},
            "backend": "openai-logprobs",
            "model": "qwen2.5:7b",
            "cost_usd": 0.0001,
        },
    )
    typryx_fake.script("/v1/outcome", 200, {"ok": True})
    cases = [
        {
            "id": "c1", "family": "arithmetic", "task": "What is 3 times 4?",
            "final_answer": "12", "truth": True,
        }
    ]  # fmt: skip
    judge_cfg = {"name": "qwen", "kind": "typed", "typed_url": typryx_fake.url, "typed_key": "k1"}
    rows = run.grade_judge(judge_cfg, cases)

    assert rows[0]["backend"] == "openai-logprobs"
    assert rows[0]["model"] == "qwen2.5:7b"
    assert rows[0]["template_version"] == "v3"
    assert rows[0]["cost_usd"] == pytest.approx(0.0001)
    assert rows[0]["value"] == pytest.approx(0.8)
    assert rows[0]["predicted"] is True
    assert rows[0]["correct"] is True

    assert typryx_fake.requests[0]["key"] == "k1"
    assert typryx_fake.requests[0]["body"]["template"] == "eval.outcome_met"
    assert typryx_fake.requests[0]["body"]["state"] == {
        "task": "What is 3 times 4?",
        "final_answer": "12",
    }


def test_timing_grader_records_latency_and_reraises_typed_unanswered() -> None:
    class Boom:
        def grade(self, case: EvalCase, output: str) -> None:
            raise TypedUnanswered(case.id, "aid", "some_reason")

    timing = run.TimingGrader(Boom())
    case = EvalCase(id="x", prompt="p", grader=GraderKind.TYPED)
    with pytest.raises(TypedUnanswered):
        timing.grade(case, "out")
    assert "x" in timing.latencies
    assert timing.latencies["x"] >= 0.0


def test_run_eval_with_llm_judge_via_stub_adapter_offline() -> None:
    """Exercises the exact shape grade_judge() uses for kind='llm_judge'
    (build_evalset + TimingGrader + LLMJudgeGrader via run_eval), with
    StubLLMAdapter standing in for AnthropicAdapter -- no network call, no
    Anthropic import, matching this task's "spend nothing" rule."""
    cases = [
        {
            "id": "c1", "family": "format", "task": "Uppercase this word: hi",
            "final_answer": "HI", "truth": True,
        }
    ]  # fmt: skip
    evalset = run.build_evalset(cases, GraderKind.LLM_JUDGE, rubric=run.CLAUDE_RUBRIC)
    stub_adapter = StubLLMAdapter(judge_value=0.75, cost_usd=0.002)
    timing = run.TimingGrader(LLMJudgeGrader(stub_adapter))
    replay = run.Replay([c["final_answer"] for c in cases])

    result = run_eval(evalset, replay, model="stub", graders={GraderKind.LLM_JUDGE: timing})

    assert len(result.scores) == 1
    assert result.scores[0].value == pytest.approx(0.75)
    assert result.scores[0].cost_usd == pytest.approx(0.002)
    assert "c1" in timing.latencies
    assert stub_adapter.judgements[0][2] == run.CLAUDE_RUBRIC  # the rubric reached the adapter


def test_cross_judge_agreement_hand_computed() -> None:
    rows_a = [
        {"case_id": "c1", "predicted": True, "unanswered": False},
        {"case_id": "c2", "predicted": False, "unanswered": False},
        {"case_id": "c3", "predicted": True, "unanswered": True},
    ]
    rows_b = [
        {"case_id": "c1", "predicted": True, "unanswered": False},
        {"case_id": "c2", "predicted": True, "unanswered": False},
    ]
    result = run.cross_judge_agreement({"a": rows_a, "b": rows_b})
    assert result["a_vs_b"]["n"] == 2
    assert result["a_vs_b"]["agreement"] == pytest.approx(0.5)


# ------------------------------------------------------------------
# The report: no verdict word, n beside every ratio.
# ------------------------------------------------------------------

_BANNED_VERDICT_WORDS = (
    "recommend",
    "should use",
    "winner",
    "best judge",
    "verdict:",
    "verdict is",
)


def _fabricated_rows(judge: str, n: int) -> list[dict]:
    return [
        {
            "case_id": f"c{i}", "family": "arithmetic", "truth": True, "judge": judge,
            "value": 0.9, "predicted": True, "correct": True, "cost_usd": 0.0,
            "unanswered": False, "reason": None, "latency_s": 0.01,
            "backend": "stub", "model": "stub-0", "template_version": "v1",
        }
        for i in range(n)
    ]  # fmt: skip


def test_report_carries_no_verdict_word() -> None:
    md, _data = run.render_report(
        {"stub": _fabricated_rows("stub", 4)},
        dataset_info={"source": "generated", "seed": 1, "n": 4, "mix": {"arithmetic": 4}},
        verdryx_commit="deadbeef",
        typryx_commit="2907caf",
    )
    lowered = md.lower()
    for banned in _BANNED_VERDICT_WORDS:
        assert banned not in lowered, f"report contains a verdict word: {banned!r}"


def test_report_states_n_beside_the_accuracy_ratio() -> None:
    md, data = run.render_report(
        {"stub": _fabricated_rows("stub", 4)},
        dataset_info={"source": "generated", "seed": 1, "n": 4, "mix": {"arithmetic": 4}},
        verdryx_commit="deadbeef",
        typryx_commit="2907caf",
    )
    assert "(n=4)" in md
    assert data["judges"]["stub"]["overall"]["n_answered"] == 4
    assert "not proven" in md.lower()
    assert data["not_proven"]


def test_report_external_dataset_says_so() -> None:
    md, data = run.render_report(
        {"stub": _fabricated_rows("stub", 2)},
        dataset_info={"source": "external", "path": "/tmp/real-labels.jsonl", "n": 2},
        verdryx_commit="deadbeef",
        typryx_commit="2907caf",
    )
    assert "external file" in md
    assert data["dataset"]["source"] == "external"


# ------------------------------------------------------------------
# The claude spend guard: refuses without confirmation, no network either
# way.
# ------------------------------------------------------------------


def test_require_claude_confirmation_refuses_without_the_env_var(monkeypatch, capsys) -> None:
    monkeypatch.delenv("BAKEOFF_CONFIRM_SPEND", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        run.require_claude_confirmation(500, "claude-haiku-4-5-20251001")
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "claude judge estimate" in out
    assert "refusing" in out


def test_require_claude_confirmation_proceeds_when_confirmed(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BAKEOFF_CONFIRM_SPEND", "yes")
    run.require_claude_confirmation(500, "claude-haiku-4-5-20251001")  # must not raise
    out = capsys.readouterr().out
    assert "proceeding" in out


def test_require_claude_confirmation_never_opens_a_socket(monkeypatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("require_claude_confirmation must never open a socket")

    monkeypatch.setattr(socket, "socket", _boom)

    monkeypatch.delenv("BAKEOFF_CONFIRM_SPEND", raising=False)
    with pytest.raises(SystemExit):
        run.require_claude_confirmation(10, "claude-haiku-4-5-20251001")

    monkeypatch.setenv("BAKEOFF_CONFIRM_SPEND", "yes")
    run.require_claude_confirmation(10, "claude-haiku-4-5-20251001")


def test_estimate_claude_cost_uses_verdryx_price_book() -> None:
    input_tokens, output_tokens, cost = run.estimate_claude_cost(100, "claude-haiku-4-5-20251001")
    assert input_tokens == run.ESTIMATED_INPUT_TOKENS_PER_CASE * 100
    assert output_tokens == run.ESTIMATED_OUTPUT_TOKENS_PER_CASE * 100
    expected_cost = PriceBook.default().price(
        "claude-haiku-4-5-20251001", input_tokens, output_tokens
    )
    assert cost == pytest.approx(expected_cost)


def test_cmd_claude_estimate_cli_exits_1_without_confirmation(monkeypatch) -> None:
    monkeypatch.delenv("BAKEOFF_CONFIRM_SPEND", raising=False)
    assert run.main(["claude-estimate", "--n", "50"]) == 1


def test_cmd_claude_estimate_cli_exits_0_when_confirmed(monkeypatch) -> None:
    monkeypatch.setenv("BAKEOFF_CONFIRM_SPEND", "yes")
    assert run.main(["claude-estimate", "--n", "50"]) == 0
