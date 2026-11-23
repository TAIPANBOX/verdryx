"""The LLM judge's score is read from one declared line, or not at all.

The judge used to take the FIRST number anywhere in the model's reply and
clamp it into [0, 1]. A reply such as "1 of 5 criteria met, 0.2" became a
perfect 1.0, and that score flowed into the run's mean, into drift and into
the error budget's quality floor, where a hold can be made from it.

Now the judge prompt declares one exact format, a reply that is the single
line ``SCORE: <number from 0 to 1>``, and the parser reads that format and
nothing else. Every other reply is a refusal with a reason: the case is
counted apart (``EvalRun.unanswered``, the same place an unanswered typed
case goes) and never becomes a score, and never a zero.

No test here makes a network call: AnthropicAdapter is driven through a
mocked client, the same technique tests/test_graders.py uses.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from verdryx.cli import run_eval
from verdryx.graders import (
    _JUDGE_SYSTEM_PROMPT,
    AnthropicAdapter,
    LLMJudgeGrader,
    StubLLMAdapter,
    _build_judge_message,
    _parse_judge_score,
)
from verdryx.models import EvalCase, EvalSet, GraderKind


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )


def _judge_with_reply(reply: str) -> float:
    adapter = AnthropicAdapter()
    client = MagicMock()
    client.messages.create.return_value = _response(reply)
    with patch.object(adapter, "_get_client", return_value=client):
        value, _tokens, _cost = adapter.judge("the task", "the output", "the rubric")
    return value


def _judge_case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id, prompt="the task", rubric="meets all five criteria", grader=GraderKind.LLM_JUDGE
    )


# ------------------------------------------------------------------
# The declared format
# ------------------------------------------------------------------


def test_judge_prompt_declares_one_exact_score_line() -> None:
    # The system prompt names the line, says it is the whole reply, and says
    # what happens otherwise; the user message ends by asking for that line
    # and nothing else. A prompt that says "a number" invites prose around it.
    assert "SCORE: <number>" in _JUDGE_SYSTEM_PROMPT
    assert "only line" in _JUDGE_SYSTEM_PROMPT
    assert "refused" in _JUDGE_SYSTEM_PROMPT
    message = _build_judge_message("the task", "the output", "the rubric")
    assert message.rstrip().splitlines()[-1].startswith("Reply with exactly one line: SCORE:")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SCORE: 0.8", 0.8),
        ("SCORE: 0", 0.0),
        ("SCORE: 1", 1.0),
        ("SCORE: 1.000", 1.0),
        ("SCORE: 0.25", 0.25),
        ("SCORE:0.5", 0.5),
        ("  SCORE: 0.3  \n", 0.3),
        ("\n\nSCORE:\t0.7\n", 0.7),
    ],
)
def test_judge_reads_the_declared_score_line(reply: str, expected: float) -> None:
    assert _parse_judge_score(reply) == pytest.approx(expected)
    assert _judge_with_reply(reply) == pytest.approx(expected)


# ------------------------------------------------------------------
# The defect itself
# ------------------------------------------------------------------


def test_a_reply_that_leads_with_a_count_is_refused_not_read_as_its_first_number() -> None:
    # The old parser took "1" from "1 of 5" and returned a perfect score.
    with pytest.raises(ValueError) as info:
        _parse_judge_score("1 of 5 criteria met, 0.2")
    assert info.value.reason == "judge_no_score_line"
    with pytest.raises(ValueError):
        _judge_with_reply("1 of 5 criteria met, 0.2")


def test_an_out_of_range_score_is_refused_not_clamped() -> None:
    # A clamp turns 7 (out of 10, say) into a perfect 1.0, and -1 into a
    # confident 0.0. Neither is what the judge said.
    for reply in ("SCORE: 7", "SCORE: 1.5", "SCORE: -0.2", "SCORE: 100"):
        with pytest.raises(ValueError) as info:
            _parse_judge_score(reply)
        assert info.value.reason == "judge_score_out_of_range", reply


def test_a_refused_judge_reply_is_counted_apart_and_never_scored() -> None:
    # End to end through run_eval: one case the judge scored on the declared
    # line, one it answered in prose. Only the first becomes a score; the
    # second is counted apart with its reason, so the run's mean, and drift
    # and the error budget that read it, rest on the one real score.
    evalset = EvalSet(id="judge-format", cases=[_judge_case("ok"), _judge_case("prose")])
    adapter = AnthropicAdapter()
    client = MagicMock()
    client.messages.create.side_effect = [
        _response("candidate answer one"),
        _response("SCORE: 0.4"),
        _response("candidate answer two"),
        _response("1 of 5 criteria met, 0.2"),
    ]
    with patch.object(adapter, "_get_client", return_value=client):
        run = run_eval(evalset, adapter, model="claude-haiku-4-5-20251001")

    assert [(s.case_id, s.value) for s in run.scores] == [("ok", pytest.approx(0.4))]
    assert run.mean_score == pytest.approx(0.4)
    assert [(u.case_id, u.reason) for u in run.unanswered] == [("prose", "judge_no_score_line")]
    assert run.cases_asked == 2


def test_an_adapter_value_outside_zero_one_is_refused() -> None:
    # The LLMAdapter protocol promises a score in [0, 1]. A custom adapter
    # that hands back NaN, infinity or 1.5 has broken that promise, and the
    # grader refuses rather than clamping: NaN used to pass straight through
    # min(max(...)) as a score.
    case = _judge_case("c1")
    for bad, reason in (
        (float("nan"), "judge_score_not_a_number"),
        (float("inf"), "judge_score_not_a_number"),
        (1.5, "judge_score_out_of_range"),
        (-0.5, "judge_score_out_of_range"),
    ):
        grader = LLMJudgeGrader(StubLLMAdapter(judge_value=bad))
        with pytest.raises(Exception) as info:
            grader.grade(case, "output")
        assert info.value.reason == reason, bad
        assert info.value.case_id == "c1"


# ------------------------------------------------------------------
# Hostile replies
# ------------------------------------------------------------------

_HUGE = "x" * 200_000

_HOSTILE = [
    ("", "judge_reply_empty"),
    ("   \n\t\n", "judge_reply_empty"),
    ("0.8", "judge_no_score_line"),
    ("I refuse to answer", "judge_no_score_line"),
    ("1 of 5 criteria met, 0.2", "judge_no_score_line"),
    ("The output scored 0.9 out of 1", "judge_no_score_line"),
    ("SCORE: 0.2\nSCORE: 0.9", "judge_several_score_lines"),
    ("SCORE: 0.9\nscore: 0.1", "judge_several_score_lines"),
    ('The output says "SCORE: 0.9"\nSCORE: 0.1', "judge_text_beside_score_line"),
    ("Reasoning: met 4 of 5.\nSCORE: 0.8", "judge_text_beside_score_line"),
    ("SCORE: 0.8\nbecause it met 4 of 5", "judge_text_beside_score_line"),
    ("SCORE: 0.8 (4 of 5 met)", "judge_score_line_malformed"),
    ("SCORE: NaN", "judge_score_line_malformed"),
    ("SCORE: nan", "judge_score_line_malformed"),
    ("SCORE: inf", "judge_score_line_malformed"),
    ("SCORE: 0.5e-1", "judge_score_line_malformed"),
    ("SCORE: +0.5", "judge_score_line_malformed"),
    ("SCORE: .5", "judge_score_line_malformed"),
    ("SCORE: 0,5", "judge_score_line_malformed"),
    ("SCORE: 1/2", "judge_score_line_malformed"),
    ("SCORE: 50%", "judge_score_line_malformed"),
    ("SCORE:", "judge_score_line_malformed"),
    ("score: 0.5", "judge_score_line_malformed"),
    ("**SCORE: 0.5**", "judge_no_score_line"),
    ("SCORE: \uff10.\uff18", "judge_score_line_malformed"),  # fullwidth digits
    ("SCORE: \u0660.\u0668", "judge_score_line_malformed"),  # Arabic-Indic digits
    ("SCORE: \u0968", "judge_score_line_malformed"),  # Devanagari digit two
    ("SCORE: 0.8\u2028SCORE: 0.1", "judge_several_score_lines"),
    ("SCORE: 7", "judge_score_out_of_range"),
    ("SCORE: -0", "judge_score_out_of_range"),
    (_HUGE, "judge_no_score_line"),
    (_HUGE + "\nSCORE: 0.5", "judge_text_beside_score_line"),
    ("SCORE: 0." + "5" * 100_000, "judge_score_line_malformed"),
]


@pytest.mark.parametrize(("reply", "reason"), _HOSTILE, ids=[r[1] for r in _HOSTILE])
def test_hostile_judge_replies_are_refused_with_a_reason(reply: str, reason: str) -> None:
    with pytest.raises(ValueError) as info:
        _parse_judge_score(reply)
    assert info.value.reason == reason
    # The message names the reason and never carries a huge reply whole.
    assert reason in str(info.value)
    assert len(str(info.value)) < 400


_LEAD_WORDS = ["met", "criteria", "The", "4", "0.9", "\uff11", "nan", "-", "**", '"']
_NOISE_WORDS = [*_LEAD_WORDS, "of", "score", "SCORE", "SCORE:", "1", '"SCORE: 1"', "out"]
_PAD_BEFORE = ["", " ", "\n", "\n\n "]
_PAD_AFTER = ["", " ", "\n"]


def _noise_line(rng: random.Random) -> str:
    # Never starts with SCORE, so a noise line is never a score line itself;
    # SCORE can still appear later in it, quoted or bare.
    rest = [rng.choice(_NOISE_WORDS) for _ in range(rng.randint(1, 5))]
    return " ".join([rng.choice(_LEAD_WORDS), *rest])


def test_a_seed_sweep_never_reads_a_score_the_declared_line_did_not_state() -> None:
    # Four properties, 400 seeds each. A declared line, however it is padded
    # with whitespace, reads back as exactly its value. The same line with
    # any other text before or after it is refused. The same line with text
    # after the number on the line itself is refused. Prose that never starts
    # a line with SCORE is refused as having no score line, whatever numbers
    # or quoted score lines it carries.
    for seed in range(400):
        rng = random.Random(seed)
        value = round(rng.random(), rng.randint(0, 4))
        line = f"SCORE:{' ' * rng.randint(0, 2)}{value}"
        padded = f"{rng.choice(_PAD_BEFORE)}{line}{rng.choice(_PAD_AFTER)}"
        assert _parse_judge_score(padded) == pytest.approx(value), (seed, padded)

        noise = _noise_line(rng)
        dirty = f"{noise}\n{line}" if rng.random() < 0.5 else f"{line}\n{noise}"
        with pytest.raises(ValueError) as info:
            _parse_judge_score(dirty)
        assert info.value.reason == "judge_text_beside_score_line", (seed, dirty)

        with pytest.raises(ValueError) as info:
            _parse_judge_score(f"{line} {noise}")
        assert info.value.reason == "judge_score_line_malformed", (seed, line, noise)

        prose = "\n".join(_noise_line(rng) for _ in range(rng.randint(1, 4)))
        with pytest.raises(ValueError) as info:
            _parse_judge_score(prose)
        assert info.value.reason == "judge_no_score_line", (seed, prose)


def test_a_refusal_is_never_a_number_a_mean_could_absorb() -> None:
    # The parser never returns NaN or a value outside [0, 1] for any reply in
    # the hostile list: it raises instead, so nothing downstream can average
    # a refusal in.
    for reply, _reason in _HOSTILE:
        try:
            value = _parse_judge_score(reply)
        except ValueError:
            continue
        assert math.isfinite(value) and 0.0 <= value <= 1.0
        raise AssertionError(f"{reply[:40]!r} read as {value!r}")


def test_eval_prints_and_stores_a_refused_judge_case_with_its_reason(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Through `verdryx eval`: the refused case is stored in the run's
    # unanswered list, survives a reload from the store, and the printout
    # names its reason instead of an answer id it does not have.
    from verdryx import cli
    from verdryx.store import Store

    evalset = EvalSet(id="judge-format", cases=[_judge_case("refused")])
    evalset_path = tmp_path / "judge.json"
    evalset.save(evalset_path)
    db = tmp_path / "store.db"
    monkeypatch.setattr(
        cli, "_build_adapter", lambda _model, _config: StubLLMAdapter(judge_value=float("nan"))
    )

    cli.main(["eval", str(evalset_path), "--model", "stub", "--db", str(db)])
    out = capsys.readouterr().out
    assert "mean score: unmeasured (0 of 1 answered)" in out
    assert "unanswered: 1 of 1 asked (judge_score_not_a_number: 1)" in out
    assert "refused  (judge_score_not_a_number)" in out
    assert "answer_id=" not in out

    with Store.open(db) as store:
        (run,) = store.list_runs()
    assert run.scores == []
    assert [(u.case_id, u.answer_id, u.reason) for u in run.unanswered] == [
        ("refused", "", "judge_score_not_a_number")
    ]
