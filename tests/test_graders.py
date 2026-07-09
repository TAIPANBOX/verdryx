"""Tests for verdryx.graders.

AnthropicAdapter tests use the same technique as Engram's own
tests/test_llm_adapters.py: patch sys.modules["anthropic"] to a MagicMock so
the adapter's local `import anthropic` resolves to it, then assert on the
kwargs the mock constructor received. No real network call is made anywhere
in this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from verdryx.graders import (
    AnthropicAdapter,
    ExactGrader,
    Grader,
    LLMAdapter,
    LLMJudgeGrader,
    OutcomeTagGrader,
    RegexGrader,
    StubLLMAdapter,
    build_graders,
)
from verdryx.models import DEFAULT_OUTCOME_SCORES, EvalCase, GraderKind
from verdryx.pricing import ModelPrice, PriceBook

# ------------------------------------------------------------------
# ExactGrader
# ------------------------------------------------------------------


def test_exact_grader_match() -> None:
    case = EvalCase(id="c1", prompt="p", expected="hello")
    assert ExactGrader().grade(case, "hello").value == 1.0


def test_exact_grader_mismatch() -> None:
    case = EvalCase(id="c1", prompt="p", expected="hello")
    assert ExactGrader().grade(case, "goodbye").value == 0.0


def test_exact_grader_requires_expected() -> None:
    case = EvalCase(id="c1", prompt="p")
    with pytest.raises(ValueError, match=r"requires case\.expected"):
        ExactGrader().grade(case, "anything")


def test_exact_grader_satisfies_grader_protocol() -> None:
    assert isinstance(ExactGrader(), Grader)


# ------------------------------------------------------------------
# RegexGrader
# ------------------------------------------------------------------


def test_regex_grader_match() -> None:
    case = EvalCase(id="c1", prompt="p", expected=r"\bsorry\b")
    assert RegexGrader().grade(case, "I am sorry about that").value == 1.0


def test_regex_grader_no_match() -> None:
    case = EvalCase(id="c1", prompt="p", expected=r"\bsorry\b")
    assert RegexGrader().grade(case, "no apology here").value == 0.0


def test_regex_grader_partial_match_anywhere_in_output_counts() -> None:
    """re.search, not re.fullmatch: a pattern matching a substring passes."""
    case = EvalCase(id="c1", prompt="p", expected="refund")
    assert RegexGrader().grade(case, "Your refund has been processed.").value == 1.0


def test_regex_grader_requires_expected() -> None:
    case = EvalCase(id="c1", prompt="p")
    with pytest.raises(ValueError, match=r"requires case\.expected"):
        RegexGrader().grade(case, "anything")


def test_regex_grader_invalid_pattern_raises_value_error() -> None:
    case = EvalCase(id="c1", prompt="p", expected="(unclosed")
    with pytest.raises(ValueError, match="not a valid regex"):
        RegexGrader().grade(case, "anything")


# ------------------------------------------------------------------
# OutcomeTagGrader
# ------------------------------------------------------------------


def test_outcome_tag_grader_default_mapping() -> None:
    grader = OutcomeTagGrader()
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    assert grader.grade(case, "case_resolved").value == 1.0
    assert grader.grade(case, "escalated").value == 0.5
    assert grader.grade(case, "abandoned").value == 0.0


def test_outcome_tag_grader_strips_whitespace() -> None:
    grader = OutcomeTagGrader()
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    assert grader.grade(case, "  case_resolved  ").value == 1.0


def test_outcome_tag_grader_unknown_tag_uses_default() -> None:
    grader = OutcomeTagGrader(default=0.25)
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    result = grader.grade(case, "never_seen_before")
    assert result.value == 0.25


def test_outcome_tag_grader_unknown_tag_defaults_to_zero() -> None:
    grader = OutcomeTagGrader()
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    assert grader.grade(case, "never_seen_before").value == 0.0


def test_outcome_tag_grader_custom_mapping_is_configurable() -> None:
    grader = OutcomeTagGrader(mapping={"solved": 1.0, "punted": 0.0})
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    assert grader.grade(case, "solved").value == 1.0
    # A default-mapping tag is not automatically recognized once the mapping
    # has been fully overridden.
    assert grader.grade(case, "escalated").value == 0.0


def test_outcome_tag_grader_custom_mapping_does_not_mutate_shared_default() -> None:
    grader = OutcomeTagGrader(mapping={"solved": 1.0})
    grader.mapping["solved"] = 0.0
    assert DEFAULT_OUTCOME_SCORES["case_resolved"] == 1.0


def test_outcome_tag_grader_default_mapping_is_a_copy_not_shared_reference() -> None:
    grader = OutcomeTagGrader()
    grader.mapping["case_resolved"] = 0.0
    assert DEFAULT_OUTCOME_SCORES["case_resolved"] == 1.0


# ------------------------------------------------------------------
# LLMJudgeGrader + StubLLMAdapter
# ------------------------------------------------------------------


def test_stub_llm_adapter_is_deterministic_and_records_calls() -> None:
    adapter = StubLLMAdapter(completion="fixed output", judge_value=0.75, tokens=12)
    text, tokens = adapter.complete("some prompt")
    assert text == "fixed output"
    assert tokens == 12
    value, jtokens, jcost = adapter.judge("prompt", "output", "rubric")
    assert value == 0.75
    assert jtokens == 12
    assert jcost == 0.0
    assert adapter.completions == ["some prompt"]
    assert adapter.judgements == [("prompt", "output", "rubric")]


def test_stub_llm_adapter_judge_reports_injected_cost_usd() -> None:
    adapter = StubLLMAdapter(judge_value=0.9, tokens=100, cost_usd=0.0035)
    value, tokens, cost_usd = adapter.judge("prompt", "output", "rubric")
    assert value == 0.9
    assert tokens == 100
    assert cost_usd == 0.0035


def test_stub_llm_adapter_satisfies_llm_adapter_protocol() -> None:
    assert isinstance(StubLLMAdapter(), LLMAdapter)


def test_llm_judge_grader_uses_stub_adapter_no_network() -> None:
    adapter = StubLLMAdapter(judge_value=0.6, tokens=7)
    grader = LLMJudgeGrader(adapter)
    case = EvalCase(id="c1", prompt="p", rubric="be nice", grader=GraderKind.LLM_JUDGE)
    result = grader.grade(case, "some output")
    assert result.value == 0.6
    assert result.tokens == 7
    assert result.cost_usd == 0.0
    assert adapter.judgements == [("p", "some output", "be nice")]


def test_llm_judge_grader_threads_cost_usd_from_adapter() -> None:
    """GradeResult.cost_usd comes straight from the adapter's judge() --
    LLMJudgeGrader does no pricing of its own (see AnthropicAdapter for the
    adapter that actually prices itself, via verdryx.pricing.PriceBook)."""
    adapter = StubLLMAdapter(judge_value=0.9, tokens=100, cost_usd=0.0042)
    grader = LLMJudgeGrader(adapter)
    case = EvalCase(id="c1", prompt="p", rubric="be nice", grader=GraderKind.LLM_JUDGE)
    result = grader.grade(case, "some output")
    assert result.cost_usd == 0.0042


def test_llm_judge_grader_requires_rubric() -> None:
    adapter = StubLLMAdapter()
    grader = LLMJudgeGrader(adapter)
    case = EvalCase(id="c1", prompt="p")
    with pytest.raises(ValueError, match=r"requires case\.rubric"):
        grader.grade(case, "output")


def test_llm_judge_grader_clamps_high_adapter_value() -> None:
    adapter = StubLLMAdapter(judge_value=1.5)
    grader = LLMJudgeGrader(adapter)
    case = EvalCase(id="c1", prompt="p", rubric="x", grader=GraderKind.LLM_JUDGE)
    assert grader.grade(case, "output").value == 1.0


def test_llm_judge_grader_clamps_low_adapter_value() -> None:
    adapter = StubLLMAdapter(judge_value=-0.5)
    grader = LLMJudgeGrader(adapter)
    case = EvalCase(id="c1", prompt="p", rubric="x", grader=GraderKind.LLM_JUDGE)
    assert grader.grade(case, "output").value == 0.0


def test_llm_judge_grader_satisfies_grader_protocol() -> None:
    assert isinstance(LLMJudgeGrader(StubLLMAdapter()), Grader)


# ------------------------------------------------------------------
# AnthropicAdapter -- construction seam (base_url / api_key forwarding)
# ------------------------------------------------------------------


def test_anthropic_adapter_default_model() -> None:
    assert AnthropicAdapter().model_name == "claude-haiku-4-5-20251001"


def test_anthropic_adapter_custom_model() -> None:
    assert AnthropicAdapter(model="claude-opus-4-1").model_name == "claude-opus-4-1"


def test_anthropic_adapter_satisfies_llm_adapter_protocol() -> None:
    assert isinstance(AnthropicAdapter(), LLMAdapter)


def test_anthropic_adapter_import_error_without_sdk() -> None:
    adapter = AnthropicAdapter()
    with (
        patch.dict("sys.modules", {"anthropic": None}),
        pytest.raises(ImportError, match=r"verdryx\[anthropic\]"),
    ):
        adapter._get_client()


def test_anthropic_adapter_base_url_and_api_key_omitted_by_default() -> None:
    mock_mod = MagicMock()
    mock_class = MagicMock(return_value=MagicMock())
    mock_mod.Anthropic = mock_class
    adapter = AnthropicAdapter()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        adapter._get_client()
    assert mock_class.call_args.kwargs == {}


def test_anthropic_adapter_base_url_passed_to_client() -> None:
    mock_mod = MagicMock()
    mock_class = MagicMock(return_value=MagicMock())
    mock_mod.Anthropic = mock_class
    adapter = AnthropicAdapter(base_url="https://tokenfuse.internal/anthropic")
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        adapter._get_client()
    assert mock_class.call_args.kwargs.get("base_url") == "https://tokenfuse.internal/anthropic"


def test_anthropic_adapter_api_key_passed_to_client() -> None:
    mock_mod = MagicMock()
    mock_class = MagicMock(return_value=MagicMock())
    mock_mod.Anthropic = mock_class
    adapter = AnthropicAdapter(api_key="sk-ant-test")
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        adapter._get_client()
    assert mock_class.call_args.kwargs.get("api_key") == "sk-ant-test"


def test_anthropic_adapter_base_url_and_api_key_both_passed_to_client() -> None:
    mock_mod = MagicMock()
    mock_class = MagicMock(return_value=MagicMock())
    mock_mod.Anthropic = mock_class
    adapter = AnthropicAdapter(
        base_url="https://tokenfuse.internal/anthropic", api_key="sk-ant-test"
    )
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        adapter._get_client()
    assert mock_class.call_args.kwargs == {
        "base_url": "https://tokenfuse.internal/anthropic",
        "api_key": "sk-ant-test",
    }


def test_anthropic_adapter_client_is_cached() -> None:
    mock_mod = MagicMock()
    mock_class = MagicMock(return_value=MagicMock())
    mock_mod.Anthropic = mock_class
    adapter = AnthropicAdapter()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        first = adapter._get_client()
        second = adapter._get_client()
    assert first is second
    assert mock_class.call_count == 1


# ------------------------------------------------------------------
# AnthropicAdapter -- complete() / judge() response parsing
# ------------------------------------------------------------------


def test_anthropic_adapter_complete_extracts_text_and_tokens() -> None:
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="hello there")],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        text, tokens = adapter.complete("hi")
    assert text == "hello there"
    assert tokens == 8


def test_anthropic_adapter_judge_parses_score_from_response() -> None:
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="0.8")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        value, tokens, _cost_usd = adapter.judge("prompt", "output", "rubric")
    assert value == pytest.approx(0.8)
    assert tokens == 11


def test_anthropic_adapter_judge_clamps_out_of_range_score() -> None:
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="7")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        value, _tokens, _cost_usd = adapter.judge("prompt", "output", "rubric")
    assert value == 1.0


def test_anthropic_adapter_judge_survives_empty_response() -> None:
    """A structurally surprising (empty content) response degrades to 0.0,
    not an exception, mirroring engram.llm._anthropic_text's hardening."""
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=0))
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        value, _tokens, _cost_usd = adapter.judge("prompt", "output", "rubric")
    assert value == 0.0


def test_anthropic_adapter_judge_survives_non_numeric_response() -> None:
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="I refuse to answer")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        value, _tokens, _cost_usd = adapter.judge("prompt", "output", "rubric")
    assert value == 0.0


# ------------------------------------------------------------------
# AnthropicAdapter -- judge() cost_usd via PriceBook (the LLM judge path
# that previously left Score.cost_usd hardcoded at 0.0)
# ------------------------------------------------------------------


def test_anthropic_adapter_judge_prices_known_model_via_default_price_book() -> None:
    """AnthropicAdapter()'s default model (claude-haiku-4-5-20251001) is an
    exact PriceBook.default() entry: $1.00 / $5.00 per Mtok input/output."""
    adapter = AnthropicAdapter()
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="0.8")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        _value, _tokens, cost_usd = adapter.judge("prompt", "output", "rubric")
    # 10 * 1.00/1e6 + 1 * 5.00/1e6 = 0.00001 + 0.000005 = 0.000015
    assert cost_usd == pytest.approx(0.000015)


def test_anthropic_adapter_judge_prices_unknown_model_via_conservative_fallback() -> None:
    adapter = AnthropicAdapter(model="some-future-model-nobody-has-priced-yet")
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="0.5")],
        usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        _value, _tokens, cost_usd = adapter.judge("prompt", "output", "rubric")
    # Fallback: $15.00 / $75.00 per Mtok -> 15 + 75 = 90 for 1M/1M tokens.
    assert cost_usd == pytest.approx(90.0)


def test_anthropic_adapter_judge_uses_injected_price_book() -> None:
    custom_book = PriceBook().with_price("weird-model", ModelPrice(1.0, 1.0))
    adapter = AnthropicAdapter(model="weird-model", price_book=custom_book)
    mock_client = MagicMock()
    response = SimpleNamespace(
        content=[SimpleNamespace(text="1")],
        usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0),
    )
    mock_client.messages.create.return_value = response
    with patch.object(adapter, "_get_client", return_value=mock_client):
        _value, _tokens, cost_usd = adapter.judge("prompt", "output", "rubric")
    assert cost_usd == pytest.approx(1.0)


# ------------------------------------------------------------------
# build_graders
# ------------------------------------------------------------------


def test_build_graders_covers_deterministic_kinds_without_adapter() -> None:
    graders = build_graders()
    assert GraderKind.EXACT in graders
    assert GraderKind.REGEX in graders
    assert GraderKind.OUTCOME_TAG in graders
    assert GraderKind.LLM_JUDGE not in graders


def test_build_graders_adds_llm_judge_when_adapter_given() -> None:
    graders = build_graders(judge_adapter=StubLLMAdapter())
    assert GraderKind.LLM_JUDGE in graders
    assert isinstance(graders[GraderKind.LLM_JUDGE], LLMJudgeGrader)


def test_build_graders_outcome_map_override_applies_to_the_built_grader() -> None:
    graders = build_graders(outcome_map={"solved": 1.0})
    grader = graders[GraderKind.OUTCOME_TAG]
    case = EvalCase(id="c1", prompt="x", grader=GraderKind.OUTCOME_TAG)
    assert grader.grade(case, "solved").value == 1.0
    assert grader.grade(case, "case_resolved").value == 0.0
