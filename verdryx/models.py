"""Core data types for Verdryx.

Verdryx is the quality-evaluation and drift plane of the TAIPANBOX
agent-governance stack: it measures whether an operator's own agents did
their job correctly. Every dataclass here describes a piece of that
measurement -- a test case, a grading result, a run, a baseline to compare
against, and the drift report that comes out of the comparison. Nothing in
this module performs I/O; loading/saving lives in :mod:`verdryx.store` and
grading behavior lives in :mod:`verdryx.graders`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

# ------------------------------------------------------------------
# Outcome-tag vocabulary shared by graders.py (OutcomeTagGrader) and
# costper.py (named accessors on CostPerOutcomeReport). Defined once here so
# both modules agree on the same three strings without importing each other.
# ------------------------------------------------------------------

OUTCOME_RESOLVED = "case_resolved"
OUTCOME_ESCALATED = "escalated"
OUTCOME_ABANDONED = "abandoned"

#: Default OutcomeTagGrader mapping. Callers may pass their own mapping to
#: OutcomeTagGrader; this is only the out-of-the-box default.
DEFAULT_OUTCOME_SCORES: dict[str, float] = {
    OUTCOME_RESOLVED: 1.0,
    OUTCOME_ESCALATED: 0.5,
    OUTCOME_ABANDONED: 0.0,
}


class GraderKind(StrEnum):
    """The variants of Grader an EvalCase can select (see graders.py for the
    behavior each kind maps to). A plain string Enum so it round-trips
    through JSON (eval set files) and SQLite (stored runs) without a lookup
    table.
    """

    EXACT = "exact"
    REGEX = "regex"
    OUTCOME_TAG = "outcome_tag"
    LLM_JUDGE = "llm_judge"
    TOOL_TRACE = "tool_trace"


@dataclass
class EvalCase:
    """One test case in an EvalSet.

    Args:
        id: Stable identifier, unique within its EvalSet. Deliberately not
            auto-generated (e.g. via uuid4): case ids must stay the same
            across repeated runs of the same eval set so Scores can be
            compared case-by-case over time.
        prompt: The input sent to the model under evaluation. For
            GraderKind.OUTCOME_TAG cases, this field instead holds the raw
            outcome tag to grade (e.g. "case_resolved") -- there is nothing
            to prompt a model with when grading an already-recorded
            production outcome; see verdryx.cli's eval loop.
        expected: Ground truth for ExactGrader/RegexGrader. For RegexGrader
            this is a pattern, not a literal string.
        rubric: Grading instructions for LLMJudgeGrader.
        grader: Which GraderKind grades this case. Defaults to EXACT.
        tools: Provider-shape tool definitions passed to the adapter
            verbatim, for GraderKind.TOOL_TRACE cases (see
            LLMAdapter.complete_with_tools in graders.py). Required
            (non-empty) for tool_trace cases; an error on any other grader.
        expected_tools: Ordered expected tool NAMES for GraderKind.TOOL_TRACE
            cases (see ToolTraceGrader in graders.py). An empty list is
            legal and means the model is expected to call no tools at all.
            Required for tool_trace cases; an error on any other grader.
    """

    id: str
    prompt: str
    expected: str | None = None
    rubric: str | None = None
    grader: GraderKind = GraderKind.EXACT
    tools: list[dict[str, object]] | None = None
    expected_tools: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        for field_name in ("id", "prompt"):
            if field_name not in data:
                raise ValueError(f"missing the required field {field_name!r}")
        raw_grader = data.get("grader", GraderKind.EXACT.value)
        try:
            grader = GraderKind(raw_grader)
        except ValueError:
            allowed = ", ".join(g.value for g in GraderKind)
            raise ValueError(f"field 'grader': {raw_grader!r} is not one of {allowed}") from None

        tools = data.get("tools")
        expected_tools = data.get("expected_tools")
        if grader == GraderKind.TOOL_TRACE:
            if "tools" not in data:
                raise ValueError("grader 'tool_trace' requires the field 'tools'")
            if not isinstance(tools, list):
                raise ValueError("field 'tools' must be a list")
            if not tools:
                raise ValueError("field 'tools' must not be empty for grader 'tool_trace'")
            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"field 'tools'[{i}] must be an object")
            if "expected_tools" not in data:
                raise ValueError("grader 'tool_trace' requires the field 'expected_tools'")
            if not isinstance(expected_tools, list):
                raise ValueError("field 'expected_tools' must be a list")
            for i, name in enumerate(expected_tools):
                if not isinstance(name, str) or not name:
                    raise ValueError(f"field 'expected_tools'[{i}] must be a non-empty string")
        else:
            if tools is not None:
                raise ValueError("field 'tools' is only valid with grader tool_trace")
            if expected_tools is not None:
                raise ValueError("field 'expected_tools' is only valid with grader tool_trace")

        return cls(
            id=data["id"],
            prompt=data["prompt"],
            expected=data.get("expected"),
            rubric=data.get("rubric"),
            grader=grader,
            tools=tools,
            expected_tools=expected_tools,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "prompt": self.prompt, "grader": self.grader.value}
        if self.expected is not None:
            d["expected"] = self.expected
        if self.rubric is not None:
            d["rubric"] = self.rubric
        if self.tools is not None:
            d["tools"] = self.tools
        if self.expected_tools is not None:
            d["expected_tools"] = self.expected_tools
        return d


@dataclass
class EvalSet:
    """A named collection of EvalCase records loaded from / saved to JSON."""

    id: str
    cases: list[EvalCase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalSet:
        if "id" not in data:
            raise ValueError("the eval set is missing the required field 'id'")
        cases_raw = data.get("cases", [])
        if not isinstance(cases_raw, list):
            raise ValueError("field 'cases' must be a list")
        cases = []
        for i, c in enumerate(cases_raw):
            if not isinstance(c, dict):
                raise ValueError(f"case #{i + 1} must be an object")
            try:
                cases.append(EvalCase.from_dict(c))
            except ValueError as e:
                hint = f" ({c.get('id')!r})" if c.get("id") else ""
                raise ValueError(f"case #{i + 1}{hint}: {e}") from e
        return cls(id=data["id"], cases=cases)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "cases": [c.to_dict() for c in self.cases]}

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        """Load an eval set from a JSON file. See README.md for the shape.

        Raises ValueError with the file named and the offending case and field
        pointed at, rather than letting a raw KeyError/JSONDecodeError
        traceback reach an operator who ran this by hand.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"{path}: cannot read: {e}") from e
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a JSON object")
        try:
            return cls.from_dict(data)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Completion:
    """The parsed result of one model response, including the ordered
    tool_use names.

    Produced by LLMAdapter.complete_with_tools() (see graders.py) for a
    GraderKind.TOOL_TRACE case's single-turn model call: `text` is every
    text content block concatenated in order (may be empty when the
    response is tool_use blocks only), `tool_names` is the ordered list of
    tool_use block names the model chose (empty if it called no tools), and
    `tokens`/`cost_usd` mirror LLMAdapter.complete()'s own accounting so
    verdryx.cli's eval loop can fold them into Score exactly the way it
    already does for complete().
    """

    text: str
    tool_names: list[str]
    tokens: int
    cost_usd: float


@dataclass
class GradeResult:
    """What a Grader returns from grade(): a Score without a case_id.

    A grader only sees the EvalCase and the output being graded, not which
    stored run it belongs to, so it cannot build a full Score itself. The
    eval runner (verdryx.cli) attaches case_id via to_score() once grading
    completes.
    """

    value: float
    tokens: int = 0
    cost_usd: float = 0.0

    def to_score(self, case_id: str) -> Score:
        return Score(case_id=case_id, value=self.value, tokens=self.tokens, cost_usd=self.cost_usd)


@dataclass
class Score:
    """One case's grade within an EvalRun.

    cost_usd is threaded through end to end (here, the SQLite store, and
    EvalRun's rollup properties). For GraderKind.LLM_JUDGE cases it is
    populated from verdryx.pricing.PriceBook against the judge call's token
    usage (see graders.py's LLMJudgeGrader and AnthropicAdapter). The other
    three graders -- ExactGrader, RegexGrader, OutcomeTagGrader -- make no
    model call, so their Scores keep cost_usd at 0.0: there is genuinely
    nothing to price. Callers who already know the price from elsewhere
    (e.g. a tokenfuse export) can still populate cost_usd directly when
    constructing Score records themselves.
    """

    case_id: str
    value: float
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class EvalRun:
    """One execution of an EvalSet against one model."""

    id: str
    model: str
    started_at: datetime
    finished_at: datetime | None = None
    scores: list[Score] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """Mean of all case scores, or 0.0 if the run has no scores yet."""
        if not self.scores:
            return 0.0
        return sum(s.value for s in self.scores) / len(self.scores)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.scores)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.scores)


@dataclass
class Baseline:
    """A frozen reference mean_score that DriftReport compares against.

    Baseline snapshots EvalRun.mean_score at the moment it is set rather
    than pointing live at the run, so a baseline's value cannot silently
    shift if the underlying run were ever re-saved.
    """

    id: str
    eval_run_id: str
    mean_score: float
    created_at: datetime
    label: str = ""


@dataclass
class DriftReport:
    """Result of comparing recent eval runs against a stored Baseline.

    Args:
        baseline_id: Which Baseline this report was computed against.
        window: Number of most-recent eval runs pooled into mean_score.
        mean_score: Mean of all case scores across the windowed runs.
        delta: mean_score - baseline.mean_score. Negative means the
            windowed runs scored lower than the baseline.
        verdict: "regressed" if delta dropped at or past the configured
            threshold, OR if the significance check below flags it,
            "on-track" otherwise. See verdryx.drift.
        baseline_n: Number of individual case scores the significance check
            was run against, reloaded from the baseline's original eval run.
            0 when that run's scores were not supplied to compute_drift (the
            comparison then falls back to the flat threshold alone).
        t_statistic: Welch's t-statistic for recent scores vs. the
            baseline run's scores, or None when baseline_n is 0 or either
            sample has fewer than 2 values (variance undefined).
            Informational: the verdict itself is driven by ci_low/ci_high,
            not this value.
        ci_low: Lower bound of the bootstrap confidence interval on delta
            (recent mean - baseline mean), or None when baseline_n is 0.
        ci_high: Upper bound of that interval, or None. When ci_high < 0,
            the drop is significant at the configured confidence level even
            if it is smaller than `threshold`, and the verdict reflects
            that.
    """

    baseline_id: str
    window: int
    mean_score: float
    delta: float
    verdict: Literal["on-track", "regressed"]
    baseline_n: int = 0
    t_statistic: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class OutcomeCost:
    """Cost statistics for one outcome tag (or the "overall" pseudo-tag)."""

    outcome: str
    count: int
    total_cost_usd: float
    mean_cost_usd: float


@dataclass
class CostPerOutcomeReport:
    """Result of verdryx.costper.cost_per_outcome().

    by_outcome covers whatever outcome tags actually appear in the input
    records (operators may configure OutcomeTagGrader with a custom
    vocabulary beyond the three defaults), so it is a plain dict rather than
    three fixed fields. The resolved/escalated/abandoned properties are
    named convenience accessors for the three tags Verdryx ships as
    defaults; they return None if that tag never appeared in the input.
    """

    by_outcome: dict[str, OutcomeCost]
    overall: OutcomeCost

    def get(self, outcome: str) -> OutcomeCost | None:
        return self.by_outcome.get(outcome)

    @property
    def resolved(self) -> OutcomeCost | None:
        return self.by_outcome.get(OUTCOME_RESOLVED)

    @property
    def escalated(self) -> OutcomeCost | None:
        return self.by_outcome.get(OUTCOME_ESCALATED)

    @property
    def abandoned(self) -> OutcomeCost | None:
        return self.by_outcome.get(OUTCOME_ABANDONED)
