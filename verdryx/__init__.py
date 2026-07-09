"""Verdryx: the quality-evaluation and drift plane of the TAIPANBOX
agent-governance stack.

Verdryx measures whether an operator's own agents did their job correctly.
It never manipulates outputs and never attacks anything; see README.md.
"""

from verdryx.config import Config
from verdryx.costper import cost_per_outcome, load_records, read_parquet
from verdryx.drift import DEFAULT_THRESHOLD, compute_drift
from verdryx.events import EventLog, resolve_events_path
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
from verdryx.pricing import ModelPrice, PriceBook
from verdryx.store import Store

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_OUTCOME_SCORES",
    "DEFAULT_THRESHOLD",
    "OUTCOME_ABANDONED",
    "OUTCOME_ESCALATED",
    "OUTCOME_RESOLVED",
    "AnthropicAdapter",
    "Baseline",
    "Config",
    "CostPerOutcomeReport",
    "DriftReport",
    "EvalCase",
    "EvalRun",
    "EvalSet",
    "EventLog",
    "ExactGrader",
    "GradeResult",
    "Grader",
    "GraderKind",
    "LLMAdapter",
    "LLMJudgeGrader",
    "ModelPrice",
    "OutcomeCost",
    "OutcomeTagGrader",
    "PriceBook",
    "RegexGrader",
    "Score",
    "Store",
    "StubLLMAdapter",
    "__version__",
    "build_graders",
    "compute_drift",
    "cost_per_outcome",
    "load_records",
    "read_parquet",
    "resolve_events_path",
]
