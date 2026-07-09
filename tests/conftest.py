"""Shared fixtures for the Verdryx test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verdryx.models import EvalCase, EvalSet, GraderKind

#: A syntactically valid Agent Passport id, matching the pattern enforced by
#: agent-event.v0.2.schema.json (^agent://[a-z0-9.-]+/[a-z0-9._/-]+$).
AGENT_ID = "agent://acme-bank.example/support/tier1-bot"

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "agent-event.v0.2.schema.json"


@pytest.fixture()
def agent_id() -> str:
    return AGENT_ID


@pytest.fixture()
def event_schema() -> dict[str, Any]:
    """The vendored Agent Passport agent-event v0.2 JSON Schema."""
    return json.loads(_SCHEMA_PATH.read_text())


@pytest.fixture()
def sample_evalset() -> EvalSet:
    """A small eval set covering all four grader kinds.

    StubLLMAdapter's defaults (completion="stub output", judge_value=1.0)
    score every case here as a perfect match, so a run over this set with a
    freshly-constructed StubLLMAdapter has mean_score == 1.0 unless a test
    overrides the adapter's defaults.
    """
    return EvalSet(
        id="fixture-v1",
        cases=[
            EvalCase(
                id="exact-1", prompt="say hi", expected="stub output", grader=GraderKind.EXACT
            ),
            EvalCase(id="regex-1", prompt="say hi", expected="stub", grader=GraderKind.REGEX),
            EvalCase(id="outcome-1", prompt="case_resolved", grader=GraderKind.OUTCOME_TAG),
            EvalCase(
                id="judge-1", prompt="say hi", rubric="greets politely", grader=GraderKind.LLM_JUDGE
            ),
        ],
    )


@pytest.fixture()
def sample_evalset_path(tmp_path: Path, sample_evalset: EvalSet) -> Path:
    """`sample_evalset`, written to a temp JSON file."""
    path = tmp_path / "evalset.json"
    sample_evalset.save(path)
    return path


@pytest.fixture()
def pyarrow_and_parquet():
    """(pyarrow, pyarrow.parquet), or skip the test if pyarrow (the
    `traces` extra) isn't installed. Shared by tests/test_costper.py and
    tests/test_cli.py -- both write small Parquet trace fixtures."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    return pa, pq
