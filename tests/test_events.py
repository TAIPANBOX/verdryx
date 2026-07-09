"""Tests for verdryx.events.

Schema validation uses a vendored copy of the Agent Passport
agent-event.v0.2.schema.json (tests/fixtures/, copied from
TAIPANBOX/agent-passport -- SPEC.md Sec 6). Vendored rather than fetched at
test time: CI checks out only this repo, and validating a wire contract
should never depend on a live network call. `event_schema` and `agent_id`
are fixtures from conftest.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from verdryx.events import EventLog, resolve_events_path


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------
# resolve_events_path
# ------------------------------------------------------------------


def test_resolve_events_path_none_when_nothing_set(monkeypatch) -> None:
    monkeypatch.delenv("VERDRYX_EVENTS_PATH", raising=False)
    assert resolve_events_path(None) is None


def test_resolve_events_path_explicit_wins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VERDRYX_EVENTS_PATH", str(tmp_path / "env.ndjson"))
    explicit = tmp_path / "explicit.ndjson"
    assert resolve_events_path(explicit) == explicit


def test_resolve_events_path_env_fallback(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / "env.ndjson"
    monkeypatch.setenv("VERDRYX_EVENTS_PATH", str(env_path))
    assert resolve_events_path(None) == env_path


# ------------------------------------------------------------------
# Skip on empty agent_id (Engram rule: never fabricate one)
# ------------------------------------------------------------------


def test_emit_skips_and_counts_when_agent_id_none_or_empty(tmp_path) -> None:
    log = EventLog(tmp_path / "events.ndjson")
    log.emit("eval_run", None, {"model": "stub"})
    log.emit("eval_run", "", {"model": "stub"})
    assert log.skipped_empty_agent_id == 2
    assert not log.path.exists()


# ------------------------------------------------------------------
# Fail-open
# ------------------------------------------------------------------


def test_emit_fails_open_on_unwritable_path(tmp_path, caplog, agent_id) -> None:
    bad_path = tmp_path / "nonexistent-dir" / "events.ndjson"
    log = EventLog(bad_path)
    with caplog.at_level(logging.WARNING, logger="verdryx.events"):
        log.emit("eval_run", agent_id, {"model": "stub"})
    assert any("verdryx.events" in r.name for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not bad_path.exists()


# ------------------------------------------------------------------
# Golden-line schema validation for each event type
# ------------------------------------------------------------------


def test_eval_run_event_is_schema_valid(tmp_path, event_schema, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit(
        "eval_run",
        agent_id,
        {"model": "stub", "cases": 5, "mean_score": 0.9, "total_tokens": 0, "total_cost_usd": 0.0},
        run_id="run-1",
    )
    events = _read_ndjson(events_path)
    assert len(events) == 1
    event = events[0]
    jsonschema.validate(instance=event, schema=event_schema)
    assert event["type"] == "eval_run"
    assert event["severity"] == "info"
    assert event["source"] == "verdryx"
    assert event["schema"] == "taipanbox.dev/agent-event/v0.2"
    assert event["agent_id"] == agent_id
    assert event["run_id"] == "run-1"
    assert "prev_hash" not in event


def test_quality_score_event_is_schema_valid(tmp_path, event_schema, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit(
        "quality_score",
        agent_id,
        {"case_id": "c1", "value": 1.0, "tokens": 0, "cost_usd": 0.0},
        run_id="run-1",
    )
    event = _read_ndjson(events_path)[0]
    jsonschema.validate(instance=event, schema=event_schema)
    assert event["type"] == "quality_score"
    assert event["severity"] == "info"


def test_quality_drift_event_is_schema_valid_and_high_severity(
    tmp_path, event_schema, agent_id
) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit(
        "quality_drift",
        agent_id,
        {
            "baseline_id": "b1",
            "window": 3,
            "mean_score": 0.7,
            "delta": -0.2,
            "verdict": "regressed",
        },
        run_id="run-9",
    )
    event = _read_ndjson(events_path)[0]
    jsonschema.validate(instance=event, schema=event_schema)
    assert event["type"] == "quality_drift"
    assert event["severity"] == "high"


def test_unknown_event_type_defaults_to_info_severity(tmp_path, event_schema, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("some_future_type", agent_id, {})
    event = _read_ndjson(events_path)[0]
    jsonschema.validate(instance=event, schema=event_schema)
    assert event["severity"] == "info"


def test_emit_appends_multiple_lines_each_schema_valid(tmp_path, event_schema, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"a": 1})
    log.emit("eval_run", agent_id, {"a": 2})
    events = _read_ndjson(events_path)
    assert len(events) == 2
    for event in events:
        jsonschema.validate(instance=event, schema=event_schema)


def test_emit_without_run_id_omits_the_field(tmp_path, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"a": 1})
    event = _read_ndjson(events_path)[0]
    assert "run_id" not in event


def test_bad_agent_id_pattern_is_rejected_by_schema(tmp_path, event_schema) -> None:
    """Sanity check that the vendored schema is actually doing work: an
    agent_id that violates the agent:// pattern must fail validation."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", "not-a-valid-agent-id", {"a": 1})
    event = _read_ndjson(events_path)[0]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=event, schema=event_schema)
