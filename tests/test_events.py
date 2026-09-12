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

from verdryx.events import (
    AGENT_ID_MAX_LENGTH,
    AGENT_ID_PATTERN,
    SCHEMA,
    EventLog,
    canonicalize,
    chain_hash,
    is_canonical_agent_id,
    resolve_events_path,
)


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


def test_delegation_chain_past_the_spec_depth_is_rejected(agent_id, event_schema) -> None:
    """SPEC Sec 5.1 caps the delegation chain at 32 entries and the canonical
    schema carries that as maxItems. The vendored copy had lost it, so this
    suite validated a chain of any depth while reporting that it had checked
    one against the wire contract.

    Both directions are asserted deliberately. A bound that refuses 33 and
    also refuses 32 is a different defect behind the same green tick.
    """

    def event(depth: int) -> dict[str, Any]:
        return {
            "schema": "taipanbox.dev/agent-event/v0.2",
            "ts": "2026-07-09T03:12:44.100Z",
            "source": "verdryx",
            "type": "eval_run",
            "agent_id": agent_id,
            "on_behalf_of": [f"agent://acme.example/a/{i}" for i in range(depth)],
        }

    jsonschema.validate(instance=event(32), schema=event_schema)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=event(33), schema=event_schema)


def test_a_delegation_proof_is_checked_and_not_merely_carried(agent_id, event_schema) -> None:
    """SPEC Sec 5.2's `delegation_proof` is optional, and where it is present
    the vendored copy has to hold its shape rather than wave it through.

    The envelope is `additionalProperties: true`, so before this object existed
    any value at all under this key validated. That is the same defect the
    maxItems test above was written for: a vendored copy that has quietly lost a
    constraint reports that it checked a line against the wire contract, and did
    not.

    Four members and no others. The field names a proof an auditor can find
    (`jti`, `iss`), says who was holding it (`jkt`), and says when it stopped
    being one (`exp`). A proof missing `jkt` is a delegation nobody can
    attribute and must not pass as one, and a proof carrying the token itself is
    the thing SPEC Sec 5.2 exists to prevent: a live credential written into a
    replicated, hash-chained record. `additionalProperties: false` is what
    refuses it, so the refusal is asserted here rather than assumed.
    """

    def event(**extra: Any) -> dict[str, Any]:
        return {
            "schema": "taipanbox.dev/agent-event/v0.2",
            "ts": "2026-08-26T14:02:09.000Z",
            "source": "verdryx",
            "type": "eval_run",
            "agent_id": agent_id,
            "on_behalf_of": ["user://acme.example/alice", agent_id],
            **extra,
        }

    proof = {
        "jti": "01J9Z0K7Q0000000000000000",
        "jkt": "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs",
        "iss": "https://idryx.acme.example",
        "exp": 1787654400,
    }

    # Optional, and absent means NOT proven rather than proven elsewhere.
    jsonschema.validate(instance=event(), schema=event_schema)
    jsonschema.validate(instance=event(delegation_proof=proof), schema=event_schema)

    for missing in ("jti", "jkt", "iss", "exp"):
        partial = {k: v for k, v in proof.items() if k != missing}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=event(delegation_proof=partial), schema=event_schema)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance=event(delegation_proof={**proof, "token": "eyJhbGciOiJFUzI1NiJ9.e30.sig"}),
            schema=event_schema,
        )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance=event(delegation_proof={**proof, "exp": "2026-08-26T14:02:09Z"}),
            schema=event_schema,
        )


def test_verdryx_writes_no_delegation_proof_because_it_has_none(tmp_path, agent_id) -> None:
    """Absent is the correct value here, and it is worth a test rather than a
    comment.

    SPEC Sec 5.2 reads absence as NOT proven, never as proven somewhere else, so
    an emitter that writes the field is asserting it verified an RFC 8693 token.
    `EventLog.emit` builds a fixed envelope out of an event type, an agent id
    and a data dict: it holds no delegation chain and no token, so it has
    nothing to assert. This goes red the day somebody adds the key without the
    verification behind it, which is the direction that mistake travels.
    """
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"model": "stub"}, run_id="run-1")

    event = _read_ndjson(events_path)[0]
    assert "delegation_proof" not in event, "verdryx verified no token, so it proves nothing"
    assert "on_behalf_of" not in event, "and it carries no chain a proof could be about"


# ------------------------------------------------------------------
# prev_hash chain (SPEC.md Sec 6.5)
# ------------------------------------------------------------------

# Cross-language pinned vectors: agent-stack-go/event/testdata/chain-vectors.json
# is the normative cross-language truth (Go: event.Canonicalize/ChainHash; Rust:
# tokenfuse's agent-event exporter; here: canonicalize/chain_hash). Every
# implementation MUST reproduce these byte-for-byte. The vector events carry
# envelope keys (on_behalf_of, run_id) that verdryx's own emit() never sets
# itself -- canonicalize/chain_hash operate on plain dicts, so that is fine.

_VEC_EVENT_1 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:00Z",
    "source": "wardryx",
    "type": "policy_deny",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "high",
    "run_id": "run-0001",
    "data": {"policy": "finance-guard", "reason": "deny_tool: shell"},
}
_VEC_CANONICAL_1 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"policy":"finance-guard",'
    '"reason":"deny_tool: shell"},"run_id":"run-0001","schema":"taipanbox.dev/agent-event/v0.2",'
    '"severity":"high","source":"wardryx","ts":"2026-07-24T12:00:00Z","type":"policy_deny"}'
)
_VEC_HASH_1 = "sha256:b43502c0ed6893238f2635be7a909cde89df1c2eecaef4d84871b83cf21cb31b"

_VEC_EVENT_2 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:01Z",
    "source": "tokenfuse",
    "type": "budget_exhausted",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "critical",
    "run_id": "run-0001",
    "on_behalf_of": ["user://acme.example/alice", "agent://acme.example/orchestrator"],
    "data": {"budget_usd": 12.5, "n": 3, "note": "обмеження діє", "nested": {"b": 2, "a": 1}},
}
_VEC_CANONICAL_2 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"budget_usd":12.5,"n":3,'
    '"nested":{"a":1,"b":2},"note":"обмеження діє"},"on_behalf_of":["user://acme.example/alice",'
    '"agent://acme.example/orchestrator"],"run_id":"run-0001",'
    '"schema":"taipanbox.dev/agent-event/v0.2","severity":"critical","source":"tokenfuse",'
    '"ts":"2026-07-24T12:00:01Z","type":"budget_exhausted"}'
)
_VEC_HASH_2 = "sha256:488f1017967bf9510c62d7c31b9d5a0086ff2000d90a7d4266f171a131430243"

_VEC_EVENT_3 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:02Z",
    "source": "qryx",
    "type": "evidence_signed",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "info",
    "data": {"algo": "ML-DSA-87"},
}
_VEC_CANONICAL_3 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"algo":"ML-DSA-87"},'
    '"schema":"taipanbox.dev/agent-event/v0.2","severity":"info","source":"qryx",'
    '"ts":"2026-07-24T12:00:02Z","type":"evidence_signed"}'
)
_VEC_HASH_3 = "sha256:998cbc146b07e115318ce378e0579fcd1927066ef4316900ec7d66ba157e7c4b"

# Vector 4. `delegation_proof` (SPEC 5.2) is a top-level sibling of `data`, so
# it is exactly the member a language's event STRUCT may have no field for.
# Python is safe by construction here, because these functions take a dict and
# a dict carries whatever the line carried, and this vector is what turns "safe
# by construction" into "proved on every run". Go's event type was not: it
# hashed a re-marshal of its own struct until 2026-08-26, the member vanished
# before the digest, and an honestly chained stream was reported as tampered
# with by our own conformance tool.
_VEC_EVENT_4 = {
    "schema": "taipanbox.dev/agent-event/v0.3",
    "ts": "2026-08-26T12:00:03Z",
    "source": "vouchryx",
    "type": "delegation_issued",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "info",
    "run_id": "run-0001",
    "delegation_proof": {
        "jti": "tok-9f2c",
        "jkt": "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs",
        "iss": "https://idryx.acme.example",
        "exp": 1786000000,
    },
    "data": {"scope": "read:tickets"},
}
_VEC_CANONICAL_4 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"scope":"read:tickets"},'
    '"delegation_proof":{"exp":1786000000,"iss":"https://idryx.acme.example",'
    '"jkt":"NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs","jti":"tok-9f2c"},'
    '"run_id":"run-0001","schema":"taipanbox.dev/agent-event/v0.3","severity":"info",'
    '"source":"vouchryx","ts":"2026-08-26T12:00:03Z","type":"delegation_issued"}'
)
_VEC_HASH_4 = "sha256:97161b1b4dd0b64d683e27611279beb7024a91d0dba2fd736d10e96edabd7680"


@pytest.mark.parametrize(
    "event,canonical,expected_hash",
    [
        (_VEC_EVENT_1, _VEC_CANONICAL_1, _VEC_HASH_1),
        (_VEC_EVENT_2, _VEC_CANONICAL_2, _VEC_HASH_2),
        (_VEC_EVENT_3, _VEC_CANONICAL_3, _VEC_HASH_3),
        (_VEC_EVENT_4, _VEC_CANONICAL_4, _VEC_HASH_4),
    ],
)
def test_canonicalize_and_chain_hash_match_pinned_vectors(
    event: dict[str, Any], canonical: str, expected_hash: str
) -> None:
    """verdryx.events.canonicalize/chain_hash MUST reproduce the
    cross-language vectors byte-for-byte."""
    assert canonicalize(event) == canonical.encode("utf-8")
    assert chain_hash(event) == expected_hash


def test_emit_chains_two_events(tmp_path, event_schema, agent_id) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"a": 1})
    log.emit("eval_run", agent_id, {"a": 2})

    events = _read_ndjson(events_path)
    assert len(events) == 2
    assert "prev_hash" not in events[0]
    assert events[1]["prev_hash"] == chain_hash(events[0])
    for event in events:
        jsonschema.validate(instance=event, schema=event_schema)


def test_reopened_event_log_resumes_the_chain(tmp_path, agent_id) -> None:
    """One file, one chain: a new EventLog over an existing file continues
    the chain rather than restarting it (SPEC.md Sec 6.5)."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"a": 1})
    log.emit("eval_run", agent_id, {"a": 2})

    resumed = EventLog(events_path)
    resumed.emit("eval_run", agent_id, {"a": 3})

    events = _read_ndjson(events_path)
    assert len(events) == 3
    assert events[2]["prev_hash"] == chain_hash(events[1])


def test_malformed_tail_starts_a_fresh_chain(tmp_path, agent_id) -> None:
    """A tail that does not parse as JSON is exactly like no file at all:
    EventLog starts a fresh chain rather than raising (fail-open)."""
    events_path = tmp_path / "events.ndjson"
    events_path.write_text("{not json at all\n")

    log = EventLog(events_path)
    log.emit("eval_run", agent_id, {"a": 1})

    lines = [line for line in events_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    new_event = json.loads(lines[1])
    assert "prev_hash" not in new_event


# ------------------------------------------------------------------
# agent_id shape (SPEC.md Sec 3.1)
# ------------------------------------------------------------------


def test_the_agent_id_rule_matches_the_vendored_schema(event_schema) -> None:
    """The two constants are a local copy of values agent-passport owns, so
    they are read back out of the schema rather than trusted.

    This is the check that makes the copy legitimate. Without it the module
    would carry a fourth hand-maintained copy of a wire rule, which is the
    shape this estate has been bitten by repeatedly.
    """
    declared = event_schema["properties"]["agent_id"]
    assert AGENT_ID_PATTERN.pattern == declared["pattern"]
    assert declared["maxLength"] == AGENT_ID_MAX_LENGTH


def test_a_nonconforming_agent_id_is_warned_counted_and_still_written(
    tmp_path, event_schema, caplog
) -> None:
    """An id the envelope rejects is reported, not swallowed and not refused.

    The event is written on purpose: refusing would empty the log for exactly
    the caller who needs to see the fault. So the line is there, a consumer
    validating it rejects it, and the operator has been told why.
    """
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="verdryx.events"):
        log.emit("eval_run", "planner", {"a": 1})

    assert log.nonconforming_agent_id == 1
    assert log.skipped_empty_agent_id == 0, "a malformed id is not an absent one"

    written = _read_ndjson(events_path)
    assert len(written) == 1, "the event is written anyway"
    assert written[0]["agent_id"] == "planner", "and it is written unchanged, not repaired"

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=written[0], schema=event_schema)

    assert any("does not match the Agent Passport grammar" in r.message for r in caplog.records)


def test_a_nonconforming_agent_id_is_warned_once_and_counted_every_time(tmp_path, caplog) -> None:
    """emit takes the id per call, so one misconfigured caller must not turn a
    log file into a flood. The count stays true; only the repetition stops."""
    log = EventLog(tmp_path / "events.ndjson")

    with caplog.at_level(logging.WARNING, logger="verdryx.events"):
        for _ in range(3):
            log.emit("eval_run", "planner", {"a": 1})

    assert log.nonconforming_agent_id == 3
    warnings = [r for r in caplog.records if "does not match" in r.message]
    assert len(warnings) == 1, f"warned {len(warnings)} times for one id"


def test_a_canonical_agent_id_is_neither_warned_nor_counted(tmp_path, agent_id, caplog) -> None:
    """The overeager case. A gate that fires on correct input gets deleted."""
    log = EventLog(tmp_path / "events.ndjson")

    with caplog.at_level(logging.WARNING, logger="verdryx.events"):
        log.emit("eval_run", agent_id, {"a": 1})

    assert log.nonconforming_agent_id == 0
    assert not [r for r in caplog.records if "does not match" in r.message]


def test_an_over_long_agent_id_is_nonconforming_even_though_it_matches(tmp_path) -> None:
    """Both halves of the rule, not just the grammar: the cap is the half a
    regex alone would miss."""
    long_id = "agent://acme.example/" + ("a" * 300)
    assert AGENT_ID_PATTERN.match(long_id) is not None, "the grammar accepts it"
    assert not is_canonical_agent_id(long_id), "the cap does not"

    log = EventLog(tmp_path / "events.ndjson")
    log.emit("eval_run", long_id, {"a": 1})
    assert log.nonconforming_agent_id == 1


# ------------------------------------------------------------------
# The 1.0 event contract (agent-passport SPEC.md Sec 6.4.1)
# ------------------------------------------------------------------


def test_a_v1_0_stamped_event_this_emitter_writes_validates_under_the_vendored_v1_0_contract(
    tmp_path, event_schema, event_schema_v1_0, agent_id
) -> None:
    """SPEC 6.4.1 says a consumer must accept v1.0 and a producer keeps
    emitting the version it emits today, moving to v1.0 in its own release.
    This proves verdryx's lines are already v1.0-shaped, so that move will be
    a one-constant change here, not a rewrite of what EventLog.emit builds;
    and it pins that the constant has NOT moved yet in this change.
    """
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

    # Unchanged behaviour: verdryx still emits v0.2 today, and it still
    # validates under the v0.2 contract.
    jsonschema.validate(instance=event, schema=event_schema)
    assert SCHEMA == "taipanbox.dev/agent-event/v0.2"

    # The line verdryx writes today is already v1.0-shaped: stamping the
    # version string is the only edit needed to validate under v1.0.
    v1_0_event = dict(event)
    v1_0_event["schema"] = "taipanbox.dev/agent-event/v1.0"
    jsonschema.validate(instance=v1_0_event, schema=event_schema_v1_0)


def test_the_vendored_v1_0_contract_widens_only_the_subject(
    event_schema, event_schema_v1_0
) -> None:
    """The one widening from v0.2 to v1.0 is the `claimed:` subject (SPEC 3.3).
    Verdryx never writes one: EventLog.emit always writes the Passport id the
    caller evaluated under, never a claim. A consumer that has not modelled
    claims is right to refuse a claimed subject, and SPEC 6.4.1 counts that as
    a processing failure rather than silent acceptance.
    """
    agent_id = event_schema_v1_0["properties"]["agent_id"]
    assert "/v1.0/" in event_schema_v1_0["$id"]
    assert event_schema_v1_0["properties"]["schema"]["const"] == "taipanbox.dev/agent-event/v1.0"
    assert agent_id["pattern"] == "^(claimed:)?agent://[a-z0-9.-]+/[a-z0-9._/-]+$"
    assert agent_id["maxLength"] == 263

    def other_properties(schema: dict[str, Any]) -> dict[str, Any]:
        others = dict(schema["properties"])
        del others["agent_id"]
        del others["schema"]
        return others

    assert other_properties(event_schema_v1_0) == other_properties(event_schema)
    assert event_schema_v1_0["required"] == event_schema["required"]

    def minimal_envelope(schema_version: str, subject: str) -> dict[str, Any]:
        return {
            "schema": schema_version,
            "ts": "2026-09-12T00:00:00.000Z",
            "source": "verdryx",
            "type": "eval_run",
            "agent_id": subject,
        }

    claimed = "claimed:agent://acme-bank.example/support/tier1-bot"
    jsonschema.validate(
        instance=minimal_envelope("taipanbox.dev/agent-event/v1.0", claimed),
        schema=event_schema_v1_0,
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance=minimal_envelope("taipanbox.dev/agent-event/v0.2", claimed),
            schema=event_schema,
        )
