"""Shared fixtures for the Verdryx test suite."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from verdryx.models import EvalCase, EvalSet, GraderKind

#: A syntactically valid Agent Passport id, matching the pattern enforced by
#: agent-event.v0.2.schema.json (^agent://[a-z0-9.-]+/[a-z0-9._/-]+$).
AGENT_ID = "agent://acme-bank.example/support/tier1-bot"

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "agent-event.v0.2.schema.json"
_SCHEMA_PATH_V1_0 = Path(__file__).parent / "fixtures" / "agent-event.v1.0.schema.json"


@pytest.fixture()
def agent_id() -> str:
    return AGENT_ID


@pytest.fixture()
def event_schema() -> dict[str, Any]:
    """The vendored Agent Passport agent-event v0.2 JSON Schema."""
    return json.loads(_SCHEMA_PATH.read_text())


@pytest.fixture()
def event_schema_v1_0() -> dict[str, Any]:
    """The vendored Agent Passport agent-event v1.0 JSON Schema."""
    return json.loads(_SCHEMA_PATH_V1_0.read_text())


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


# ------------------------------------------------------------------
# A real loopback HTTP server standing in for typryx (tests/test_graders.py,
# tests/test_cli.py). No mocking of urllib anywhere: TypryxClient talks to
# this exactly like it would talk to a real typryx process, over
# 127.0.0.1 on a kernel-assigned port (port 0).
# ------------------------------------------------------------------


class _ScriptedTypryxHandler(BaseHTTPRequestHandler):
    #: Class-level state, reset per fixture instantiation below -- one
    #: handler *class* is bound to the whole server, so a fresh dict/list
    #: is installed on it for every test rather than shared across tests.
    lock: threading.Lock
    requests: list[dict[str, Any]]
    #: Each queued entry is (status, body_bytes) -- the fixture's own
    #: `script()` JSON-encodes a dict for the common case, but a test can
    #: queue arbitrary bytes directly (see `script_raw`) to simulate a
    #: malformed response body.
    responses: dict[str, list[tuple[int, bytes]]]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"_unparseable": raw.decode("utf-8", errors="replace")}
        with type(self).lock:
            type(self).requests.append(
                {"path": self.path, "key": self.headers.get("X-Typryx-Key"), "body": body}
            )
            queue = type(self).responses.get(self.path)
            status, payload = (
                queue.pop(0) if queue else (404, json.dumps({"error": "unscripted_path"}).encode())
            )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:  # silence stderr request logging
        pass


class TypryxFake:
    """A controllable, real HTTP stand-in for typryx.

    `script(path, status, payload)` queues one response for the next
    request to `path` (FIFO per path, so a test can script a sequence of
    answers to repeated calls). `requests` is every request received so
    far, in order, each `{"path", "key", "body"}` -- `key` is exactly what
    arrived in the `X-Typryx-Key` header, and `body` is the parsed JSON
    request body, so a test can assert on precisely what verdryx sent.
    """

    def __init__(self, server: ThreadingHTTPServer, url: str) -> None:
        self._server = server
        self.url = url

    def script(self, path: str, status: int, payload: dict[str, Any]) -> None:
        _ScriptedTypryxHandler.responses.setdefault(path, []).append(
            (status, json.dumps(payload).encode("utf-8"))
        )

    def script_raw(self, path: str, status: int, body: bytes) -> None:
        """Like `script`, but queues raw response bytes instead of a dict --
        for a test simulating a response body that is not valid JSON."""
        _ScriptedTypryxHandler.responses.setdefault(path, []).append((status, body))

    @property
    def requests(self) -> list[dict[str, Any]]:
        with _ScriptedTypryxHandler.lock:
            return list(_ScriptedTypryxHandler.requests)


@pytest.fixture()
def typryx_fake():
    """A real `http.server.ThreadingHTTPServer` on 127.0.0.1:0, standing in
    for typryx. No mocking of urllib: TypryxClient makes a real HTTP
    request to this server over the loopback interface, exactly as it
    would to a real typryx process."""
    _ScriptedTypryxHandler.lock = threading.Lock()
    _ScriptedTypryxHandler.requests = []
    _ScriptedTypryxHandler.responses = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedTypryxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake = TypryxFake(server, f"http://127.0.0.1:{server.server_port}")
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
