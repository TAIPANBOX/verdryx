"""Tests for verdryx.otel.

Mirrors wardryx/internal/otel/otel_test.go's coverage: payload shape is
pure and unit-tested without HTTP; Export's fire-and-forget behavior is
tested against a real local HTTP server (stdlib http.server, matching this
repo's zero-core-dependency bias -- no third-party test-server library).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from verdryx.otel import OTLPExporter, Span, build_payload


def _span(**overrides: object) -> Span:
    defaults: dict[str, object] = {
        "name": "eval_run",
        "run_id": "run-1",
        "agent_id": "agent://acme.example/support/bot",
        "attributes": {"model": "stub", "cases": 5, "mean_score": 0.9},
        "timestamp_ns": 1_700_000_000_000_000_000,
    }
    defaults.update(overrides)
    return Span(**defaults)  # type: ignore[arg-type]


def _payload_span(payload: dict[str, object]) -> dict[str, object]:
    resource_spans = payload["resourceSpans"]
    assert isinstance(resource_spans, list) and len(resource_spans) == 1
    scope_spans = resource_spans[0]["scopeSpans"]
    assert isinstance(scope_spans, list) and len(scope_spans) == 1
    spans = scope_spans[0]["spans"]
    assert isinstance(spans, list) and len(spans) == 1
    return spans[0]


def _attr(span: dict[str, object], key: str) -> object | None:
    for entry in span["attributes"]:  # type: ignore[union-attr]
        if entry["key"] == key:
            return entry["value"]["stringValue"]
    return None


# ------------------------------------------------------------------
# Endpoint normalization
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "want"),
    [
        ("http://h:4318", "http://h:4318/v1/traces"),
        ("http://h:4318/", "http://h:4318/v1/traces"),
        ("http://h:4318/v1/traces", "http://h:4318/v1/traces"),
        ("http://h:4318/v1/traces/", "http://h:4318/v1/traces"),
    ],
)
def test_normalizes_endpoint(endpoint: str, want: str) -> None:
    assert OTLPExporter(endpoint)._url == want


# ------------------------------------------------------------------
# build_payload: pure, deterministic, unit-tested without HTTP
# ------------------------------------------------------------------


def test_build_payload_shape() -> None:
    span = _payload_span(build_payload(_span()))
    assert span["name"] == "eval_run"
    assert span["kind"] == 1  # SPAN_KIND_INTERNAL
    assert len(span["traceId"]) == 32
    assert len(span["spanId"]) == 16
    assert _attr(span, "verdryx.run_id") == "run-1"
    assert _attr(span, "verdryx.agent_id") == "agent://acme.example/support/bot"
    assert _attr(span, "verdryx.model") == "stub"
    assert _attr(span, "verdryx.cases") == "5"
    assert _attr(span, "verdryx.mean_score") == "0.9"


def test_build_payload_omits_none_attributes() -> None:
    span = _payload_span(
        build_payload(_span(attributes={"delta": -0.2, "t_statistic": None, "ci_low": None}))
    )
    assert _attr(span, "verdryx.delta") == "-0.2"
    assert _attr(span, "verdryx.t_statistic") is None
    assert _attr(span, "verdryx.ci_low") is None


def test_build_payload_omits_agent_id_when_none() -> None:
    span = _payload_span(build_payload(_span(agent_id=None)))
    assert _attr(span, "verdryx.agent_id") is None


def test_build_payload_service_name_on_resource() -> None:
    payload = build_payload(_span())
    resource = payload["resourceSpans"][0]["resource"]  # type: ignore[index]
    attrs = resource["attributes"]
    assert len(attrs) == 1
    assert attrs[0]["key"] == "service.name"
    assert attrs[0]["value"]["stringValue"] == "verdryx"


def test_same_run_shares_trace_id() -> None:
    a = _payload_span(build_payload(_span(name="eval_run", timestamp_ns=1)))
    b = _payload_span(build_payload(_span(name="quality_drift", timestamp_ns=2)))
    assert a["traceId"] == b["traceId"]


def test_different_runs_get_different_trace_ids() -> None:
    a = _payload_span(build_payload(_span(run_id="run-1")))
    b = _payload_span(build_payload(_span(run_id="run-2")))
    assert a["traceId"] != b["traceId"]


def test_different_spans_for_same_run_get_different_span_ids() -> None:
    # An eval_run span, then a later quality_drift check against the same
    # run: same trace, distinct spans.
    a = _payload_span(build_payload(_span(name="eval_run", timestamp_ns=1)))
    b = _payload_span(build_payload(_span(name="quality_drift", timestamp_ns=2)))
    assert a["traceId"] == b["traceId"]
    assert a["spanId"] != b["spanId"]


def test_build_payload_deterministic() -> None:
    span = _span()
    assert json.dumps(build_payload(span)) == json.dumps(build_payload(span))


# ------------------------------------------------------------------
# OTLPExporter.export: fire-and-forget over real HTTP
# ------------------------------------------------------------------


class _CapturingHandler(BaseHTTPRequestHandler):
    received: threading.Event
    path: str | None = None
    content_type: str | None = None
    body: dict[str, object] | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).path = self.path
        type(self).content_type = self.headers.get("Content-Type")
        type(self).body = json.loads(raw)
        self.send_response(200)
        self.end_headers()
        type(self).received.set()

    def log_message(self, *_args: object) -> None:  # silence stderr request logging
        pass


def _start_capturing_server() -> tuple[HTTPServer, str]:
    _CapturingHandler.received = threading.Event()
    _CapturingHandler.path = _CapturingHandler.content_type = _CapturingHandler.body = None
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_export_posts_to_configured_endpoint() -> None:
    server, url = _start_capturing_server()
    try:
        OTLPExporter(url).export(_span())
        assert _CapturingHandler.received.wait(timeout=2.0), (
            "timed out waiting for the exported span to be posted"
        )
        assert _CapturingHandler.path == "/v1/traces"
        assert _CapturingHandler.content_type == "application/json"
        assert "resourceSpans" in (_CapturingHandler.body or {})
    finally:
        server.shutdown()


def test_export_does_not_block_caller() -> None:
    # A server that never responds within the test's patience -- export()
    # must still return immediately, proving it never blocks on the POST.
    block = threading.Event()
    accepted = threading.Event()

    class _Blocking(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            accepted.set()
            block.wait(timeout=5.0)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Blocking)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        done = threading.Event()

        def run() -> None:
            OTLPExporter(f"http://127.0.0.1:{server.server_port}").export(_span())
            done.set()

        threading.Thread(target=run, daemon=True).start()
        assert done.wait(timeout=1.0), (
            "export() blocked the caller instead of firing the POST in the background"
        )
    finally:
        block.set()
        server.shutdown()


def test_export_survives_unreachable_endpoint() -> None:
    # Nothing listening on this port: export() must not raise and must
    # return promptly regardless of what the background thread does.
    done = threading.Event()

    def run() -> None:
        OTLPExporter("http://127.0.0.1:1").export(_span())
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=1.0), (
        "export() did not return promptly against an unreachable endpoint"
    )


# ------------------------------------------------------------------
# OTLPExporter.wait: the one-shot-CLI-specific delivery guarantee
# ------------------------------------------------------------------


def test_wait_blocks_until_export_actually_delivered() -> None:
    # Without wait(), a one-shot CLI process would exit and kill the
    # daemon thread before this POST completes -- this is the exact bug a
    # live end-to-end CLI run caught (a real collector never receiving the
    # span). wait() must make delivery observable before it returns.
    server, url = _start_capturing_server()
    try:
        exporter = OTLPExporter(url)
        exporter.export(_span())
        exporter.wait(timeout=2.0)
        assert _CapturingHandler.received.is_set(), (
            "wait() returned before the export it was waiting for was delivered"
        )
    finally:
        server.shutdown()


def test_wait_respects_timeout_against_an_unresponsive_collector() -> None:
    block = threading.Event()
    accepted = threading.Event()

    class _Blocking(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            accepted.set()
            block.wait(timeout=5.0)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Blocking)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        exporter = OTLPExporter(f"http://127.0.0.1:{server.server_port}")
        exporter.export(_span())
        assert accepted.wait(timeout=2.0), "collector never received the request"

        started = time.monotonic()
        exporter.wait(timeout=0.2)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"wait(timeout=0.2) took {elapsed:.2f}s, should return promptly"
    finally:
        block.set()
        server.shutdown()


def test_wait_with_no_exports_returns_immediately() -> None:
    started = time.monotonic()
    OTLPExporter("http://127.0.0.1:1").wait(timeout=2.0)
    assert time.monotonic() - started < 0.5
