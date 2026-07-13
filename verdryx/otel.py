"""Hand-rolled OTLP/HTTP-JSON span exporter for eval/drift operations.

Mirrors Wardryx's own exporter (``wardryx/internal/otel/otel.go``), which
in turn mirrors TokenFuse's (``tokenfuse/crates/gateway/src/otel.rs``): a
direct POST to ``<endpoint>/v1/traces`` built against the OTLP/HTTP JSON
wire format, not a full OpenTelemetry SDK. Stdlib-only (``urllib``), not
just SDK-free, matching this package's zero-core-dependency bias
(``pyproject.toml``'s ``dependencies = []``).

Fire-and-forget, same invariant as :mod:`verdryx.events`: exporting must
never block or fail the eval/drift operation it describes. A background
thread does the POST; any failure (DNS, connect, timeout, non-2xx) is
caught and dropped silently.

One real difference from Wardryx's exporter: Wardryx runs inside a
long-lived server process, so its fire-and-forget goroutine has the rest
of the process's lifetime to finish. Verdryx's ``eval``/``drift`` commands
are one-shot CLI invocations that exit as soon as the handler returns, and
a daemon thread does not survive its process exiting -- so every CLI
command handler that calls :meth:`OTLPExporter.export` must also call
:meth:`OTLPExporter.wait` before returning, or the export is silently
killed mid-flight essentially every time, not just occasionally. See
:meth:`OTLPExporter.wait`'s docstring.

Unlike :mod:`verdryx.events` (which only emits ``quality_drift`` on an
actual regression -- a governance-alert signal for the shared agent-event
bus), a span is exported for *every* eval run and *every* drift check
regardless of verdict: OTLP tracing's job is observability of what ran and
when, not alerting, so an unremarkable "on-track" check is exactly as
trace-worthy as a regression -- an operator graphing eval/drift frequency
over time in Grafana/Datadog/Honeycomb needs the full activity pattern.

No cross-service trace correlation: the trace id is derived from
``run_id`` alone (SHA-256, Wardryx's own scheme, not TokenFuse's Rust
``DefaultHasher``, which has no portable Python equivalent), so it is
stable across a Verdryx process but does not itself merge into the same
trace as TokenFuse's or Wardryx's spans for that run_id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Bounded so an unresponsive collector cannot leak threads/connections
#: indefinitely. Export already runs off the CLI command's own return path,
#: so this only protects the exporter's background work, never the caller.
_TIMEOUT_SECONDS = 5.0

#: SPAN_KIND_INTERNAL: eval/drift are local computations a CLI invocation
#: triggers, not an inbound request Verdryx serves (contrast Wardryx's own
#: SERVER-kind span for its inbound /v1/decide) or an outbound call it
#: makes (contrast TokenFuse's CLIENT-kind span for its outbound LLM call).
_SPAN_KIND_INTERNAL = 1


@dataclass(frozen=True)
class Span:
    """One eval/drift operation worth exporting as an OTLP span.

    Args:
        name: Span name, e.g. ``"eval_run"`` or ``"quality_drift"``.
        run_id: Correlates every span for one eval run onto one trace.
        agent_id: The evaluated agent's Passport id, or ``None`` -- unlike
            :meth:`verdryx.events.EventLog.emit`, a span with no agent_id
            is still exported (agent_id is optional context for tracing,
            not a required envelope field the way agent-event demands).
        attributes: Free-form span attributes, stringified into OTLP's
            ``stringValue`` form (span attributes are a debugging/
            observability aid, not a typed contract another service
            parses, so precision loss from e.g. float-to-string is
            acceptable here in a way it would not be in events.py's
            ``data`` payload).
        timestamp_ns: Wall-clock time of the operation, Unix nanoseconds.
    """

    name: str
    run_id: str
    agent_id: str | None
    attributes: dict[str, Any]
    timestamp_ns: int


class OTLPExporter:
    """Posts one OTLP/HTTP-JSON span per :meth:`export` call.

    Args:
        endpoint: Collector base URL. Normalized to end in ``/v1/traces``
            the same way Wardryx's and TokenFuse's own exporters do.
    """

    def __init__(self, endpoint: str) -> None:
        base = endpoint.rstrip("/")
        self._url = base if base.endswith("/v1/traces") else base + "/v1/traces"
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        """Export span in a background thread and return immediately.

        The thread is tracked so :meth:`wait` can join it later -- see
        :meth:`wait`'s docstring for why a one-shot CLI needs that call
        where a long-running server (Wardryx's own exporter) does not.
        """
        payload = _build_payload(span)
        thread = threading.Thread(target=self._post, args=(payload,), daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def wait(self, timeout: float = _TIMEOUT_SECONDS + 1.0) -> None:
        """Block until every :meth:`export` call so far has finished or
        timed out.

        Wardryx's own OTLP exporter (this module's Go counterpart) never
        needs this: it runs inside a long-lived server process, so a
        fire-and-forget goroutine has the rest of the process's lifetime to
        complete its POST. Verdryx's ``eval``/``drift`` commands are
        one-shot CLI invocations -- the process exits as soon as the
        command handler returns, and a daemon thread does not get to
        finish once its process exits. Without this call, an export
        started right before a command returns would be silently killed
        mid-flight essentially every time, not just occasionally: caught
        by a real end-to-end test (a live local collector never receiving
        the span), not just reasoned about.

        CLI command handlers call this once, after their own printed
        output is ready, so the process's own visible work is never
        delayed by a slow collector -- only the final exit is.
        """
        with self._lock:
            threads = list(self._threads)
        deadline = time.monotonic() + timeout
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)

    def _post(self, payload: dict[str, Any]) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS):
                pass
        except (urllib.error.URLError, OSError, ValueError):
            logger.debug("verdryx.otel: export failed (dropped)", exc_info=True)


def build_payload(span: Span) -> dict[str, Any]:
    """Build span's OTLP/HTTP-JSON ``resourceSpans`` payload.

    Pure and deterministic given the same span -- unit-tested directly, no
    network involved, mirroring Wardryx's own ``buildPayload``. Exported
    (not prefixed ``_build_payload``) so tests can call it without
    reaching into the module's private API.
    """
    return _build_payload(span)


def _build_payload(span: Span) -> dict[str, Any]:
    nanos = str(span.timestamp_ns)
    attrs = [_attr_str("verdryx.run_id", span.run_id)]
    if span.agent_id:
        attrs.append(_attr_str("verdryx.agent_id", span.agent_id))
    for key, value in span.attributes.items():
        if value is None:
            continue
        attrs.append(_attr_str(f"verdryx.{key}", str(value)))

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr_str("service.name", "verdryx")]},
                "scopeSpans": [
                    {
                        "scope": {"name": "verdryx"},
                        "spans": [
                            {
                                "traceId": _trace_id(span.run_id),
                                "spanId": _span_id(span.run_id, span.name, span.timestamp_ns),
                                "name": span.name,
                                "kind": _SPAN_KIND_INTERNAL,
                                "startTimeUnixNano": nanos,
                                "endTimeUnixNano": nanos,
                                "attributes": attrs,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _attr_str(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _trace_id(run_id: str) -> str:
    """16-byte (32 hex char) OTLP trace id derived from run_id alone, so
    every span for one eval run shares a trace."""
    return hashlib.sha256(f"verdryx-trace|{run_id}".encode()).hexdigest()[:32]


def _span_id(run_id: str, name: str, timestamp_ns: int) -> str:
    """8-byte (16 hex char) OTLP span id. Folding in name and timestamp (not
    just run_id) keeps an eval_run span and a same-run quality_drift span
    -- or two drift checks against the same run in quick succession -- from
    colliding on the same span id."""
    return hashlib.sha256(f"verdryx-span|{run_id}|{name}|{timestamp_ns}".encode()).hexdigest()[:16]
