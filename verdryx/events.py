"""Opt-in NDJSON exporter for Agent Passport events.

See ``agent-passport/SPEC.md`` Sec 6 and
``schemas/agent-event.v0.2.schema.json`` in the sibling
``TAIPANBOX/agent-passport`` repo for the wire format this module
implements. Verdryx is a wave-2 service and is the ``source: "verdryx"``
emitter in that shared envelope, on schema v0.2 (an open `source` string,
not the v0.1 closed enum -- see SPEC.md Sec 6.4): every event written here
lets the surrounding governance stack observe what Verdryx measured about
an agent's quality, without any other product depending on Verdryx's
internals.

Why this does not violate "no network calls at write time"
------------------------------------------------------------
:meth:`EventLog.emit` performs a local filesystem append: open, write,
close. No socket is opened, no DNS lookup happens, no external service is
contacted. The "no network calls at write time" invariant exists so Verdryx
stays usable fully offline and so write latency stays bounded by local disk
I/O rather than a remote service's availability; a local NDJSON append
preserves both properties exactly as writing to the SQLite store itself
does. If a backend wants to ship these events over the network, that
belongs in a separate consumer process that tails the file, never in this
module. (VERDRYX_OTLP_ENDPOINT / verdryx.otel is exactly that separate
consumer for a related but distinct purpose -- OTLP tracing, not the
agent-event bus; this module does not use it.)

Opt-in only
-----------
:class:`EventLog` is only ever constructed when the caller asks for it, via
an explicit path or the ``VERDRYX_EVENTS_PATH`` environment variable (see
:func:`resolve_events_path`). When neither is set, no EventLog is
constructed and every call site pays exactly one ``is None`` check: no file
handle, no thread, no allocation.

Fail-open
---------
Any I/O error raised while appending is caught inside :meth:`EventLog.emit`,
logged as a warning, and swallowed. Losing an event is acceptable; losing or
delaying the eval/drift operation the event describes is not. ``emit``
therefore never raises.

Event types and severities (fixed mapping, per SPEC.md Sec 6.2 "verdryx" row)
---------------------------------------------------------------------------
=====================  ========  ==============================================
type                   severity  data
=====================  ========  ==============================================
eval_run               info      run_id, model, cases, mean_score, total_tokens,
                                  total_cost_usd
quality_score          info      case_id, value, tokens, cost_usd
quality_drift          high      baseline_id, window, mean_score, delta, verdict,
                                  baseline_n, t_statistic, ci_low, ci_high
=====================  ========  ==============================================

``baseline_n``/``t_statistic``/``ci_low``/``ci_high`` come from
:func:`verdryx.drift.compute_drift`'s optional two-sample significance
check (Welch's t-statistic plus a bootstrap confidence interval on the
delta). They are ``0``/``None``/``None``/``None`` when that check did not
run (fewer than 2 scores in either the baseline run or the recent window).

``prev_hash`` chain (SPEC.md Sec 6.5)
---------------------------------------
Each events file maintains its own append-only hash chain: every event's
``prev_hash`` is ``"sha256:" + hex(sha256(C))``, where ``C`` is the RFC 8785
(JSON Canonicalization Scheme) serialization of the PREVIOUS event in this
file with its own ``prev_hash`` removed -- see :func:`canonicalize` and
:func:`chain_hash`. The first event in a file carries no ``prev_hash``, and
reopening an existing file resumes the chain from its tail instead of
starting a new one, so one file stays one chain across process restarts.
Resuming is fail-open like the rest of this module: an absent, empty, or
malformed tail starts a fresh chain rather than blocking construction.

This is tamper-EVIDENCE, not tamper-proof: it makes a dropped or edited
line detectable, not impossible for an attacker who can rewrite the whole
file. Verify a file with ``agent-conform -chain <file>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import rfc8785

logger = logging.getLogger(__name__)

#: Schema identifier for the envelope this module emits. See SPEC.md Sec 6.4:
#: wave-2 services (wardryx, verdryx, mockryx) emit v0.2, whose only
#: difference from v0.1 is that `source` is an open string, not a closed enum.
SCHEMA = "taipanbox.dev/agent-event/v0.2"

#: This module's fixed ``source`` value in the shared envelope.
SOURCE = "verdryx"

Severity = Literal["info", "low", "medium", "high", "critical"]

#: Fixed event-type -> severity mapping. See the module docstring table.
#: Not user-configurable: severities are a taxonomy decision, not per-call
#: data. Matches agent-passport/SPEC.md Sec 6.2's "verdryx" row exactly.
EVENT_SEVERITY: dict[str, Severity] = {
    "eval_run": "info",
    "quality_score": "info",
    "quality_drift": "high",
}

#: Environment variable fallback for an explicit events path.
ENV_EVENTS_PATH = "VERDRYX_EVENTS_PATH"


def resolve_events_path(explicit: str | Path | None) -> Path | None:
    """Resolve the effective events file path.

    Args:
        explicit: An explicit path, e.g. from a CLI ``--events`` flag.

    Returns:
        ``Path(explicit)`` if given; otherwise ``Path(VERDRYX_EVENTS_PATH)``
        if that environment variable is set and non-empty; otherwise
        ``None`` (events are fully disabled).
    """
    if explicit is not None:
        return Path(explicit)
    env_value = os.environ.get(ENV_EVENTS_PATH)
    if env_value:
        return Path(env_value)
    return None


def _now_rfc3339() -> str:
    """Return the current UTC time as RFC 3339 with a literal ``Z`` suffix."""
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


#: How far back :func:`EventLog.__post_init__` reads when resuming an
#: existing file's chain (SPEC.md Sec 6.5). One complete event line always
#: fits comfortably inside this window: real envelopes run to a few hundred
#: bytes, and 1 MiB is orders of magnitude beyond any single line this
#: module writes.
_RESUME_WINDOW = 1 << 20


def canonicalize(envelope: dict[str, Any]) -> bytes:
    """Return the RFC 8785 (JCS) canonical serialization of *envelope*.

    SPEC.md Sec 6.5: the chain-hash input is the JCS canonical form of an
    event object with its own ``prev_hash`` field removed. The removal
    happens on a shallow copy -- the caller's dict is never mutated -- and
    canonicalization itself (key sorting, number/string normalization) is
    delegated entirely to ``rfc8785`` (Trail of Bits), never hand-rolled.
    """
    copy = dict(envelope)
    copy.pop("prev_hash", None)
    return rfc8785.dumps(copy)


def chain_hash(envelope: dict[str, Any]) -> str:
    """Return the SPEC.md Sec 6.5 hash of *envelope*.

    ``"sha256:" + hex(sha256(canonicalize(envelope)))`` -- the value the
    NEXT event in a chained NDJSON stream carries as its own ``prev_hash``.
    """
    digest = hashlib.sha256(canonicalize(envelope)).hexdigest()
    return f"sha256:{digest}"


def _tail_chain_hash(path: Path) -> str | None:
    """Resume a chain from *path*'s existing tail (SPEC.md Sec 6.5).

    Reads at most the last :data:`_RESUME_WINDOW` bytes of *path*, keeps the
    last non-blank line, and parses it as one event. Returns that event's
    :func:`chain_hash` so the next :meth:`EventLog.emit` call re-links to
    what is actually on disk -- one file stays one chain across process
    restarts.

    Fail-open, mirroring the rest of this module: a missing or empty file,
    a tail that is not valid JSON or not a JSON object, or any I/O error
    along the way all yield ``None`` (start a fresh chain) rather than
    raising. A malformed tail is exactly the same "start fresh" case as no
    file at all -- nothing here ever blocks construction.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None

    try:
        with path.open("rb") as fh:
            start = max(0, size - _RESUME_WINDOW)
            fh.seek(start)
            tail = fh.read()
    except OSError:
        return None

    lines = tail.split(b"\n")
    if start > 0:
        # A mid-file cut: the first scanned line is likely partial.
        lines = lines[1:]

    last: bytes | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped:
            last = stripped
    if last is None:
        return None

    try:
        parsed = json.loads(last)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    try:
        return chain_hash(parsed)
    except rfc8785.CanonicalizationError:
        return None


@dataclass
class EventLog:
    """Appends Agent Passport event envelopes (SPEC.md Sec 6) to an NDJSON file.

    The destination file is opened in append mode on every :meth:`emit`
    call rather than held open, so a deleted parent directory or revoked
    permission surfaces as a logged warning on the next write rather than a
    crash at construction time.

    Construction also seeds the SPEC.md Sec 6.5 ``prev_hash`` chain from the
    file's existing tail, if any (see :func:`_tail_chain_hash`), so a fresh
    :class:`EventLog` opened over a file another instance already wrote
    resumes the same chain rather than restarting it.

    Args:
        path: Destination NDJSON file. Parent directory must already exist;
            if it does not, writes fail open (see the module docstring).
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    skipped_empty_agent_id: int = field(default=0, init=False)
    _next_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._next_hash = _tail_chain_hash(self.path)

    def emit(
        self,
        event_type: str,
        agent_id: str | None,
        data: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        """Append one event to the NDJSON file.

        Never raises: any failure to resolve or write the event is caught,
        logged as a warning, and swallowed so the caller's eval/drift
        operation always completes.

        The event is stamped with the SPEC.md Sec 6.5 ``prev_hash`` chain
        (omitted at a chain head, i.e. the first event in a fresh file) and
        the chain only advances after a successful write, all under
        :attr:`_lock` -- see the module docstring.

        Args:
            event_type: One of the fixed verdryx event types documented on
                :data:`EVENT_SEVERITY`. Unknown types default to ``"info"``
                severity rather than raising, so a future new type never
                breaks emission.
            agent_id: The evaluated agent's Passport id (an opaque string,
                e.g. an Agent Passport ``agent://...`` URI). If ``None`` or
                empty, the event is skipped and counted in
                :attr:`skipped_empty_agent_id` -- Verdryx never fabricates
                an agent_id to satisfy the envelope's required field.
            data: Free-form event payload, owned by the caller.
            run_id: Optional eval-run correlation id.
        """
        if not agent_id:
            self.skipped_empty_agent_id += 1
            return

        try:
            envelope: dict[str, Any] = {
                "schema": SCHEMA,
                "ts": _now_rfc3339(),
                "source": SOURCE,
                "type": event_type,
                "severity": EVENT_SEVERITY.get(event_type, "info"),
                "agent_id": agent_id,
                "data": data,
            }
            if run_id is not None:
                envelope["run_id"] = run_id

            with self._lock:
                if self._next_hash:
                    envelope["prev_hash"] = self._next_hash
                line = json.dumps(envelope, separators=(",", ":")) + "\n"
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                self._next_hash = chain_hash(envelope)
        except OSError:
            logger.warning(
                "verdryx.events: failed to write %r event to %s (event dropped)",
                event_type,
                self.path,
                exc_info=True,
            )
