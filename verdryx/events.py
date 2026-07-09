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
does. If a future backend wants to ship these events over the network, that
belongs in a separate consumer process that tails the file, never in this
module. (VERDRYX_OTLP_ENDPOINT in verdryx.config is read for exactly that
future consumer; this module does not use it.)

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
quality_drift          high      baseline_id, window, mean_score, delta, verdict
=====================  ========  ==============================================

No ``prev_hash``
-----------------
Verdryx keeps no hash chain over its own writes, so the optional
``prev_hash`` field (present in TokenFuse's audit trail) is always omitted
rather than fabricated.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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


@dataclass
class EventLog:
    """Appends Agent Passport event envelopes (SPEC.md Sec 6) to an NDJSON file.

    The destination file is opened in append mode on every :meth:`emit`
    call rather than held open, so a deleted parent directory or revoked
    permission surfaces as a logged warning on the next write rather than a
    crash at construction time.

    Args:
        path: Destination NDJSON file. Parent directory must already exist;
            if it does not, writes fail open (see the module docstring).
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    skipped_empty_agent_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

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

            line = json.dumps(envelope, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            logger.warning(
                "verdryx.events: failed to write %r event to %s (event dropped)",
                event_type,
                self.path,
                exc_info=True,
            )
