"""Environment-driven configuration, read once at process start.

verdryx.cli reads Config.from_env() exactly once in main() and threads the
result down to whichever subcommand handles the call, rather than each
module reaching into os.environ independently. CLI flags always take
precedence over the environment (see cli.py); Config only supplies defaults
for flags the caller omitted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

ENV_DB = "VERDRYX_DB"
ENV_EVENTS_PATH = "VERDRYX_EVENTS_PATH"
ENV_OTLP_ENDPOINT = "VERDRYX_OTLP_ENDPOINT"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_ANTHROPIC_BASE_URL = "ANTHROPIC_BASE_URL"

#: The installed stack's published home, where `stack-up` puts binaries, the
#: shared venv and the per-tool stores. Honoured here for the same reason
#: stack-up honours it: so the whole layout can be pointed at a scratch
#: directory for a clean-machine test without touching the real one.
ENV_TAIPAN_HOME = "TAIPAN_HOME"

#: Filename of the SQLite store, shared by both the published-home and the
#: working-directory candidates below.
DB_FILENAME = "verdryx.db"

#: Last-resort store path, relative to the process's working directory.
DEFAULT_DB = DB_FILENAME


def default_taipan_home(env: Mapping[str, str]) -> str:
    """The published stack home: ``$TAIPAN_HOME``, else ``~/.taipan``.

    Reads ``HOME`` from ``env`` rather than calling ``expanduser`` blindly,
    so a caller passing an explicit mapping (tests, and any embedder) gets an
    answer derived only from what it passed.
    """
    explicit = env.get(ENV_TAIPAN_HOME)
    if explicit:
        return explicit
    user_home = env.get("HOME") or os.path.expanduser("~")
    return os.path.join(user_home, ".taipan")


def resolve_db_path(env: Mapping[str, str]) -> str:
    """Where the store lives, most explicit candidate first.

    1. ``VERDRYX_DB``, honoured verbatim even if it does not exist yet: an
       explicit override must fail loudly on the path the operator named,
       never fall through to a different store silently.
    2. ``<taipan home>/verdryx.db``, but only when that directory already
       exists, i.e. only on a machine where the stack is actually installed.
    3. ``./verdryx.db``, this tool's historical default.

    Why candidate 2 exists at all. Verdryx has no server, so its SQLite file
    IS its machine-readable surface, and the Genaryx console reads it from
    exactly these three places, in exactly this order (the console's own
    ``quality/env.rs`` documents the same list). Before this, verdryx wrote
    to the working directory while the console looked in the published home,
    so a plain ``verdryx eval`` run from anywhere else produced real results
    the console could never find. Nothing was lost, but nothing was seen
    either, which for a plane whose entire job is to be looked at amounts to
    the same thing.

    Note the deliberate asymmetry with the console: the console requires the
    FILE to exist before trusting candidate 2 (an unchecked guess would be
    indistinguishable from a real stack-managed store), while this function
    requires only the DIRECTORY. It has to: on the write side, waiting for
    the file to exist would mean nothing ever creates it.
    """
    explicit = env.get(ENV_DB)
    if explicit:
        return explicit
    home = default_taipan_home(env)
    if os.path.isdir(home):
        return os.path.join(home, DB_FILENAME)
    return DEFAULT_DB


@dataclass(frozen=True)
class Config:
    """Process configuration, resolved once from the environment.

    Args:
        db_path: SQLite store path.
        events_path: Opt-in NDJSON event log path, or None if unset (events
            are then fully disabled; see verdryx.events).
        otlp_endpoint: Collector base URL for verdryx.otel.OTLPExporter, a
            span per eval run / drift check. Independent of events_path:
            OTLP tracing and the NDJSON agent-event log are separate,
            optional sinks with different purposes (see verdryx.otel's
            module docstring), either or both may be enabled.
        anthropic_api_key: Explicit Anthropic API key for the real
            LLMJudgeGrader adapter. When None, the Anthropic SDK falls back
            to its own ANTHROPIC_API_KEY lookup.
        anthropic_base_url: Optional proxy endpoint (e.g. TokenFuse) for
            the real LLMJudgeGrader adapter.
    """

    db_path: str
    events_path: str | None
    otlp_endpoint: str | None
    anthropic_api_key: str | None
    anthropic_base_url: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        e = env if env is not None else os.environ
        return cls(
            db_path=resolve_db_path(e),
            events_path=e.get(ENV_EVENTS_PATH) or None,
            otlp_endpoint=e.get(ENV_OTLP_ENDPOINT) or None,
            anthropic_api_key=e.get(ENV_ANTHROPIC_API_KEY) or None,
            anthropic_base_url=e.get(ENV_ANTHROPIC_BASE_URL) or None,
        )
