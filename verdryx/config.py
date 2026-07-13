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

#: Default SQLite store path when neither --db nor VERDRYX_DB is set.
DEFAULT_DB = "verdryx.db"


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
            db_path=e.get(ENV_DB) or DEFAULT_DB,
            events_path=e.get(ENV_EVENTS_PATH) or None,
            otlp_endpoint=e.get(ENV_OTLP_ENDPOINT) or None,
            anthropic_api_key=e.get(ENV_ANTHROPIC_API_KEY) or None,
            anthropic_base_url=e.get(ENV_ANTHROPIC_BASE_URL) or None,
        )
