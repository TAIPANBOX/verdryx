"""Shared builders for the findings, fix-verification and human-grade tests.

Runs are built directly rather than through `verdryx eval`, because a stub
adapter scores everything 1.0 and a regression needs scores it did not give.
Times are relative to the moment the suite runs, because a fix is registered at
the real clock and "a run after the fix" has to start after that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from verdryx.models import Baseline, EvalRun, Score
from verdryx.store import Store

AGENT = "agent://acme-bank.example/support/tier1-bot"
NOW = datetime.now(tz=UTC).replace(microsecond=0)
#: Before any fix a test registers.
BEFORE = NOW - timedelta(days=2)
#: After any fix a test registers.
AFTER = NOW + timedelta(days=2)


def make_run(
    values: list[float],
    *,
    started: datetime = BEFORE,
    run_id: str | None = None,
    model: str = "m1",
    version: str | None = None,
    agent: str | None = AGENT,
) -> EvalRun:
    return EvalRun(
        id=run_id or f"run-{started.isoformat()}-{len(values)}-{sum(values):.2f}",
        model=model,
        started_at=started,
        finished_at=started + timedelta(seconds=30),
        scores=[Score(case_id=f"c{i}", value=v) for i, v in enumerate(values)],
        agent_id=agent,
        agent_version=version,
    )


def seed(
    db: Path,
    runs: list[EvalRun],
    *,
    baseline_run: EvalRun | None = None,
    baseline_id: str = "b1",
) -> None:
    """Store `runs` and snapshot `baseline_run` (default: the first) as a baseline."""
    with Store.open(db) as store:
        for r in runs:
            store.save_run(r)
        ref = baseline_run or runs[0]
        store.set_baseline(
            Baseline(
                id=baseline_id,
                eval_run_id=ref.id,
                mean_score=ref.mean_score,
                created_at=ref.started_at,
            )
        )


def good(n: int = 20) -> list[float]:
    return [1.0] * n


def bad(n: int = 20) -> list[float]:
    return [0.0] * n


def regressed_store(
    tmp_path: Path,
    *,
    version_before: str | None = None,
    version_after: str | None = None,
    agent: str | None = AGENT,
    model: str = "m1",
) -> tuple[Path, EvalRun, EvalRun]:
    """A baseline run (all 1.0) and one newer run (all 0.0) of the same agent."""
    db = tmp_path / "store.db"
    baseline = make_run(
        good(), started=BEFORE, run_id="base-run", version=version_before, agent=agent, model=model
    )
    newer = make_run(
        bad(),
        started=BEFORE + timedelta(hours=1),
        run_id="bad-run",
        version=version_after,
        agent=agent,
        model=model,
    )
    seed(db, [baseline, newer], baseline_run=baseline)
    return db, baseline, newer


def trace_file(
    tmp_path: Path,
    outcomes: list[str],
    *,
    start_ms: int,
    agent: str = AGENT,
    key: str = "",
    name: str = "trace.ndjson",
    prefix: str = "t",
) -> Path:
    """An NDJSON file of per-run records, one per outcome, one second apart."""
    path = tmp_path / name
    rows: list[dict[str, Any]] = [
        {
            "run_id": f"{prefix}{i}",
            "agent_id": agent,
            "key_id": key,
            "outcome": outcome,
            "cost_usd": 0.01,
            "last_ts_millis": start_ms + i * 1000,
        }
        for i, outcome in enumerate(outcomes)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
