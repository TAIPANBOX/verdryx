"""Cost-per-outcome unit economics from a flat outcome+cost record export.

Three input shapes, all funneling into the same {"outcome": <str>,
"cost_usd": <float>} record shape that `cost_per_outcome` consumes:

- NDJSON or CSV of already-resolved records, one row per unit of work (a
  run or a case), each tagged with its final outcome -- for example a
  `tokenfuse outcomes --json` export (after converting its cost_microusd
  column to cost_usd by dividing by 1_000_000) or a hand-rolled export from
  agent-event / trace outcome tags. See `read_ndjson`/`read_csv`.
- A tokenfuse Parquet trace (a single `*.parquet` file, or a directory of
  them, e.g. TOKENFUSE_DATA_DIR) read directly via `read_parquet`.

`read_parquet` reproduces tokenfuse-core's "last non-empty x-fuse-outcome tag
per run wins" reduction (see tokenfuse's crates/core/src/outcomes.rs
`compute_outcomes`): a run whose agent tags more than one of its calls (e.g.
`escalated`, then later `case_resolved` once the situation resolves) is
reduced to its LAST non-empty tag in `step` order -- the per-run sequence
counter, not `ts_millis`, because a fast run can share a millisecond but
never a step -- so it is counted once, under its final outcome, instead of
once per tagged call. See `read_parquet`'s docstring for what is and is not
covered.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from verdryx.models import CostPerOutcomeReport, OutcomeCost

#: Bucket name for the report's overall (all outcomes pooled) row.
OVERALL = "overall"

_NDJSON_SUFFIXES = {".ndjson", ".jsonl"}
_CSV_SUFFIXES = {".csv"}
_PARQUET_SUFFIXES = {".parquet"}

#: Columns verdryx reads from a tokenfuse Parquet trace -- see tokenfuse's
#: crates/gateway/src/sink.rs (CallRecord / ParquetSink::schema): the raw
#: per-call outcome tag, the settled cost in microdollars, the run id each
#: call belongs to, and the per-run sequence counter used to order calls
#: within a run (see `_reduce_tagged_rows`).
_PARQUET_OUTCOME_COLUMN = "outcome"
_PARQUET_COST_COLUMN = "cost_microusd"
_PARQUET_RUN_ID_COLUMN = "run_id"
_PARQUET_STEP_COLUMN = "step"


def read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line, skipping blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV with an `outcome` column and a `cost_usd` column."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pyarrow_parquet() -> Any:
    """Import pyarrow.parquet lazily so it stays an optional dependency.

    Mirrors AnthropicAdapter._get_client's lazy-import pattern in
    graders.py: the module-level `import verdryx.costper` must not require
    pyarrow, only calling `read_parquet` (or `load_records` on a `.parquet`
    path / directory) does.

    Raises:
        ImportError: If pyarrow is not installed, with a pip-install hint.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "reading Parquet traces requires pyarrow. Install with: pip install 'verdryx[traces]'"
        ) from exc
    return pq


def _parquet_file_paths(path: str | Path) -> list[Path]:
    """A single `.parquet` file, or every `*.parquet` file directly inside a
    directory, sorted for deterministic order.

    tokenfuse's ParquetSink writes rotating `calls-NNNNNNNN.parquet`
    segments into TOKENFUSE_DATA_DIR -- point `read_parquet` at that
    directory to read the whole trace.
    """
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.parquet"))
    return [p]


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read tokenfuse Parquet trace file(s) into {outcome, cost_usd} records,
    one per resolved RUN.

    Args:
        path: A single `.parquet` file, or a directory containing one or
            more `*.parquet` files (see `_parquet_file_paths`).

    Reads four columns from tokenfuse's trace schema (tokenfuse's
    crates/gateway/src/sink.rs `CallRecord` / `ParquetSink`): the `outcome`
    tag (Utf8), `cost_microusd` (Int64, converted here to `cost_usd` by
    dividing by 1_000_000), `run_id` (Utf8), and `step` (UInt32 -- the
    per-run sequence counter). `outcome` is declared NULLABLE in tokenfuse's
    own read schema (added in schema phase P4; a hand-built or pre-P4 file
    may lack it entirely), so a file missing any of these columns, or a row
    with a null cell, is tolerated: a missing/null `outcome` or
    `cost_microusd` reads as "" / 0 same as always; a missing/null `run_id`
    means that row cannot be correlated with any other (see
    `_reduce_tagged_rows`) and is kept as its own independent record -- which
    is also exactly what happens for every caller reading a file with no
    `run_id` column at all (e.g. a hand-built fixture predating that column).

    Rows with no outcome tag ("", the common case -- tokenfuse expects an
    agent to tag only its run's FINAL call, so most rows in a raw trace
    carry no tag at all) are dropped before reduction. An empty tag is not a
    resolved outcome; feeding it through would otherwise create a misleading
    "" bucket in the report instead of a real outcome tag.

    A run whose agent tags more than one of its calls (e.g. `escalated`,
    then later `case_resolved` once the situation resolves) is reduced to
    ONE record: its LAST non-empty outcome tag in `step` order, with
    cost_usd summed across the tagged calls folded into it -- see
    `_reduce_tagged_rows` and this module's docstring. Prefer an
    already-reduced export (e.g. `tokenfuse outcomes --json`) via
    `read_ndjson`/`read_csv` if you would rather not depend on this
    reduction.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    pq = _pyarrow_parquet()
    tagged_rows: list[tuple[str | None, int, str, float]] = []
    fallback_step = 0
    for file_path in _parquet_file_paths(path):
        table = pq.read_table(file_path)
        columns = set(table.column_names)
        outcomes = (
            table.column(_PARQUET_OUTCOME_COLUMN).to_pylist()
            if _PARQUET_OUTCOME_COLUMN in columns
            else None
        )
        costs = (
            table.column(_PARQUET_COST_COLUMN).to_pylist()
            if _PARQUET_COST_COLUMN in columns
            else None
        )
        run_ids = (
            table.column(_PARQUET_RUN_ID_COLUMN).to_pylist()
            if _PARQUET_RUN_ID_COLUMN in columns
            else None
        )
        steps = (
            table.column(_PARQUET_STEP_COLUMN).to_pylist()
            if _PARQUET_STEP_COLUMN in columns
            else None
        )
        for i in range(table.num_rows):
            outcome = (outcomes[i] if outcomes is not None else None) or ""
            if not outcome:
                fallback_step += 1
                continue
            cost_microusd = (costs[i] if costs is not None else None) or 0
            run_id: str | None = run_ids[i] if run_ids is not None else None
            step: int | None = steps[i] if steps is not None else None
            if step is None:
                # No `step` column (or a null cell): fall back to encounter
                # order across every tagged row read so far, so ordering
                # within a run_id stays deterministic instead of undefined.
                step = fallback_step
            tagged_rows.append((run_id, step, outcome, cost_microusd / 1_000_000))
            fallback_step += 1
    return _reduce_tagged_rows(tagged_rows)


def _reduce_tagged_rows(
    tagged_rows: list[tuple[str | None, int, str, float]],
) -> list[dict[str, Any]]:
    """Reduce (run_id, step, outcome, cost_usd) tagged-call rows to one
    {outcome, cost_usd} record per run_id.

    Mirrors tokenfuse-core's own reduction (crates/core/src/outcomes.rs
    `compute_outcomes`): sort each run's tagged calls by `step` and let the
    LAST non-empty tag win, then sum cost_usd across the tagged calls folded
    into that run's winning tag.

    A row whose run_id is None (the `run_id` column was missing from its
    file entirely, or this particular cell was null) cannot be correlated
    with any other row, so it is kept as its own independent record, in its
    original encounter order -- this is also what makes every caller reading
    a file with no `run_id` column at all see no change in behavior:
    nothing to reduce, one record per tagged call, same as before this
    reduction existed.
    """
    with_run_id: list[tuple[str, int, str, float]] = []
    for run_id, step, outcome, cost_usd in tagged_rows:
        if run_id is not None:
            with_run_id.append((run_id, step, outcome, cost_usd))

    # Sort by (run_id, step), mirroring tokenfuse's
    # `ordered.sort_by(|a, b| a.run_id.cmp(&b.run_id).then(a.step.cmp(&b.step)))`,
    # then scan in that order so each run's LAST non-empty tag is whatever a
    # later (higher-step) row most recently overwrote it with.
    ordered = sorted(with_run_id, key=lambda row: (row[0], row[1]))
    winner: dict[str, str] = {}
    total_cost: dict[str, float] = {}
    for run_id, _step, outcome, cost_usd in ordered:
        winner[run_id] = outcome
        total_cost[run_id] = total_cost.get(run_id, 0.0) + cost_usd

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for run_id, _step, outcome, cost_usd in tagged_rows:
        if run_id is None:
            records.append({"outcome": outcome, "cost_usd": cost_usd})
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        records.append({"outcome": winner[run_id], "cost_usd": total_cost[run_id]})
    return records


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Dispatch on shape/extension: a directory or a `.parquet` file ->
    Parquet (see `read_parquet`), `.ndjson`/`.jsonl` -> NDJSON, `.csv` ->
    CSV.

    Raises:
        ValueError: If `path` is a file with an unrecognized extension.
    """
    p = Path(path)
    if p.is_dir():
        return read_parquet(p)
    suffix = p.suffix.lower()
    if suffix in _NDJSON_SUFFIXES:
        return read_ndjson(p)
    if suffix in _CSV_SUFFIXES:
        return read_csv(p)
    if suffix in _PARQUET_SUFFIXES:
        return read_parquet(p)
    raise ValueError(
        f"don't know how to read {path!r}: expected a directory of .parquet files, or one of "
        f"{sorted(_NDJSON_SUFFIXES | _CSV_SUFFIXES | _PARQUET_SUFFIXES)}"
    )


def _summarize(outcome: str, costs: list[float]) -> OutcomeCost:
    count = len(costs)
    total = sum(costs)
    mean = total / count if count else 0.0
    return OutcomeCost(outcome=outcome, count=count, total_cost_usd=total, mean_cost_usd=mean)


def cost_per_outcome(records: Iterable[Mapping[str, Any]]) -> CostPerOutcomeReport:
    """Bucket records by their `outcome` field and total/average `cost_usd`.

    Args:
        records: An iterable of mappings, each with an `outcome` (str) key
            and a `cost_usd` (float-coercible) key. CSV rows (str-only
            values from csv.DictReader) are coerced automatically.

    Returns:
        A CostPerOutcomeReport with one OutcomeCost per outcome tag that
        appeared in `records`, plus an `overall` OutcomeCost pooling all of
        them. Use `.resolved` / `.escalated` / `.abandoned` for the three
        outcome tags Verdryx ships as defaults (models.OUTCOME_RESOLVED
        etc.), or `.get(tag)` for a custom tag.

    Raises:
        KeyError: If a record is missing `outcome` or `cost_usd`.
        ValueError: If a record's `cost_usd` cannot be parsed as a float.
    """
    buckets: dict[str, list[float]] = {}
    for rec in records:
        outcome = str(rec["outcome"])
        cost = float(rec["cost_usd"])
        buckets.setdefault(outcome, []).append(cost)

    by_outcome = {outcome: _summarize(outcome, costs) for outcome, costs in buckets.items()}
    all_costs = [cost for costs in buckets.values() for cost in costs]
    overall = _summarize(OVERALL, all_costs)
    return CostPerOutcomeReport(by_outcome=by_outcome, overall=overall)
