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

`read_parquet` reproduces tokenfuse-core's per-run reduction (see
tokenfuse's crates/core/src/outcomes.rs `compute_outcomes`) for a
run_id-bearing trace:

- A run whose agent tags more than one of its calls (e.g. `escalated`, then
  later `case_resolved` once the situation resolves) is reduced to its LAST
  non-empty tag in `step` order -- the per-run sequence counter, not
  `ts_millis`, because a fast run can share a millisecond but never a step
  -- so it is counted once, under its final outcome, instead of once per
  tagged call.
- Every call belonging to a run folds into that run's bucket, not only its
  tagged calls: an untagged intermediate call's cost is not dropped, and a
  run that never gets tagged at all still produces a record, under the
  `UNTAGGED` ("(untagged)") label, instead of vanishing from the report.
- A Breaker-blocked call (`decision` one of tokenfuse's nine block
  reasons) is still counted, but excluded from cost -- its `cost_microusd`
  is an avoided estimate, never a real settled charge.

See `read_parquet`'s docstring for what is and is not covered, including
the narrower behavior kept for files with no `run_id` column at all.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from verdryx.models import CostPerOutcomeReport, OutcomeCost

#: Bucket name for the report's overall (all outcomes pooled) row.
OVERALL = "overall"

#: Outcome label read_parquet emits for a run whose calls all carry a
#: run_id but none of them ever set a non-empty outcome tag -- mirrors
#: tokenfuse's own "(untagged)" rendering of compute_outcomes's
#: `outcome: None` row (tokenfuse's crates/core/src/outcomes.rs).
UNTAGGED = "(untagged)"

_NDJSON_SUFFIXES = {".ndjson", ".jsonl"}
_CSV_SUFFIXES = {".csv"}
_PARQUET_SUFFIXES = {".parquet"}

#: Columns verdryx reads from a tokenfuse Parquet trace -- see tokenfuse's
#: crates/gateway/src/sink.rs (CallRecord / ParquetSink::schema): the raw
#: per-call outcome tag, the settled cost in microdollars, the run id each
#: call belongs to, the per-run sequence counter used to order calls within
#: a run, and the wire decision string used to detect a blocked call (see
#: `_reduce_call_rows`).
_PARQUET_OUTCOME_COLUMN = "outcome"
_PARQUET_COST_COLUMN = "cost_microusd"
_PARQUET_RUN_ID_COLUMN = "run_id"
_PARQUET_STEP_COLUMN = "step"
_PARQUET_DECISION_COLUMN = "decision"
_PARQUET_AGENT_ID_COLUMN = "agent_id"
_PARQUET_KEY_ID_COLUMN = "key_id"
_PARQUET_TS_COLUMN = "ts_millis"


class _CallRow(NamedTuple):
    """One trace row, named rather than positional.

    It was a five-wide tuple until 2026-08-26, when the SLO reader needed
    identity and time from the same rows. Widening a positional tuple to eight
    is how a reduction starts reading the wrong field, and the alternative,
    a second reader in `slo.py` doing its own reduction, is the drift this
    repository was already bitten by once: verdryx held seven of tokenfuse's
    block reasons while tokenfuse had nine, and for eleven days avoided
    estimates were counted as real money. One reduction, named fields.
    """

    run_id: str | None
    step: int
    outcome: str
    cost_usd: float
    decision: str
    agent_id: str
    key_id: str
    ts_millis: int | None


#: The nine Breaker block-decision wire strings (tokenfuse's
#: crates/core/src/outcomes.rs BLOCKED_DECISIONS, read off
#: BreakerReason::as_wire_str() in crates/core/src/breaker.rs: the original
#: seven plus the identity-map pair, unit_budget_exceeded and
#: identity_mismatch, added in tokenfuse commit 833d6aa on 2026-07-23). A
#: blocked call's cost_microusd holds an avoided estimate, never a real
#: settled charge, so `_reduce_call_rows` excludes these rows from cost the
#: same way tokenfuse-core's `compute_outcomes` does, while still counting
#: them as calls.
_BLOCKED_DECISIONS = frozenset(
    {
        "budget_exceeded",
        "policy_violation",
        "loop_detected",
        "killed",
        "wasm_policy",
        "taint_blocked",
        "dlp_blocked",
        "unit_budget_exceeded",
        "identity_mismatch",
    }
)


#: The two refusals tokenfuse writes to the trace that are NOT among the nine
#: above, and the reason they need naming separately.
#:
#: `wardryx_deny` and `wardryx_hold` are written by the gateway's policy hook
#: (tokenfuse's crates/gateway/src/proxy.rs) and both return HTTP 403, but
#: neither is a `BreakerReason`, so neither appears in tokenfuse-core's
#: `BLOCKED_DECISIONS` and `_is_blocked_decision` is False for both. That is
#: correct for COST, which is what the nine are about: a blocked call's
#: `cost_microusd` is an avoided estimate and these two rows carry zero
#: anyway. It is wrong for any question of the form "did anything stop this
#: run", which is exactly what the containment SLI asks, and a reliability
#: figure built on the nine alone would report a fleet as unrestrained while
#: the policy plane was refusing half its calls.
#:
#: Kept as its own frozenset rather than folded into `_BLOCKED_DECISIONS`,
#: because widening that set would silently change what `cost_per_outcome`
#: excludes from spend, which is a different question with a different right
#: answer. Verdryx reads traces and does not get to redefine them
#: (invariant 4).
_REFUSED_DECISIONS = frozenset({"wardryx_deny", "wardryx_hold"})


def _is_blocked_decision(decision: str) -> bool:
    """Whether `decision` is one of the nine Breaker block reasons."""
    return decision in _BLOCKED_DECISIONS


def is_refused_decision(decision: str) -> bool:
    """Whether `decision` means SOMETHING refused this call, by any plane.

    The nine Breaker reasons plus the two Wardryx ones. This is the predicate
    a reliability question wants; `_is_blocked_decision` is the one a cost
    question wants, and they are deliberately different sets.
    """
    return decision in _BLOCKED_DECISIONS or decision in _REFUSED_DECISIONS


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

    Reads five columns from tokenfuse's trace schema (tokenfuse's
    crates/gateway/src/sink.rs `CallRecord` / `ParquetSink`): the `outcome`
    tag (Utf8), `cost_microusd` (Int64, converted here to `cost_usd` by
    dividing by 1_000_000), `run_id` (Utf8), `step` (UInt32 -- the per-run
    sequence counter), and `decision` (Utf8 -- the wire decision string used
    to detect a Breaker-blocked call). `outcome` is declared NULLABLE in
    tokenfuse's own read schema (added in schema phase P4; a hand-built or
    pre-P4 file may lack it entirely), so a file missing any of these
    columns, or a row with a null cell, is tolerated: a missing/null
    `outcome`, `cost_microusd`, or `decision` reads as "" / 0 / "" same as
    always; a missing/null `run_id` means that row cannot be correlated with
    any other (see `_reduce_call_rows`) and, if it also has no outcome tag
    of its own, is dropped -- which is also exactly what happens for every
    caller reading a file with no `run_id` column at all (e.g. a hand-built
    fixture predating that column): nothing to fold an untagged call's cost
    into, so only its tagged calls are kept, one record each.

    A run whose agent tags more than one of its calls (e.g. `escalated`,
    then later `case_resolved` once the situation resolves) is reduced to
    ONE record: its LAST non-empty outcome tag in `step` order. Every call
    belonging to that run -- tagged or not -- folds its cost into that one
    record, and a run that is never tagged at all still produces a record,
    under the `UNTAGGED` label, rather than vanishing from the report. A
    Breaker-blocked call (see `_is_blocked_decision`) is still counted, but
    its cost is excluded, since a blocked call's `cost_microusd` is an
    avoided estimate, never a real settled charge. See `_reduce_call_rows`
    and this module's docstring. Prefer an already-reduced export (e.g.
    `tokenfuse outcomes --json`) via `read_ndjson`/`read_csv` if you would
    rather not depend on this reduction.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    return [_cost_projection(r) for r in _read_parquet_records(path)]


def _read_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    """The shared body: read every column, reduce once."""
    pq = _pyarrow_parquet()
    call_rows: list[_CallRow] = []
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
        decisions = (
            table.column(_PARQUET_DECISION_COLUMN).to_pylist()
            if _PARQUET_DECISION_COLUMN in columns
            else None
        )
        agent_ids = (
            table.column(_PARQUET_AGENT_ID_COLUMN).to_pylist()
            if _PARQUET_AGENT_ID_COLUMN in columns
            else None
        )
        key_ids = (
            table.column(_PARQUET_KEY_ID_COLUMN).to_pylist()
            if _PARQUET_KEY_ID_COLUMN in columns
            else None
        )
        timestamps = (
            table.column(_PARQUET_TS_COLUMN).to_pylist() if _PARQUET_TS_COLUMN in columns else None
        )
        for i in range(table.num_rows):
            outcome = (outcomes[i] if outcomes is not None else None) or ""
            run_id: str | None = run_ids[i] if run_ids is not None else None
            if run_id is None and not outcome:
                # Nothing to fold this call into (no run) and no tag of its
                # own: there is no record to produce from it.
                fallback_step += 1
                continue
            cost_microusd = (costs[i] if costs is not None else None) or 0
            decision = (decisions[i] if decisions is not None else None) or ""
            step: int | None = steps[i] if steps is not None else None
            if step is None:
                # No `step` column (or a null cell): fall back to encounter
                # order across every row read so far, so ordering within a
                # run_id stays deterministic instead of undefined.
                step = fallback_step
            call_rows.append(
                _CallRow(
                    run_id=run_id,
                    step=step,
                    outcome=outcome,
                    cost_usd=cost_microusd / 1_000_000,
                    decision=decision,
                    # Identity and time are read where present and default to
                    # the same empty sentinel tokenfuse writes, never to a
                    # guess: a fabricated subject is worse than an absent one,
                    # and `slo.py` counts the absences rather than hiding them.
                    agent_id=(agent_ids[i] if agent_ids is not None else None) or "",
                    key_id=(key_ids[i] if key_ids is not None else None) or "",
                    ts_millis=(timestamps[i] if timestamps is not None else None),
                )
            )
            fallback_step += 1
    return _reduce_call_rows(call_rows)


def _cost_projection(record: dict[str, Any]) -> dict[str, Any]:
    """`read_parquet`'s documented shape, projected off the full record.

    `read_parquet` says it produces `{outcome, cost_usd}` records and callers
    compare whole dicts against that, so the reliability fields the SLO plane
    needs are NOT bolted onto it: widening a documented return shape breaks
    every consumer doing an equality check, which is how this was found
    (thirteen of this repository's own tests, each comparing a whole dict).
    `read_parquet_runs` returns the full record for callers that want it.
    """
    return {"outcome": record["outcome"], "cost_usd": record["cost_usd"]}


def read_parquet_runs(path: str | Path) -> list[dict[str, Any]]:
    """Every per-run record `read_parquet` reduces, with nothing projected away.

    The same one reduction, so a cost figure and a reliability figure can
    never describe two different scans of the same run. Adds `run_id`,
    `agent_id`, `key_id`, `calls`, `refused_calls` and the first/last
    `ts_millis` the run was seen at.

    `refused_calls` counts the nine Breaker reasons AND the two Wardryx ones
    (`is_refused_decision`), which is deliberately a wider set than the one
    `cost_usd` excludes: "what did this cost" and "did anything stop this run"
    are different questions and tokenfuse answers them with different
    vocabularies.
    """
    return _read_parquet_records(path)


def _reduce_call_rows(
    call_rows: list[_CallRow],
) -> list[dict[str, Any]]:
    """Reduce (run_id, step, outcome, cost_usd, decision) call rows to one
    {outcome, cost_usd} record per run_id.

    Mirrors tokenfuse-core's own reduction (crates/core/src/outcomes.rs
    `compute_outcomes`): sort each run's calls by `step`, let the LAST
    non-empty tag win (a run with no non-empty tag at all resolves to
    `UNTAGGED`, not dropped), then sum cost_usd across every one of that
    run's calls -- not only the tagged ones -- excluding any call whose
    `decision` is one of the nine Breaker block reasons.

    A row whose run_id is None (the `run_id` column was missing from its
    file entirely, or this particular cell was null) cannot be correlated
    with any other row, so it is kept as its own independent record, in its
    original encounter order, with the same blocked-decision cost exclusion
    applied -- this is also what makes every caller reading a file with no
    `run_id` column at all see the pre-reduction shape: one record per
    tagged call (an untagged, run_id-less row has nothing to fold into and
    never reaches this function -- see `read_parquet`).
    """
    with_run_id: list[_CallRow] = []
    without_run_id: list[_CallRow] = []
    for row in call_rows:
        if row.run_id is not None:
            with_run_id.append(row)
        else:
            without_run_id.append(row)

    # Sort by (run_id, step), mirroring tokenfuse's
    # `ordered.sort_by(|a, b| a.run_id.cmp(&b.run_id).then(a.step.cmp(&b.step)))`,
    # then scan in that order so each run's LAST non-empty tag is whatever a
    # later (higher-step) row most recently overwrote it with, and each
    # run's total cost sums every non-blocked call regardless of its tag.
    ordered = sorted(with_run_id, key=lambda row: (row.run_id or "", row.step))
    winner: dict[str, str] = {}
    total_cost: dict[str, float] = {}
    # Reliability facts, accumulated in the SAME pass and by the same rule the
    # cost total uses, so a record can never describe one scan of a run and a
    # different scan of the same run.
    calls: dict[str, int] = {}
    refused: dict[str, int] = {}
    identity: dict[str, tuple[str, str]] = {}
    first_ts: dict[str, int | None] = {}
    last_ts: dict[str, int | None] = {}
    for row in ordered:
        run_id = row.run_id or ""
        if row.outcome:
            winner[run_id] = row.outcome
        total_cost[run_id] = total_cost.get(run_id, 0.0) + (
            0.0 if _is_blocked_decision(row.decision) else row.cost_usd
        )
        calls[run_id] = calls.get(run_id, 0) + 1
        if is_refused_decision(row.decision):
            refused[run_id] = refused.get(run_id, 0) + 1
        # The LAST non-empty identity wins, matching how the outcome tag is
        # resolved, so a run whose first call arrived before the identity map
        # was consulted is described by what the gateway finally knew. An
        # empty value never overwrites a known one, because "" is tokenfuse's
        # unset sentinel rather than a value.
        prev_agent, prev_key = identity.get(run_id, ("", ""))
        identity[run_id] = (row.agent_id or prev_agent, row.key_id or prev_key)
        if row.ts_millis is not None:
            if first_ts.get(run_id) is None:
                first_ts[run_id] = row.ts_millis
            last_ts[run_id] = row.ts_millis

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in with_run_id:
        run_id = row.run_id or ""
        if run_id in seen:
            continue
        seen.add(run_id)
        agent_id, key_id = identity.get(run_id, ("", ""))
        records.append(
            {
                "outcome": winner.get(run_id, UNTAGGED),
                "cost_usd": total_cost[run_id],
                "run_id": run_id,
                "agent_id": agent_id,
                "key_id": key_id,
                "calls": calls.get(run_id, 0),
                "refused_calls": refused.get(run_id, 0),
                "first_ts_millis": first_ts.get(run_id),
                "last_ts_millis": last_ts.get(run_id),
            }
        )

    for row in without_run_id:
        records.append(
            {
                "outcome": row.outcome,
                "cost_usd": 0.0 if _is_blocked_decision(row.decision) else row.cost_usd,
                "run_id": "",
                "agent_id": row.agent_id,
                "key_id": row.key_id,
                "calls": 1,
                "refused_calls": 1 if is_refused_decision(row.decision) else 0,
                "first_ts_millis": row.ts_millis,
                "last_ts_millis": row.ts_millis,
            }
        )
    return records


def load_run_records(path: str | Path) -> list[dict[str, Any]]:
    """`load_records`, but keeping every field the reduction produced.

    Same dispatch on shape and extension. NDJSON and CSV pass straight
    through, because a hand-written file carries whatever its author put in
    it and this function must not invent the fields a Parquet trace would
    have had.
    """
    p = Path(path)
    if p.is_dir():
        return read_parquet_runs(p)
    suffix = p.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        return read_parquet_runs(p)
    return load_records(p)


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
