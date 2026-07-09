"""Cost-per-outcome unit economics from a flat outcome+cost record export.

MVP input: NDJSON or CSV of {"outcome": <str>, "cost_usd": <float>} records,
one row per already-resolved unit of work (a run or a case), each tagged
with its final outcome -- for example a `tokenfuse outcomes --json` export
(after converting its cost_microusd column to cost_usd by dividing by
1_000_000) or a hand-rolled export from agent-event / trace outcome tags.

Reading Parquet traces directly, and reproducing tokenfuse-core's full
"last non-empty x-fuse-outcome tag per run wins" aggregation (see
tokenfuse's crates/core/src/outcomes.rs) inside verdryx itself, is a
documented later enhancement. This module assumes that aggregation has
already happened upstream and each input record is one final, resolved
outcome.
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


def read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line, skipping blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV with an `outcome` column and a `cost_usd` column."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Dispatch on file extension: .ndjson/.jsonl -> NDJSON, .csv -> CSV.

    Raises:
        ValueError: If the extension is not recognized.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _NDJSON_SUFFIXES:
        return read_ndjson(path)
    if suffix in _CSV_SUFFIXES:
        return read_csv(path)
    raise ValueError(
        f"don't know how to read {path!r}: expected one of "
        f"{sorted(_NDJSON_SUFFIXES | _CSV_SUFFIXES)}"
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
