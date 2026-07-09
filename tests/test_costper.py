"""Tests for verdryx.costper."""

from __future__ import annotations

from pathlib import Path

import pytest

from verdryx.costper import cost_per_outcome, load_records, read_csv, read_ndjson
from verdryx.models import OUTCOME_RESOLVED

_FIXTURES = Path(__file__).parent / "fixtures"
_NDJSON = _FIXTURES / "outcomes_sample.ndjson"
_CSV = _FIXTURES / "outcomes_sample.csv"


# ------------------------------------------------------------------
# Readers
# ------------------------------------------------------------------


def test_read_ndjson_parses_each_line() -> None:
    records = read_ndjson(_NDJSON)
    assert len(records) == 5
    assert records[0] == {"outcome": OUTCOME_RESOLVED, "cost_usd": 0.10}


def test_read_ndjson_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "with_blanks.ndjson"
    path.write_text('{"outcome": "case_resolved", "cost_usd": 1.0}\n\n\n')
    assert read_ndjson(path) == [{"outcome": "case_resolved", "cost_usd": 1.0}]


def test_read_csv_parses_rows() -> None:
    records = read_csv(_CSV)
    assert len(records) == 3
    assert records[0]["outcome"] == OUTCOME_RESOLVED
    assert records[0]["cost_usd"] == "0.10"  # CSV values are strings until coerced


def test_load_records_dispatches_ndjson_by_extension() -> None:
    assert load_records(_NDJSON) == read_ndjson(_NDJSON)


def test_load_records_dispatches_jsonl_extension(tmp_path) -> None:
    jsonl_path = tmp_path / "outcomes.jsonl"
    jsonl_path.write_text(_NDJSON.read_text())
    assert load_records(jsonl_path) == read_ndjson(_NDJSON)


def test_load_records_dispatches_csv_by_extension() -> None:
    assert load_records(_CSV) == read_csv(_CSV)


def test_load_records_unknown_extension_raises(tmp_path) -> None:
    bad = tmp_path / "outcomes.txt"
    bad.write_text("outcome,cost_usd\n")
    with pytest.raises(ValueError, match="don't know how to read"):
        load_records(bad)


# ------------------------------------------------------------------
# cost_per_outcome math
# ------------------------------------------------------------------


def test_cost_per_outcome_math_on_ndjson_fixture() -> None:
    records = read_ndjson(_NDJSON)
    report = cost_per_outcome(records)

    assert report.resolved is not None
    assert report.resolved.count == 2
    assert report.resolved.total_cost_usd == pytest.approx(0.30)
    assert report.resolved.mean_cost_usd == pytest.approx(0.15)

    assert report.escalated is not None
    assert report.escalated.count == 1
    assert report.escalated.total_cost_usd == pytest.approx(0.50)
    assert report.escalated.mean_cost_usd == pytest.approx(0.50)

    assert report.abandoned is not None
    assert report.abandoned.count == 2
    assert report.abandoned.total_cost_usd == pytest.approx(0.08)
    assert report.abandoned.mean_cost_usd == pytest.approx(0.04)

    assert report.overall.count == 5
    assert report.overall.total_cost_usd == pytest.approx(0.88)
    assert report.overall.mean_cost_usd == pytest.approx(0.176)


def test_cost_per_outcome_math_on_csv_fixture_coerces_string_costs() -> None:
    records = read_csv(_CSV)
    report = cost_per_outcome(records)
    assert report.overall.count == 3
    assert report.overall.total_cost_usd == pytest.approx(0.65)
    assert report.resolved is not None
    assert report.resolved.total_cost_usd == pytest.approx(0.10)


def test_cost_per_outcome_empty_input() -> None:
    report = cost_per_outcome([])
    assert report.by_outcome == {}
    assert report.overall.count == 0
    assert report.overall.total_cost_usd == 0.0
    assert report.overall.mean_cost_usd == 0.0


def test_cost_per_outcome_missing_outcome_key_raises() -> None:
    with pytest.raises(KeyError):
        cost_per_outcome([{"cost_usd": 1.0}])


def test_cost_per_outcome_missing_cost_key_raises() -> None:
    with pytest.raises(KeyError):
        cost_per_outcome([{"outcome": "case_resolved"}])


def test_cost_per_outcome_unparseable_cost_raises_value_error() -> None:
    with pytest.raises(ValueError):
        cost_per_outcome([{"outcome": "case_resolved", "cost_usd": "not-a-number"}])


def test_cost_per_outcome_custom_outcome_tag_reachable_via_get() -> None:
    report = cost_per_outcome([{"outcome": "custom_tag", "cost_usd": 1.0}])
    custom = report.get("custom_tag")
    assert custom is not None
    assert custom.count == 1
    assert report.resolved is None
    assert report.escalated is None
    assert report.abandoned is None
