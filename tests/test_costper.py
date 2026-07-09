"""Tests for verdryx.costper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from verdryx.costper import cost_per_outcome, load_records, read_csv, read_ndjson, read_parquet
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


# ------------------------------------------------------------------
# read_parquet / load_records(.parquet) -- requires the optional `pyarrow`
# dependency (the `traces` extra). Each test below self-skips via the
# `pyarrow_and_parquet` fixture (tests/conftest.py) if it is not installed.
# ------------------------------------------------------------------


def test_read_parquet_single_file_extracts_outcome_and_cost_usd(
    tmp_path, pyarrow_and_parquet
) -> None:
    pa, pq = pyarrow_and_parquet
    table = pa.table(
        {"outcome": ["case_resolved", "escalated"], "cost_microusd": [150_000, 500_000]}
    )
    path = tmp_path / "calls-00000000.parquet"
    pq.write_table(table, path)

    assert read_parquet(path) == [
        {"outcome": "case_resolved", "cost_usd": 0.15},
        {"outcome": "escalated", "cost_usd": 0.5},
    ]


def test_read_parquet_drops_rows_with_no_outcome_tag(tmp_path, pyarrow_and_parquet) -> None:
    """Most rows in a raw trace carry no tag (tokenfuse only expects an
    agent to tag its run's final call) -- those rows must not become a
    spurious "" bucket in the report."""
    pa, pq = pyarrow_and_parquet
    table = pa.table(
        {"outcome": ["", "case_resolved", ""], "cost_microusd": [10_000, 100_000, 20_000]}
    )
    path = tmp_path / "calls-00000000.parquet"
    pq.write_table(table, path)

    assert read_parquet(path) == [{"outcome": "case_resolved", "cost_usd": 0.10}]


def test_read_parquet_tolerates_missing_outcome_column(tmp_path, pyarrow_and_parquet) -> None:
    """A pre-P4 tokenfuse trace file predates the `outcome` column
    entirely -- every row reads as untagged and is dropped, not an error."""
    pa, pq = pyarrow_and_parquet
    table = pa.table({"cost_microusd": [10_000, 20_000]})
    path = tmp_path / "calls-00000000.parquet"
    pq.write_table(table, path)

    assert read_parquet(path) == []


def test_read_parquet_tolerates_missing_cost_column(tmp_path, pyarrow_and_parquet) -> None:
    pa, pq = pyarrow_and_parquet
    table = pa.table({"outcome": ["case_resolved", "abandoned"]})
    path = tmp_path / "calls-00000000.parquet"
    pq.write_table(table, path)

    assert read_parquet(path) == [
        {"outcome": "case_resolved", "cost_usd": 0.0},
        {"outcome": "abandoned", "cost_usd": 0.0},
    ]


def test_read_parquet_reads_every_file_in_a_directory(tmp_path, pyarrow_and_parquet) -> None:
    pa, pq = pyarrow_and_parquet
    pq.write_table(
        pa.table({"outcome": ["case_resolved"], "cost_microusd": [100_000]}),
        tmp_path / "calls-00000000.parquet",
    )
    pq.write_table(
        pa.table({"outcome": ["escalated"], "cost_microusd": [500_000]}),
        tmp_path / "calls-00000001.parquet",
    )
    # A non-parquet file alongside the trace segments must be ignored.
    (tmp_path / "notes.txt").write_text("not a trace")

    records = read_parquet(tmp_path)
    assert {(r["outcome"], r["cost_usd"]) for r in records} == {
        ("case_resolved", 0.10),
        ("escalated", 0.50),
    }


def test_read_parquet_empty_directory_returns_empty_list(tmp_path, pyarrow_and_parquet) -> None:
    _pa, _pq = pyarrow_and_parquet
    assert read_parquet(tmp_path) == []


def test_read_parquet_round_trip_through_cost_per_outcome(tmp_path, pyarrow_and_parquet) -> None:
    """The scenario the `traces` extra exists for: a tokenfuse Parquet
    trace, read directly and fed through the same cost_per_outcome
    aggregation used for NDJSON/CSV input."""
    pa, pq = pyarrow_and_parquet
    table = pa.table(
        {
            "outcome": [
                "",  # mid-run call, not yet (or never) tagged -- dropped
                "case_resolved",
                "case_resolved",
                "escalated",
                "abandoned",
            ],
            "cost_microusd": [999_000, 100_000, 200_000, 500_000, 80_000],
        }
    )
    path = tmp_path / "calls-00000000.parquet"
    pq.write_table(table, path)

    report = cost_per_outcome(read_parquet(path))

    assert report.resolved is not None
    assert report.resolved.count == 2
    assert report.resolved.total_cost_usd == pytest.approx(0.30)
    assert report.resolved.mean_cost_usd == pytest.approx(0.15)

    assert report.escalated is not None
    assert report.escalated.total_cost_usd == pytest.approx(0.50)

    assert report.abandoned is not None
    assert report.abandoned.total_cost_usd == pytest.approx(0.08)

    # The untagged row is excluded from both the buckets and the overall
    # pool: it is not a resolved outcome.
    assert report.overall.count == 4
    assert report.overall.total_cost_usd == pytest.approx(0.88)


def test_read_parquet_missing_pyarrow_raises_clear_import_error(tmp_path) -> None:
    """Same technique as test_graders.py's
    test_anthropic_adapter_import_error_without_sdk: `None` in sys.modules
    makes the lazy `import` raise ImportError regardless of whether
    pyarrow is actually installed in this environment."""
    with (
        patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}),
        pytest.raises(ImportError, match=r"verdryx\[traces\]"),
    ):
        read_parquet(tmp_path / "does-not-matter.parquet")


# ------------------------------------------------------------------
# load_records dispatch for Parquet
# ------------------------------------------------------------------


def test_load_records_dispatches_parquet_file_by_extension(tmp_path, pyarrow_and_parquet) -> None:
    pa, pq = pyarrow_and_parquet
    table = pa.table({"outcome": ["case_resolved"], "cost_microusd": [100_000]})
    path = tmp_path / "trace.parquet"
    pq.write_table(table, path)

    assert load_records(path) == read_parquet(path)


def test_load_records_dispatches_directory_to_parquet(tmp_path, pyarrow_and_parquet) -> None:
    pa, pq = pyarrow_and_parquet
    pq.write_table(
        pa.table({"outcome": ["escalated"], "cost_microusd": [500_000]}),
        tmp_path / "calls-00000000.parquet",
    )

    assert load_records(tmp_path) == read_parquet(tmp_path)
