"""Tests for verdryx.drift."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from verdryx.drift import DEFAULT_THRESHOLD, compute_drift
from verdryx.models import Baseline, EvalRun, Score


def _run(run_id: str, values: list[float]) -> EvalRun:
    return EvalRun(
        id=run_id,
        model="stub",
        started_at=datetime.now(tz=UTC),
        scores=[Score(case_id=f"c{i}", value=v) for i, v in enumerate(values)],
    )


def _baseline(mean_score: float) -> Baseline:
    return Baseline(
        id="b1", eval_run_id="r0", mean_score=mean_score, created_at=datetime.now(tz=UTC)
    )


def test_default_threshold_value() -> None:
    assert DEFAULT_THRESHOLD == 0.05


def test_compute_drift_on_track_when_scores_match_baseline() -> None:
    report = compute_drift([_run("r1", [1.0, 1.0, 0.8, 0.8])], _baseline(0.9))
    assert report.mean_score == pytest.approx(0.9)
    assert report.delta == pytest.approx(0.0)
    assert report.verdict == "on-track"
    assert report.window == 1
    assert report.baseline_id == "b1"


def test_compute_drift_on_track_within_threshold() -> None:
    # baseline 0.90, current 0.87 -> delta -0.03, threshold 0.05 -> on-track
    report = compute_drift([_run("r1", [0.87])], _baseline(0.90), threshold=0.05)
    assert report.verdict == "on-track"


def test_compute_drift_regressed_past_threshold() -> None:
    # baseline 0.90, current 0.70 -> delta -0.20, threshold 0.05 -> regressed
    report = compute_drift([_run("r1", [0.7])], _baseline(0.90), threshold=0.05)
    assert report.verdict == "regressed"
    assert report.delta == pytest.approx(-0.2)


def test_compute_drift_exactly_at_threshold_counts_as_regressed() -> None:
    report = compute_drift([_run("r1", [0.85])], _baseline(0.90), threshold=0.05)
    assert report.delta == pytest.approx(-0.05)
    assert report.verdict == "regressed"


def test_compute_drift_pools_scores_across_window_not_mean_of_means() -> None:
    # Run r1 has 2 scores at 1.0, run r2 has 2 scores at 0.0: a mean-of-means
    # would also give 0.5, so add a third, smaller run to prove pooling
    # (not an unweighted average of per-run means) is what's happening.
    runs = [_run("r1", [1.0, 1.0]), _run("r2", [0.0, 0.0]), _run("r3", [1.0])]
    report = compute_drift(runs, _baseline(0.5))
    # Pooled: (1+1+0+0+1)/5 = 0.6; mean-of-means would be (1.0+0.0+1.0)/3 = 0.667
    assert report.mean_score == pytest.approx(0.6)
    assert report.window == 3


def test_compute_drift_improvement_is_on_track_with_positive_delta() -> None:
    report = compute_drift([_run("r1", [1.0])], _baseline(0.5))
    assert report.delta == pytest.approx(0.5)
    assert report.verdict == "on-track"


def test_compute_drift_empty_recent_runs_raises() -> None:
    with pytest.raises(ValueError, match="recent_runs is empty"):
        compute_drift([], _baseline(0.9))


def test_compute_drift_runs_with_no_scores_raises() -> None:
    empty_run = EvalRun(id="r1", model="stub", started_at=datetime.now(tz=UTC))
    with pytest.raises(ValueError, match="no scores found"):
        compute_drift([empty_run], _baseline(0.9))


def test_compute_drift_ignores_empty_runs_mixed_with_scored_ones() -> None:
    empty_run = EvalRun(id="r0", model="stub", started_at=datetime.now(tz=UTC))
    report = compute_drift([empty_run, _run("r1", [1.0])], _baseline(0.5))
    assert report.mean_score == pytest.approx(1.0)
    assert report.window == 2
