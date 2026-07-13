"""Tests for verdryx.drift."""

from __future__ import annotations

import random
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


# ------------------------------------------------------------------
# Two-sample significance check (baseline_run)
# ------------------------------------------------------------------


def test_compute_drift_without_baseline_run_skips_significance_check() -> None:
    # No baseline_run passed (the default): compute_drift must behave
    # exactly as it did before this check existed -- no significance
    # fields populated, verdict driven by the flat threshold alone.
    report = compute_drift([_run("r1", [0.87])], _baseline(0.90), threshold=0.05)
    assert report.baseline_n == 0
    assert report.t_statistic is None
    assert report.ci_low is None
    assert report.ci_high is None
    assert report.verdict == "on-track"


def test_compute_drift_baseline_run_with_one_score_falls_back_to_threshold_only() -> None:
    # A single-score baseline run has no defined variance, so the
    # significance check must not run at all (not crash, not report a
    # misleadingly precise interval from one point).
    baseline_run = _run("r0", [0.90])
    report = compute_drift(
        [_run("r1", [0.87])], _baseline(0.90), threshold=0.05, baseline_run=baseline_run
    )
    assert report.baseline_n == 0
    assert report.t_statistic is None
    assert report.ci_low is None
    assert report.ci_high is None


def test_compute_drift_significant_drop_below_threshold_still_flags_regressed() -> None:
    # Baseline run: 10 cases, every one scoring exactly 0.90 (zero
    # variance). Recent: 10 cases, every one scoring exactly 0.87 (also
    # zero variance). delta = -0.03, which the flat 0.05 threshold alone
    # would call "on-track" -- but the drop is perfectly consistent across
    # every case, so the bootstrap CI on the delta collapses to a single
    # point at -0.03 (resampling a constant list always reproduces the
    # same mean), entirely below zero: this is exactly the "smaller but
    # statistically real" drop the significance check exists to catch.
    baseline_run = _run("r0", [0.90] * 10)
    recent = [_run("r1", [0.87] * 10)]
    report = compute_drift(recent, _baseline(0.90), threshold=0.05, baseline_run=baseline_run)

    assert report.delta == pytest.approx(-0.03)
    assert report.baseline_n == 10
    # Zero variance in both samples: no meaningful t-statistic.
    assert report.t_statistic is None
    assert report.ci_low == pytest.approx(-0.03)
    assert report.ci_high == pytest.approx(-0.03)
    assert report.verdict == "regressed"


def test_compute_drift_identical_distributions_stay_on_track() -> None:
    # Baseline and recent draw from the exact same constant value: delta
    # is 0 and the bootstrap CI collapses to a single point at 0, which is
    # not below zero, so the significance check must not flip this to
    # "regressed" on its own.
    baseline_run = _run("r0", [0.9] * 8)
    recent = [_run("r1", [0.9] * 8)]
    report = compute_drift(recent, _baseline(0.9), threshold=0.05, baseline_run=baseline_run)

    assert report.delta == pytest.approx(0.0)
    assert report.ci_low == pytest.approx(0.0)
    assert report.ci_high == pytest.approx(0.0)
    assert report.verdict == "on-track"


def test_compute_drift_welch_t_statistic_matches_hand_computed_value() -> None:
    # sample_a (recent) = [1, 2, 3]: mean 2.0, sample variance 1.0
    # sample_b (baseline) = [4, 5, 6]: mean 5.0, sample variance 1.0
    # standard_error = sqrt(1/3 + 1/3) = sqrt(2/3)
    # t = (2.0 - 5.0) / sqrt(2/3)
    baseline_run = _run("r0", [4.0, 5.0, 6.0])
    recent = [_run("r1", [1.0, 2.0, 3.0])]
    report = compute_drift(
        recent, _baseline(5.0), threshold=0.05, baseline_run=baseline_run, rng=random.Random(1)
    )
    expected_t = (2.0 - 5.0) / ((1.0 / 3 + 1.0 / 3) ** 0.5)
    assert report.t_statistic == pytest.approx(expected_t)


def test_compute_drift_bootstrap_ci_is_deterministic_given_seeded_rng() -> None:
    baseline_run = _run("r0", [0.9, 0.7, 0.8, 0.85, 0.75])
    recent = [_run("r1", [0.85, 0.75, 0.82, 0.88, 0.7])]

    report_a = compute_drift(
        recent, _baseline(0.8), baseline_run=baseline_run, rng=random.Random(1234)
    )
    report_b = compute_drift(
        recent, _baseline(0.8), baseline_run=baseline_run, rng=random.Random(1234)
    )
    assert report_a.ci_low == report_b.ci_low
    assert report_a.ci_high == report_b.ci_high
    assert report_a.ci_low <= report_a.ci_high


def test_compute_drift_ci_bounds_widen_with_higher_confidence() -> None:
    baseline_run = _run("r0", [0.9, 0.6, 0.8, 0.95, 0.7, 0.85])
    recent = [_run("r1", [0.8, 0.5, 0.7, 0.85, 0.6, 0.75])]

    narrow = compute_drift(
        recent, _baseline(0.8), baseline_run=baseline_run, confidence=0.80, rng=random.Random(7)
    )
    wide = compute_drift(
        recent, _baseline(0.8), baseline_run=baseline_run, confidence=0.99, rng=random.Random(7)
    )
    assert wide.ci_low <= narrow.ci_low
    assert wide.ci_high >= narrow.ci_high
