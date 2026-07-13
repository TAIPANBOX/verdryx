"""Quality drift detection: compare recent eval runs against a stored baseline.

A DriftReport pools every case Score across a window of the most recent eval
runs for one model into a single mean, then compares that mean to a stored
Baseline.mean_score. Pooling raw scores (rather than averaging each run's own
mean_score) keeps the comparison correct even when runs have different case
counts, avoiding a Simpson's-paradox-style distortion that a mean-of-means
would introduce.

The verdict starts as a threshold call: `on-track` unless the pooled mean has
dropped by at least `threshold` below the baseline, in which case
`regressed`. That flat threshold never goes away -- it is always sufficient
on its own to trip the verdict, exactly as before this module gained
significance testing.

When the caller also supplies `baseline_run` (the baseline's original
EvalRun, reloaded via `Store.load_run(baseline.eval_run_id)`), compute_drift
additionally runs a genuine two-sample comparison between the baseline run's
own per-case scores and the recent pooled scores: a Welch's t-statistic
(informational) plus a bootstrap confidence interval on the delta. If that
interval's upper bound stays below zero, the drop is significant at the
configured confidence level even when it is smaller than `threshold`, and
the verdict is `regressed` on that basis too. This closes the gap a flat
threshold alone leaves: a small, consistent drop across many cases can be
statistically real while still sitting under a threshold sized to filter
noise. `baseline_run` is optional and defaults to None, which reproduces the
original flat-threshold-only behavior exactly -- the significance check is
additive, never a replacement.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from verdryx.models import Baseline, DriftReport, EvalRun

#: Default drop, in absolute score points (mean_score is 0..1), that counts
#: as a regression. 0.05 means a 5-percentage-point drop trips the verdict.
DEFAULT_THRESHOLD = 0.05

#: Default number of bootstrap resamples for the delta confidence interval.
#: 2000 is the conventional floor for a stable 95% CI without taking long.
DEFAULT_BOOTSTRAP_ITERATIONS = 2000

#: Default confidence level for the bootstrap interval (e.g. 0.95 -> 95%,
#: reporting the 2.5th/97.5th percentile of resampled deltas).
DEFAULT_CONFIDENCE = 0.95

#: Minimum sample size, in either group, for the significance check to run
#: at all. Below this, variance (and therefore a t-statistic) is undefined
#: or too noisy to trust, so compute_drift silently falls back to the flat
#: threshold alone rather than reporting a misleadingly precise interval.
_MIN_SAMPLE_FOR_SIGNIFICANCE = 2


def _welch_t_statistic(sample_a: Sequence[float], sample_b: Sequence[float]) -> float | None:
    """Welch's t-statistic for two independent samples of unequal size/variance.

    Returns None when either sample's variance can't be computed (fewer
    than 2 values) or the pooled standard error is exactly zero (both
    samples are constant, e.g. every score identical) -- there is no
    meaningful t-statistic in either case.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return None
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)
    standard_error = math.sqrt(var_a / n_a + var_b / n_b)
    if standard_error == 0:
        return None
    return (statistics.fmean(sample_a) - statistics.fmean(sample_b)) / standard_error


def _bootstrap_delta_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    iterations: int,
    confidence: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for mean(sample_a) - mean(sample_b).

    Resamples each group independently, with replacement, at its own size,
    `iterations` times, and returns the `confidence` percentile interval of
    the resampled deltas (e.g. the 2.5th/97.5th percentile for
    confidence=0.95). Needs no assumption about the underlying score
    distribution, unlike a t-test p-value.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    deltas = []
    for _ in range(iterations):
        resampled_a = [sample_a[rng.randrange(n_a)] for _ in range(n_a)]
        resampled_b = [sample_b[rng.randrange(n_b)] for _ in range(n_b)]
        deltas.append(statistics.fmean(resampled_a) - statistics.fmean(resampled_b))
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * iterations)
    high_index = min(int((1.0 - tail) * iterations), iterations - 1)
    return deltas[low_index], deltas[high_index]


def compute_drift(
    recent_runs: Sequence[EvalRun],
    baseline: Baseline,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    baseline_run: EvalRun | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: random.Random | None = None,
) -> DriftReport:
    """Compare the pooled mean score of `recent_runs` against `baseline`.

    Args:
        recent_runs: The window of most-recent eval runs to pool together.
            Order does not matter; every case Score across every run in the
            sequence is pooled into one mean. Typically the caller has
            already filtered these to the same model as the baseline's run
            (verdryx.cli's `drift` command does this).
        baseline: The stored reference point to compare against.
        threshold: Minimum drop (mean_score - baseline.mean_score, so a
            negative number) that counts as "regressed" on its own.
        baseline_run: The baseline's original EvalRun, with its per-case
            Scores, if the caller has it (verdryx.cli's `drift` command
            already loads this to filter by model, and passes it through).
            When given, unlocks the two-sample significance check described
            in this module's docstring. When None (the default), compute_drift
            behaves exactly as it did before this check existed.
        bootstrap_iterations: Resample count for the delta confidence
            interval. Only used when baseline_run is given.
        confidence: Confidence level for that interval, e.g. 0.95.
        rng: Source of randomness for the bootstrap. Defaults to a fresh,
            unseeded random.Random() for real use; tests pass a seeded one
            for a deterministic interval.

    Returns:
        A DriftReport with window = len(recent_runs).

    Raises:
        ValueError: If recent_runs is empty, or none of the runs in it have
            any scores -- there is nothing to compute a mean over.
    """
    if not recent_runs:
        raise ValueError("recent_runs is empty; cannot compute drift")

    all_values = [s.value for run in recent_runs for s in run.scores]
    if not all_values:
        raise ValueError("no scores found across recent_runs; cannot compute drift")

    mean_score = sum(all_values) / len(all_values)
    delta = mean_score - baseline.mean_score
    verdict = "regressed" if delta <= -threshold else "on-track"

    baseline_values = [s.value for s in baseline_run.scores] if baseline_run is not None else []
    baseline_n = 0
    t_statistic: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    if (
        len(baseline_values) >= _MIN_SAMPLE_FOR_SIGNIFICANCE
        and len(all_values) >= _MIN_SAMPLE_FOR_SIGNIFICANCE
    ):
        baseline_n = len(baseline_values)
        t_statistic = _welch_t_statistic(all_values, baseline_values)
        resolved_rng = rng if rng is not None else random.Random()
        ci_low, ci_high = _bootstrap_delta_ci(
            all_values,
            baseline_values,
            iterations=bootstrap_iterations,
            confidence=confidence,
            rng=resolved_rng,
        )
        if ci_high < 0:
            verdict = "regressed"

    return DriftReport(
        baseline_id=baseline.id,
        window=len(recent_runs),
        mean_score=mean_score,
        delta=delta,
        verdict=verdict,
        baseline_n=baseline_n,
        t_statistic=t_statistic,
        ci_low=ci_low,
        ci_high=ci_high,
    )
