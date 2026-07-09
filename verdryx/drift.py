"""Quality drift detection: compare recent eval runs against a stored baseline.

A DriftReport pools every case Score across a window of the most recent eval
runs for one model into a single mean, then compares that mean to a stored
Baseline.mean_score. Pooling raw scores (rather than averaging each run's own
mean_score) keeps the comparison correct even when runs have different case
counts, avoiding a Simpson's-paradox-style distortion that a mean-of-means
would introduce.

The verdict is a threshold call: `on-track` unless the pooled mean has
dropped by at least `threshold` below the baseline, in which case
`regressed`. Verdryx does not attempt significance testing (no p-values, no
confidence intervals) for the MVP -- a flat threshold is the documented
starting point; a statistically-aware verdict is a later enhancement.
"""

from __future__ import annotations

from collections.abc import Sequence

from verdryx.models import Baseline, DriftReport, EvalRun

#: Default drop, in absolute score points (mean_score is 0..1), that counts
#: as a regression. 0.05 means a 5-percentage-point drop trips the verdict.
DEFAULT_THRESHOLD = 0.05


def compute_drift(
    recent_runs: Sequence[EvalRun],
    baseline: Baseline,
    *,
    threshold: float = DEFAULT_THRESHOLD,
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
            negative number) that counts as "regressed".

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

    return DriftReport(
        baseline_id=baseline.id,
        window=len(recent_runs),
        mean_score=mean_score,
        delta=delta,
        verdict=verdict,
    )
