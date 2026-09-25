"""Pure metric functions for the bake-off report: no I/O, no network, so
every one of these is testable against a hand-computed value with nothing
faked out.

A "pair" throughout is `(p, truth)`: `p` is the judge's stated probability
that the case is true (`GraderKind.TYPED`'s noul `probabilities["true"]`, or
`LLMJudgeGrader`'s 0..1 score treated the same way), `truth` is the case's
known-by-construction ground truth (see dataset.py). The predicted label is
always `p >= 0.5`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyAt:
    """Confusion-style summary at one threshold (always 0.5 here), always
    carried with its own `n` -- CLAUDE.md invariant 8's rule ("a ratio
    travels with its interval and its n") applied to this harness's own
    report, not only to verdryx's SLO layer."""

    n: int
    accuracy: float
    passed_wrong: int  # predicted true, truth false (a false positive)
    failed_right: int  # predicted false, truth true (a false negative)


def accuracy_at(pairs: list[tuple[float, bool]], threshold: float = 0.5) -> AccuracyAt:
    if not pairs:
        return AccuracyAt(n=0, accuracy=float("nan"), passed_wrong=0, failed_right=0)
    correct = passed_wrong = failed_right = 0
    for p, truth in pairs:
        predicted = p >= threshold
        if predicted == truth:
            correct += 1
        elif predicted and not truth:
            passed_wrong += 1
        elif (not predicted) and truth:
            failed_right += 1
    return AccuracyAt(
        n=len(pairs),
        accuracy=correct / len(pairs),
        passed_wrong=passed_wrong,
        failed_right=failed_right,
    )


def mean_confidence(pairs: list[tuple[float, bool]]) -> float:
    """Mean of max(p, 1-p): how sure the judge stated it was, regardless of
    whether it was right. NaN for an empty list (never 0.0, which would
    read as "certain of the wrong answer every time")."""
    if not pairs:
        return float("nan")
    return sum(max(p, 1.0 - p) for p, _ in pairs) / len(pairs)


def brier(pairs: list[tuple[float, bool]]) -> float:
    """Mean squared error between the stated probability and the 0/1 truth.
    0.0 is perfect, 1.0 is confidently wrong every time; a judge that always
    states 0.5 scores 0.25 regardless of the truth distribution."""
    if not pairs:
        return float("nan")
    return sum((p - (1.0 if truth else 0.0)) ** 2 for p, truth in pairs) / len(pairs)


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected Calibration Error, binned on the stated probability `p`
    itself (equal-width bins over [0, 1]), NOT on max(p, 1-p) ("confidence"
    binning, the other common convention). Per bin: the mean of the judge's
    own `p` compared against the empirical fraction of that bin's cases that
    were actually true, weighted by bin size and summed. Documented here so
    the report can say which convention it used, per the brief's "say
    which": **p-binned**, not confidence-binned.

    0.0 is perfectly calibrated (in every bin, the fraction of true cases
    matches the stated probability). An empty bin contributes nothing.
    """
    if not pairs or bins < 1:
        return float("nan")
    bucket_sum_p = [0.0] * bins
    bucket_sum_truth = [0.0] * bins
    bucket_n = [0] * bins
    for p, truth in pairs:
        clamped = min(max(p, 0.0), 1.0)
        idx = min(int(clamped * bins), bins - 1)
        bucket_sum_p[idx] += clamped
        bucket_sum_truth[idx] += 1.0 if truth else 0.0
        bucket_n[idx] += 1
    total = len(pairs)
    error = 0.0
    for n_b, sum_p, sum_t in zip(bucket_n, bucket_sum_p, bucket_sum_truth, strict=True):
        if n_b == 0:
            continue
        mean_p = sum_p / n_b
        freq_true = sum_t / n_b
        error += (n_b / total) * abs(mean_p - freq_true)
    return error


def percentile(values: list[float], pct: float) -> float:
    """The `pct`-th percentile (0..100) of `values`, linear interpolation
    between the two nearest ranks (the same method `numpy.percentile`'s
    default uses), so a caller doesn't have to install numpy for one number.
    NaN for an empty list. Raises ValueError for pct outside [0, 100].
    """
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


@dataclass(frozen=True)
class Agreement:
    n: int
    agreement: float  # fraction of the intersection where both predicted labels match


def agreement(preds_a: dict[str, bool], preds_b: dict[str, bool]) -> Agreement:
    """Fraction of cases both judges answered (by case id) where their
    predicted labels (p >= 0.5) match. `n` is the size of that
    intersection, carried alongside the fraction for the same reason every
    other ratio in this module carries one."""
    shared = preds_a.keys() & preds_b.keys()
    if not shared:
        return Agreement(n=0, agreement=float("nan"))
    matches = sum(1 for case_id in shared if preds_a[case_id] == preds_b[case_id])
    return Agreement(n=len(shared), agreement=matches / len(shared))


def cost_per_1000_answered(total_cost_usd: float, answered: int) -> float:
    """USD cost per 1000 answered cases. NaN when nothing was answered (a
    cost per zero answers is not zero, it is undefined -- CLAUDE.md
    invariant 8's "an unmeasured indicator is never a zero" applied here)."""
    if answered == 0:
        return float("nan")
    return total_cost_usd / answered * 1000.0
