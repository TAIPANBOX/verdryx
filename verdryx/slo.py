"""Agent SLOs: service level indicators, error budgets, and burn rate.

The question this answers is the one enterprises say blocks them and cannot
express: *when is an agent reliable enough*. Quality scores say how good one
answer was; drift says whether that has changed. Neither converts into a
number an operator can spend, and "spend" is what makes an objective
actionable: a budget you can be out of is a budget somebody argues about
before it is gone.

## Ratios, never means

Every SLI here is `good / eligible` over RUNS. That is the one design decision
the rest follows from, so it is worth the paragraph.

A mean score cannot become a budget. Ask "how much of our allowance for bad
answers is left" of a mean and there is no answer, because a mean has no
allowance in it. It also hides the shape: a fleet scoring 0.9 on every run and
a fleet alternating 1.0 and 0.8 have the same mean and very different
reliability, and only the second one is going to embarrass somebody. A ratio
of runs over a threshold prices exactly what an operator cares about, which is
how often the thing was unacceptable.

The unit is a RUN and not a call, because that is what a person asked the
agent to do. Ten calls inside one task is one success or one failure, not ten,
and counting calls would let a chatty agent dilute its own failures. This
matches `costper._reduce_call_rows`, which already reduces a trace to one
record per `run_id` under its last non-empty outcome tag, mirroring
tokenfuse-core's own `compute_outcomes`.

## Two numbers, and the difference between them is the point

`observed` is what happened. `ci_low`/`ci_high` is the Wilson interval around
it, and the reason both are here is that a ratio without its sample size is
not evidence: three of four runs is 0.75 and means nothing, three hundred of
four hundred is 0.75 and means a great deal. Anything that acts on `observed`
alone will act on noise the first quiet week a fleet has.

Wilson rather than the textbook normal interval, because the normal one is
wrong exactly where this lives: near 1.0, with the small samples a single
agent produces, it happily reports an upper bound above 1 and a lower bound
that has no relationship to the data.

## What is NOT here, and why

**Latency.** @measured 2026-08-26: tokenfuse's `CallRecord`
(`crates/gateway/src/sink.rs`) carries `ts_millis` and no duration field of
any kind, and its own `focusexport.rs` says so in as many words: "the trace
records exactly one timestamp per call (`ts_millis`, the settle time), there
is no separate call-start timestamp". So a latency SLI is not something this
module chose to omit, it is not computable from the record that exists, and
inventing one from the gap between two rows would be measuring the agent's
think time and the operator's queue as one number.

**Enforcement.** This module MEASURES. It writes no policy, demotes no agent
and gates nothing. `@yurii 2026-08-26`: "лише вимірювання". That is not a
half-built state waiting to be finished, it is the honest shape of what the
estate can support today, and the reason is worth carrying:

  - wardryx's PDP is a pure function of (policy set, request), enforced by its
    own `scripts/decision-path-purity.sh`, so it cannot read a budget. A gate
    would have to WRITE policy.
  - No autonomy tier exists anywhere in the estate to write. There is no
    per-agent mode, level or state in wardryx, in the Passport, or on the bus.
  - The budget can only be keyed soundly on `key_id`, which tokenfuse resolves
    server-side, while wardryx policies target `agent_id` globs, which
    tokenfuse's own source calls "unsound as the key of a budget, which a
    caller could then move off simply by sending a different one". The join
    between the two is the gateway's identity map, which is off by default.

A gate built over that seam would look identical whether it was holding or
not, which is the failure this whole plane exists to end.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from verdryx.costper import UNTAGGED
from verdryx.events import is_canonical_agent_id
from verdryx.models import OUTCOME_RESOLVED, SloMeasurement, SloReport

#: The four indicators. Named here rather than inferred, because a consumer
#: routing on the name needs the set to be a decision somebody made.
SLI_TASK_SUCCESS = "task_success"
SLI_QUALITY_FLOOR = "quality_floor"
SLI_CONTAINMENT = "containment"
SLI_COST_DISCIPLINE = "cost_discipline"
SLI_NAMES = (SLI_TASK_SUCCESS, SLI_QUALITY_FLOOR, SLI_CONTAINMENT, SLI_COST_DISCIPLINE)

#: Which outcome tags count as the task having succeeded.
#:
#: Only `case_resolved`. `escalated` is deliberately a FAILURE of this
#: indicator, and it is the one default most likely to be argued with, so the
#: reason is here rather than in a commit message: the budget prices autonomy,
#: and an escalation is a person being pulled in. An escalation is a good
#: outcome for the customer and a failure of the objective "this agent handles
#: this without us", and those are different questions. An operator who wants
#: the other reading passes `good_outcomes` and gets it.
DEFAULT_GOOD_OUTCOMES = (OUTCOME_RESOLVED,)

#: The objective, as a fraction of eligible runs that must be good.
DEFAULT_TARGET = 0.95

#: Matches `drift.DEFAULT_CONFIDENCE`, so the two things this repository says
#: about a fleet are said at the same confidence rather than at two.
DEFAULT_CONFIDENCE = 0.95

#: Below this many eligible runs an SLI is reported UNMEASURED rather than
#: computed. Not a smoothing constant: the interval already widens on small
#: samples, and this exists so a subject with four runs produces a stated
#: absence rather than a number somebody screenshots.
DEFAULT_MIN_EVENTS = 20

#: A per-run score at or above this counts as meeting the quality floor.
DEFAULT_QUALITY_FLOOR = 0.8

#: A run costing more than this multiple of the fleet reference fails cost
#: discipline. Measured against the FLEET, never against the agent's own
#: history, because an agent that has been expensive since the day it shipped
#: would otherwise be graded against its own bad habit and pass forever.
DEFAULT_COST_MULTIPLE = 3.0

#: Burn-rate triggers. `slow_burn` is computed and reported and is
#: deliberately NOT emitted onto the event bus: severity is fixed per type in
#: this estate, so one type is one paging band, and "the budget will be gone
#: by Friday" does not belong in the band "the budget is gone". See
#: `events.EVENT_SEVERITY` and the same reasoning tokenfuse recorded when
#: `breaker_tripped` was lowered from critical on 2026-08-03.
TRIGGER_EXHAUSTED = "exhausted"
TRIGGER_FAST_BURN = "fast_burn"
TRIGGER_SLOW_BURN = "slow_burn"

#: Burn multiples that trip each trigger. A burn rate of 1.0 means the budget
#: is being consumed exactly as fast as the window allows, so it lands empty
#: precisely at the end and nothing is wrong.
FAST_BURN_RATE = 6.0
SLOW_BURN_RATE = 2.0

#: What fraction of the window the burn rate is measured over.
#:
#: **This is load-bearing and the module was WRONG without it.** Written first
#: with the budget and the burn rate computed over the same runs, which makes
#: `burn_rate` exactly `1 - remaining`: rate >= 1 then implies remaining <= 0,
#: exhaustion wins every comparison, and `fast_burn` and `slow_burn` become
#: unreachable code. A trigger that can never fire reports identically to one
#: with nothing to report, which is the failure this estate names most often
#: in its own tooling, and it was caught here by a test asserting a slow burn
#: and getting `None`.
#:
#: So the two numbers answer two different questions, which is what they were
#: always meant to do: the BUDGET is "how much of the allowance is left across
#: the whole window", and the RATE is "how fast is it going right now". Only
#: the second can warn, because the first is already history.
BURN_WINDOW_FRACTION = 1.0 / 28.0

#: Identity fields a subject may be grouped on.
IDENTITY_AGENT_ID = "agent_id"
IDENTITY_KEY_ID = "key_id"
IDENTITY_FIELDS = (IDENTITY_AGENT_ID, IDENTITY_KEY_ID)

#: What each identity field is worth, said once so every report can say it.
IDENTITY_NOTES = {
    IDENTITY_AGENT_ID: (
        "client-supplied header; sound for attribution a cooperating fleet "
        "reports about itself, unsound as the key of anything enforced"
    ),
    IDENTITY_KEY_ID: (
        "server-resolved from the presented credential; empty unless the "
        "gateway has client keys configured, which is off by default"
    ),
}


class SloInputError(ValueError):
    """A caller asked for something this module cannot honestly compute."""


def wilson_interval(good: int, total: int, confidence: float) -> tuple[float, float]:
    """The Wilson score interval for `good`/`total` at `confidence`.

    Returns `(0.0, 1.0)` for an empty sample, which is the honest answer: with
    no observations the proportion could be anything, and returning a point
    estimate of zero would read as "everything failed".
    """
    if total <= 0:
        return (0.0, 1.0)
    if not 0.0 < confidence < 1.0:
        raise SloInputError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    z = _z_for(confidence)
    p = good / total
    denom = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z * z) / (4 * total * total))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _z_for(confidence: float) -> float:
    """The two-sided normal quantile for `confidence`.

    `statistics.NormalDist` is stdlib, so this needs no dependency and no
    hand-rolled table. Invariant 1 is the reason that matters: a z-table
    copied in from somewhere is a second implementation of a published
    constant, which is the shape this estate keeps getting bitten by.
    """
    return statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def error_budget_remaining(observed: float, target: float) -> float:
    """What fraction of the allowance for bad runs is left.

    `1.0` is untouched, `0.0` is exactly spent, and NEGATIVE is overspent:
    `-1.0` means twice as many bad runs as the objective allows. Not clamped,
    because an operator told "0% left" for both a budget just used up and one
    blown through four times over has been told two very different situations
    look identical.
    """
    allowance = 1.0 - target
    if allowance <= 0.0:
        # A target of 1.0 permits no failures at all, so there is no budget to
        # have a fraction of. Perfect is the only non-exhausted state.
        return 1.0 if observed >= 1.0 else -math.inf
    return 1.0 - ((1.0 - observed) / allowance)


def burn_rate(observed: float, target: float) -> float:
    """How many times faster than sustainable the budget is being spent.

    `1.0` spends the whole budget exactly across the window. `0.0` spends
    none. Above `1.0` the budget runs out early, and by how much is the whole
    signal a burn-rate alert carries.
    """
    allowance = 1.0 - target
    if allowance <= 0.0:
        return 0.0 if observed >= 1.0 else math.inf
    return (1.0 - observed) / allowance


def trigger_for(remaining: float, rate: float) -> str | None:
    """Which trigger, if any, this measurement fires.

    Exhaustion outranks rate: once the budget is gone, how fast it went is
    history, and reporting `fast_burn` for a subject already out of budget
    would tell an operator to watch something that has already happened.
    """
    if remaining <= 0.0:
        return TRIGGER_EXHAUSTED
    if rate <= 0.0:
        # Either genuinely nothing failed recently, or the rate could not be
        # measured at all. Neither is a warning, and the report tells the two
        # apart through `burn_events_seen`.
        return None
    if rate >= FAST_BURN_RATE:
        return TRIGGER_FAST_BURN
    if rate >= SLOW_BURN_RATE:
        return TRIGGER_SLOW_BURN
    return None


# --------------------------------------------------------------------------
# The indicators
# --------------------------------------------------------------------------
#
# Each returns `True` (good), `False` (bad), or `None` (this run cannot be
# judged by this indicator, so it is not in the denominator either). The third
# case is the one that keeps the numbers honest: an untagged run is not a
# failed run, and counting it as one would make an unconfigured fleet look
# broken rather than unmeasured.


def _sli_task_success(run: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool | None:
    outcome = str(run.get("outcome") or "")
    if not outcome or outcome == UNTAGGED:
        return None
    return outcome in cfg["good_outcomes"]


def _sli_quality_floor(run: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool | None:
    score = run.get("score")
    if score is None:
        return None
    return float(score) >= cfg["quality_floor"]


def _sli_containment(run: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool | None:
    """Did anything have to stop this run.

    Reads `refused_calls`, which `costper._reduce_call_rows` counts using
    `is_refused_decision`: the nine Breaker reasons AND the two Wardryx ones.
    The two matter and are easy to miss, because `wardryx_deny` and
    `wardryx_hold` are written to the trace, return 403, and are NOT
    `BreakerReason`s, so the nine alone would report a fleet as unrestrained
    while the policy plane refused half its calls.
    """
    refused = run.get("refused_calls")
    if refused is None:
        return None
    return int(refused) == 0


def _sli_cost_discipline(run: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool | None:
    reference = cfg.get("cost_reference")
    if reference is None or reference <= 0.0:
        return None
    cost = run.get("cost_usd")
    if cost is None:
        return None
    return float(cost) <= cfg["cost_multiple"] * reference


SLI_FUNCS = {
    SLI_TASK_SUCCESS: _sli_task_success,
    SLI_QUALITY_FLOOR: _sli_quality_floor,
    SLI_CONTAINMENT: _sli_containment,
    SLI_COST_DISCIPLINE: _sli_cost_discipline,
}

#: Why an SLI came back unmeasured, in words an operator can act on. Keyed by
#: indicator, because "no data" sends somebody to look at the wrong thing:
#: quality is absent for a structural reason and cost for a configuration one.
UNMEASURED_REASONS = {
    SLI_TASK_SUCCESS: (
        "no run carried an outcome tag: the gateway records the "
        "X-Fuse-Outcome header verbatim and it was never sent"
    ),
    # This reason described a limitation that no longer exists, and a reason
    # that has stopped being true is worse than none: it sends a reader to
    # close a gap already closed. Until 2026-08-26 the eval store's
    # `eval_runs` table carried no subject, so a score could not be attributed
    # to a fleet at all and the indicator was dark for a missing COLUMN rather
    # than for missing data. It now names the two commands that carry a score
    # from where it is produced to where it is read.
    SLI_QUALITY_FLOOR: (
        "no per-run score reached this subject. Scores live in verdryx's own "
        "eval store: `verdryx eval --agent-id <id>` stamps the subject onto "
        "the run, and `verdryx slo --scores-db <path>` joins them back. A run "
        "evaluated with no --agent-id belongs to nobody and is counted as an "
        "unattributed score rather than filed under somebody"
    ),
    SLI_CONTAINMENT: "no run carried a call count, so nothing says what was refused",
    SLI_COST_DISCIPLINE: (
        "no cost reference could be computed: every run cost zero, or none carried a cost at all"
    ),
}


def cost_reference(runs: Iterable[Mapping[str, Any]]) -> float | None:
    """The fleet's median run cost, or None when there is nothing to compare.

    The MEDIAN and not the mean, because the mean of a cost distribution is
    dragged by exactly the runaway runs this indicator exists to catch, and an
    indicator whose reference moves toward the thing it measures grades a
    worsening fleet as steady.
    """
    costs = [float(r["cost_usd"]) for r in runs if r.get("cost_usd") is not None]
    positive = [c for c in costs if c > 0.0]
    if not positive:
        return None
    return statistics.median(positive)


def _recent_burn(
    runs: list[Mapping[str, Any]],
    fn: Any,
    cfg: Mapping[str, Any],
    target: float,
) -> tuple[float, int]:
    """The burn rate over the most recent slice of the window, and its n.

    Returns `(0.0, 0)` when the runs carry no usable timestamp, which is an
    honest zero rather than a computed one: without a clock there is no
    "recently", and a rate computed over the whole window is the budget with
    its sign flipped, not a warning about anything.

    The caller must therefore treat `burn_events_seen == 0` as UNMEASURED and
    not as "burning at zero". `_blind_spots` does exactly that, so a report
    over a trace with no `ts_millis` column says the burn windows are blind
    rather than reporting every subject as calm.
    """
    stamped = [(int(r["last_ts_millis"]), r) for r in runs if r.get("last_ts_millis") is not None]
    if len(stamped) < cfg["min_events"]:
        return (0.0, 0)
    stamped.sort(key=lambda pair: pair[0])
    newest = stamped[-1][0]
    oldest = stamped[0][0]
    span = newest - oldest
    if span <= 0:
        # Every run carries the same instant, so there is no recent slice to
        # take. Not an error and not a rate: the same absence as no clock.
        return (0.0, 0)
    cutoff = newest - span * cfg["burn_window_fraction"]
    recent = [r for ts, r in stamped if ts >= cutoff]
    judged = [v for v in (fn(r, cfg) for r in recent) if v is not None]
    if len(judged) < cfg["min_burn_events"]:
        # The window is real and holds too few runs to say anything about.
        # Reported as unmeasured for the same reason as no clock at all.
        return (0.0, 0)
    observed_recent = sum(1 for v in judged if v) / len(judged)
    return (burn_rate(observed_recent, target), len(judged))


def _unmeasured_reason(
    sli: str,
    runs: list[Mapping[str, Any]],
    total: int,
    cfg: Mapping[str, Any],
) -> str:
    """Which of the three absences this is, said in the words for that one.

    Three, not two, and the third arrived with the eval-store join. A subject
    can now be known ONLY from its scores (an agent evaluated offline whose
    gateway traffic nobody tagged), and for such a subject every trace-fed
    indicator has no runs at all. Reporting "no run carried an outcome tag"
    there is the wrong diagnosis: it sends an operator to look at a header
    they never had a chance to send, on runs that were never in the file.
    """
    if not runs:
        return (
            "no runs at all for this subject in this window: it is known here "
            "only from its scores, so nothing in the trace describes it"
        )
    if total == 0:
        return UNMEASURED_REASONS[sli]
    return (
        f"{total} eligible run(s), below the {cfg['min_events']} this "
        f"report will compute a ratio from"
    )


def evaluate_sli(
    runs: list[Mapping[str, Any]],
    sli: str,
    subject: str,
    cfg: Mapping[str, Any],
) -> SloMeasurement:
    """Measure one indicator over one subject's runs."""
    if sli not in SLI_FUNCS:
        raise SloInputError(f"unknown SLI {sli!r}; known: {', '.join(SLI_NAMES)}")
    fn = SLI_FUNCS[sli]
    judged = [v for v in (fn(r, cfg) for r in runs) if v is not None]
    total = len(judged)
    good = sum(1 for v in judged if v)
    target = float(cfg["targets"][sli])

    if total < cfg["min_events"]:
        return SloMeasurement(
            sli=sli,
            subject=subject,
            observed=0.0,
            target=target,
            events=total,
            good=good,
            ci_low=0.0,
            ci_high=1.0,
            remaining=0.0,
            burn_rate=0.0,
            burn_events_seen=0,
            trigger=None,
            measured=False,
            unmeasured_reason=_unmeasured_reason(sli, runs, total, cfg),
        )

    observed = good / total
    low, high = wilson_interval(good, total, cfg["confidence"])
    remaining = error_budget_remaining(observed, target)
    rate, rate_events = _recent_burn(runs, fn, cfg, target)
    return SloMeasurement(
        sli=sli,
        subject=subject,
        observed=observed,
        target=target,
        events=total,
        good=good,
        ci_low=low,
        ci_high=high,
        remaining=remaining,
        burn_rate=rate,
        burn_events_seen=rate_events,
        trigger=trigger_for(remaining, rate),
    )


def compute_slo(
    records: Iterable[Mapping[str, Any]],
    *,
    scores: Iterable[Mapping[str, Any]] | None = None,
    identity_field: str = IDENTITY_AGENT_ID,
    window: str = "28d",
    targets: Mapping[str, float] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    min_events: int = DEFAULT_MIN_EVENTS,
    good_outcomes: Iterable[str] = DEFAULT_GOOD_OUTCOMES,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    cost_multiple: float = DEFAULT_COST_MULTIPLE,
    burn_window_fraction: float = BURN_WINDOW_FRACTION,
    min_burn_events: int = 5,
) -> SloReport:
    """Measure every indicator over every subject in `records`.

    `records` are per-run records as `costper.load_records` produces them.

    Runs whose identity field is empty are counted in `unattributed_runs` and
    appear in NO subject. They are not dropped quietly and they are not
    bucketed under a placeholder subject: a fleet gets a better score the less
    of it is identified either way, and only one of the two says so.

    ## Why `scores` is a second stream and not a column on the first

    `SLI_QUALITY_FLOOR` needs a score per judged unit, and the trace has none:
    tokenfuse records what a call cost and how it was decided, never how good
    the answer was. The scores live in verdryx's own eval store, and the two
    have no run identity in common. A tokenfuse `run_id` is minted by the
    caller for a piece of gateway traffic; an eval run id is a `uuid4` minted
    by `cli.run_eval` for a batch of cases that never went through a gateway.
    Nothing anywhere maps one to the other.

    So the join is on the SUBJECT and on nothing else, which is why a subject
    had to exist in the eval store before this argument could. Each score
    record carries its own identity field, exactly like a run record, and is
    grouped the same way. That shape is deliberate rather than convenient: a
    `{subject: [scores]}` mapping cannot carry a score that HAS no subject
    except under a placeholder key, and a placeholder key is the bucketing
    this report refuses to do to runs.

    Both streams reach `SLI_QUALITY_FLOOR` together. A record that carries its
    own `score` field is a legitimate input (a hand-written NDJSON export
    does) and did not stop being one, so the denominator is every judged unit
    from either source.

    A subject present only in `scores` gets a row of its own. Dropping it
    would be a silent loss of an agent somebody evaluated, and filing it under
    a trace subject would be a fabrication; its three trace-fed indicators
    report that there were no runs at all, which is a different sentence from
    "no run carried an outcome tag" and sends a reader somewhere different.

    ## What joining two streams does to the quality BURN window

    @measured 2026-08-26, running `verdryx slo --traces <dir> --scores-db
    <db>` against a live tokenfuse trace: a subject with 240 scoreless trace
    runs and 40 stored scores computed `quality_floor` fine and reported its
    burn rate as not measurable.

    That is `_recent_burn` behaving as designed and it is worth knowing about.
    The recent slice is a slice of TIME over every row the indicator was
    handed, and a trace run carrying no score is a row `_sli_quality_floor`
    cannot judge. When the scoreless runs are the newest rows, they fill the
    slice, `judged` comes back under `min_burn_events`, and the rate is
    reported unmeasured rather than computed off whatever survived.

    Building the window out of judgeable rows only would fix that figure and
    break something worse: "recently" would then mean a different span for
    each indicator on the same report, one an hour and another a month, with
    nothing on the page saying so. An unmeasured rate that `_blind_spots`
    names is the better of the two, and the two clocks are the reason it can
    happen at all (`store.score_records` stamps the eval run's finish time,
    the trace stamps the gateway's settle time).
    """
    if identity_field not in IDENTITY_FIELDS:
        raise SloInputError(
            f"identity_field must be one of {', '.join(IDENTITY_FIELDS)}, got {identity_field!r}"
        )
    resolved_targets = dict.fromkeys(SLI_NAMES, DEFAULT_TARGET)
    for name, value in (targets or {}).items():
        if name not in SLI_NAMES:
            raise SloInputError(f"unknown SLI in targets: {name!r}")
        if not 0.0 <= value <= 1.0:
            raise SloInputError(f"target for {name!r} must be within 0..1, got {value!r}")
        resolved_targets[name] = float(value)

    rows = list(records)
    cfg: dict[str, Any] = {
        "targets": resolved_targets,
        "confidence": confidence,
        "min_events": min_events,
        "good_outcomes": frozenset(good_outcomes),
        "quality_floor": quality_floor,
        "cost_multiple": cost_multiple,
        "burn_window_fraction": burn_window_fraction,
        "min_burn_events": min_burn_events,
        # Computed over the WHOLE input rather than per subject, which is what
        # makes it a fleet reference. Recomputing it inside each group would
        # compare every agent to itself and pass all of them.
        "cost_reference": cost_reference(rows),
    }

    grouped, unattributed = _group_by_subject(rows, identity_field)
    score_rows = list(scores or ())
    scored, unattributed_scores = _group_by_subject(score_rows, identity_field)

    subjects = {
        subject: [
            evaluate_sli(
                # Both streams, and only for the one indicator a score can
                # speak to. A run carrying no `score` returns None from
                # `_sli_quality_floor` and stays out of the denominator, so
                # concatenating cannot double-count anything.
                grouped.get(subject, []) + scored.get(subject, [])
                if sli == SLI_QUALITY_FLOOR
                else grouped.get(subject, []),
                sli,
                subject,
                cfg,
            )
            for sli in SLI_NAMES
        ]
        for subject in sorted(set(grouped) | set(scored))
    }

    return SloReport(
        window=window,
        identity_field=identity_field,
        subjects=subjects,
        unattributed_runs=unattributed,
        total_runs=len(rows),
        blind_spots=_blind_spots(subjects, min_events),
        cost_reference_usd=cfg["cost_reference"],
        unattributed_scores=unattributed_scores,
        total_scores=len(score_rows),
    )


def _group_by_subject(
    rows: list[Mapping[str, Any]],
    identity_field: str,
) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    """Bucket rows by their identity field, and count the ones that have none.

    One function for runs and for scores, because the rule they are held to is
    one rule: a row with no subject is COUNTED and appears in nobody's
    numbers. Two copies of it would eventually be one copy and one drift, and
    the drift would look like a fleet quietly improving.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    unattributed = 0
    for row in rows:
        subject = str(row.get(identity_field) or "")
        if not subject:
            unattributed += 1
            continue
        grouped.setdefault(subject, []).append(row)
    return grouped, unattributed


def _blind_spots(subjects: Mapping[str, list[SloMeasurement]], min_events: int) -> list[str]:
    """What this report could not see, in the report itself.

    An indicator that never has enough runs to be computed reports exactly
    like one with nothing to report, and the difference matters more than
    almost anything else the report says: the first is a gap in the
    measurement and the second is a healthy fleet.
    """
    out: list[str] = []
    for subject, measurements in subjects.items():
        missing = [m.sli for m in measurements if not m.measured]
        if missing:
            out.append(
                f"{subject}: {', '.join(missing)} not computed "
                f"(fewer than {min_events} eligible runs, or no input for it)"
            )
        blind_rate = [m.sli for m in measurements if m.measured and m.burn_events_seen == 0]
        if blind_rate:
            out.append(
                f"{subject}: burn rate not measurable for "
                f"{', '.join(blind_rate)} (no usable timestamp, or too few "
                f"recent runs), so a fast burn there can never fire"
            )
    return out


def breach_is_established(m: SloMeasurement) -> bool:
    """Whether the evidence shows the objective is missed, not merely that the
    point estimate is below it.

    The upper end of the Wilson interval has to sit below the target. This is
    the difference between "we measured worse than the objective" and "we can
    show we are worse than the objective", and it is the whole reason the
    interval is computed at all.

    It exists because of what the alternative does on a quiet week. A subject
    with 60 runs at 0.9333 against a target of 0.95 has spent a third more
    than its budget by the point estimate, and an interval of [0.841, 0.974]
    that comfortably contains the target: 56 good out of 60 is simply not
    enough evidence to say a fleet is unreliable. Paging on that teaches an
    operator that this sender is noise, and then the one that mattered is
    filtered with it.

    The REPORT still shows the point estimate, the interval and the trigger,
    so nothing is hidden and nobody has to guess why no event was sent. Only
    the bus is held to the higher bar, because a line on the bus is read as a
    fact by consumers that never see the interval.

    `@claude` 2026-08-26. This is a judgement about where to put the bar, not
    a derivation; the two-bar shape is the standard SRE one (alert on the
    observation, act on the evidence) applied to a plane that only observes.
    """
    return m.measured and m.ci_high < m.target


def subject_is_emittable(subject: str) -> bool:
    """Whether this subject can be the `agent_id` of an envelope at all.

    Grouping on `key_id` is the SOUND choice for anything enforced, and it
    produces subjects like `k-ops`, which are credentials rather than agents.
    The envelope has one subject field, it is `agent_id`, and SPEC 3.1 fixes
    its grammar; there is nowhere in it to say "this measurement is about a
    key". So a key-keyed measurement is reported and NOT emitted.

    The alternative was to let `EventLog.emit` do what it does with any
    non-conforming id: warn once and write it anyway. That is right for an
    operator who mislabelled an agent, because the line is still about an
    agent and a consumer can repair it. It is wrong here, because the line
    would not be about an agent at all, and a consumer validating the envelope
    would reject a subject this plane knew was malformed before it wrote it.

    `@claude` 2026-08-26. The gap is real and is stated in the report rather
    than papered over: the identity that is sound to gate on is not the
    identity the bus can carry, which is the same seam invariant 9 records.
    """
    return is_canonical_agent_id(subject)


def burn_events(report: SloReport) -> list[dict[str, Any]]:
    """The `slo_burn` payloads this report justifies, and only those.

    `exhausted` and `fast_burn` reach the bus. A slow burn does not, and the
    omission is the design rather than an oversight: severity is fixed per
    type (`events.EVENT_SEVERITY`), so one type is one paging band, and a
    budget that will be gone by Friday does not belong in the band reserved
    for one already gone. The slow-burn figure is in this report and in its
    JSON, where a dashboard reads it without waking anybody.

    A trigger is necessary and not sufficient: `breach_is_established` also
    has to hold, so the bus carries breaches the interval SHOWS rather than
    every point estimate that happens to sit below a target. See that
    function for why, and note the asymmetry is deliberate: the report is
    generous and the bus is strict, because a report is read by somebody who
    can see the sample size and a bus line is read by a program that cannot.

    An UNMEASURED indicator emits nothing at all. There is no honest event to
    send about a number that was not computed, and sending one would put an
    absence of evidence onto a bus whose readers treat every line as a fact.
    """
    payloads: list[dict[str, Any]] = []
    for measurements in report.subjects.values():
        for m in measurements:
            if m.trigger not in (TRIGGER_EXHAUSTED, TRIGGER_FAST_BURN):
                continue
            if not breach_is_established(m):
                continue
            payloads.append(
                {
                    "sli": m.sli,
                    "target": m.target,
                    "observed": m.observed,
                    "ci_low": m.ci_low,
                    "ci_high": m.ci_high,
                    "events": m.events,
                    "window": report.window,
                    "budget_remaining": m.remaining,
                    "burn_rate": m.burn_rate,
                    "trigger": m.trigger,
                    "identity_field": report.identity_field,
                    "_subject": m.subject,
                }
            )
    return payloads
