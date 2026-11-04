"""Findings: what `drift` and `slo` write down when they see a failure, and the
fixes registered against them.

Everything in this module is deterministic and reads only the stores and traces
already on disk. No model is called and none is needed: a finding has to be as
useful with the model gone as with it, which is why the suspect, the evidence
and the verdict on a fix are all arithmetic over runs.

## One failure, one finding

A routine runs `drift` every night. A regression that lasts a week must be one
finding seen seven times, not seven findings, so each finding carries a
`dedupe_key` (`drift|<agent>|<baseline>` or `slo|<agent>|<sli>`) and the same
failure seen again refreshes the open one. A closed finding is history: the
same failure after a verified fix is a new finding.

## A fix draws a line in time

Pooled windows keep old runs for a long time. `drift --window 5` still holds the
four runs from before a fix after the fix has shipped, and the 28 day error
budget holds a month of them. So a registered fix marks a horizon (the moment it
was registered): runs from before it are the runs the fix was written against,
they cannot verify it and they cannot undo it. Only runs after the horizon
decide a fix, and only runs after it can open a new finding for the same key.

## Verified needs a measurement

`verified` means the runs after the fix were measured and the failure is gone.
No run since, too few runs to compute the ratio, a run from another build than
the one the fix names, an interval that still covers the target: each of these
leaves the fix `pending` with the reason written down. An unmeasured indicator
is never a pass (CLAUDE.md invariant 8), and a fix that could be "verified" by
silence would close findings by the passage of time.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any

from verdryx import slo
from verdryx.drift import compute_drift
from verdryx.models import (
    FINDING_DRIFT,
    FINDING_SLO_EXHAUSTED,
    FINDING_SLO_FAST_BURN,
    FIX_FAILED,
    FIX_PENDING,
    FIX_VERIFIED,
    STATUS_CLOSED,
    STATUS_FIX_PENDING,
    STATUS_OPEN,
    Baseline,
    DriftReport,
    EvalRun,
    Finding,
    Fix,
    SloMeasurement,
    SloReport,
)
from verdryx.store import Store

#: How many of the lowest scoring cases a drift finding names beside the runs.
WORST_CASES = 5

_SLO_KINDS = (FINDING_SLO_EXHAUSTED, FINDING_SLO_FAST_BURN)


class FixError(ValueError):
    """A fix that cannot be registered, with the reason in the message."""


@dataclass(frozen=True)
class Suspect:
    """The model or version that changed across the runs examined.

    `note` always says which of three things this is: a change (and from what
    to what, at which run), no change across N runs, or nothing to compare. The
    last two both leave `model`, `version` and `since_run` None, and only the
    note tells them apart, so it is never empty.
    """

    model: str | None
    version: str | None
    since_run: str | None
    note: str


def dedupe_key(family: str, agent_id: str | None, scope: str) -> str:
    return f"{family}|{agent_id or ''}|{scope}"


def _label(run: EvalRun) -> str:
    return f"{run.model} / {run.agent_version or 'unlabelled'}"


def find_suspect(runs: Sequence[EvalRun]) -> Suspect:
    """The most recent change of model or version across `runs`, oldest to newest.

    The latest change point is the suspect, because it is the newest thing that
    is different about the agent. `model` and `version` are filled in only for
    the part that moved: a version bump on one model suspects the version and
    not the model.
    """
    ordered = sorted({r.id: r for r in runs}.values(), key=lambda r: r.started_at)
    if len(ordered) < 2:
        return Suspect(
            None,
            None,
            None,
            "fewer than two runs to examine: there is nothing to compare, so a change in "
            "model or version cannot be seen",
        )
    change: tuple[EvalRun, EvalRun] | None = None
    for prev, cur in pairwise(ordered):
        if (prev.model, prev.agent_version) != (cur.model, cur.agent_version):
            change = (prev, cur)
    if change is None:
        note = f"model and version did not change across the {len(ordered)} runs examined"
        if all(r.agent_version is None for r in ordered):
            note += " (none of them carries a version label)"
        return Suspect(None, None, None, note)
    prev, cur = change
    return Suspect(
        cur.model if cur.model != prev.model else None,
        cur.agent_version if cur.agent_version != prev.agent_version else None,
        cur.id,
        f"changed from {_label(prev)} to {_label(cur)} at run {cur.id}",
    )


def _num(x: float | None) -> float | str | None:
    """A float that survives JSON: infinities become text, never a silent null."""
    if x is None or math.isfinite(x):
        return x
    return "inf" if x > 0 else "-inf" if x < 0 else "nan"


def _merge_evidence(new: list[str], old: list[str]) -> list[str]:
    seen: set[str] = set()
    merged = []
    for item in [*new, *old]:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged[: slo.EVIDENCE_CAP]


def _new_finding(
    *,
    key: str,
    agent_id: str | None,
    kind: str,
    sli: str | None,
    window: str,
    evidence: list[str],
    detail: dict[str, Any],
    suspect: Suspect,
    now: datetime,
) -> Finding:
    return Finding(
        id=str(uuid.uuid4()),
        dedupe_key=key,
        agent_id=agent_id,
        kind=kind,
        sli=sli,
        window=window,
        status=STATUS_OPEN,
        opened_at=now,
        last_seen_at=now,
        seen_count=1,
        evidence=evidence[: slo.EVIDENCE_CAP],
        detail=detail,
        suspect_model=suspect.model,
        suspect_version=suspect.version,
        suspect_since_run=suspect.since_run,
        suspect_note=suspect.note,
    )


def _upsert(store: Store, draft: Finding) -> tuple[Finding, bool]:
    """Store `draft`, or refresh the active finding with the same key.

    Returns the stored finding and whether it is new. A refresh keeps the id,
    the opening time and the status, counts the sighting, merges the evidence
    (newest first, capped) and takes the latest numbers, window and suspect.
    """
    active = store.find_active_by_key(draft.dedupe_key)
    if active is None:
        store.add_finding(draft)
        return draft, True
    active.last_seen_at = draft.last_seen_at
    active.seen_count += 1
    active.kind = draft.kind
    active.window = draft.window
    active.evidence = _merge_evidence(draft.evidence, active.evidence)
    active.detail = draft.detail
    active.suspect_model = draft.suspect_model
    active.suspect_version = draft.suspect_version
    active.suspect_since_run = draft.suspect_since_run
    active.suspect_note = draft.suspect_note
    store.update_finding(active)
    return active, False


def _pending_fix(finding: Finding) -> Fix | None:
    pending = [f for f in finding.fixes if f.status == FIX_PENDING]
    return pending[-1] if pending else None


def _short(identifier: str) -> str:
    return identifier[:8]


# ------------------------------------------------------------------
# Registering a fix
# ------------------------------------------------------------------


def register_fix(
    store: Store, finding_id: str, note: str, *, version: str | None, now: datetime
) -> Fix:
    """Record that a fix is on the way to a finding; the next runs will judge it."""
    finding = store.get_finding(finding_id)
    if finding is None:
        raise FixError(f"no such finding: {finding_id!r}")
    if not note.strip():
        raise FixError("a fix needs a note saying what was changed")
    if finding.status == STATUS_CLOSED:
        raise FixError(f"finding {_short(finding.id)} is closed: a verified fix already closed it")
    pending = _pending_fix(finding)
    if pending is not None:
        raise FixError(
            f"finding {_short(finding.id)} already has fix {_short(pending.id)} pending: "
            "it has to be decided by a run before another is registered"
        )
    if version and finding.kind != FINDING_DRIFT:
        raise FixError(
            "--version can only select eval runs, and this finding rests on a trace, "
            "which carries no version"
        )
    fix = Fix(
        id=str(uuid.uuid4()),
        finding_id=finding.id,
        note=note.strip(),
        registered_at=now,
        version=version or None,
    )
    store.add_fix(fix)
    finding.status = STATUS_FIX_PENDING
    store.update_finding(finding)
    return fix


def _apply(
    store: Store,
    finding: Finding,
    fix: Fix,
    status: str,
    result: dict[str, Any],
    now: datetime,
) -> str:
    """Write a fix's evaluation onto the fix and the finding; return the CLI line."""
    fix.result = result
    if status == FIX_PENDING:
        store.update_fix(fix)
        return f"fix {_short(fix.id)} for finding {_short(finding.id)} is still pending: {result['reason']}"
    fix.status = status
    fix.evaluated_at = now
    store.update_fix(fix)
    if status == FIX_VERIFIED:
        finding.status = STATUS_CLOSED
        finding.closed_at = now
        store.update_finding(finding)
        return (
            f"fix {_short(fix.id)} verified: finding {_short(finding.id)} is closed "
            f"({result['summary']})"
        )
    finding.status = STATUS_OPEN
    finding.last_seen_at = now
    finding.seen_count += 1
    store.update_finding(finding)
    return (
        f"fix {_short(fix.id)} failed: finding {_short(finding.id)} is open again "
        f"({result['summary']}); the attempt stays on its record"
    )


# ------------------------------------------------------------------
# Drift
# ------------------------------------------------------------------


def runs_that_test_a_fix(runs: Sequence[EvalRun], fix: Fix) -> list[EvalRun]:
    """The runs after `fix` was registered that carry the build it names.

    A fix that names a version is tested by runs of that version only: the next
    scheduled run may still be the old build, and judging the fix by it would
    fail a fix that was never in the run.
    """
    after = [r for r in runs if r.started_at > fix.registered_at]
    if fix.version is None:
        return after
    return [r for r in after if r.agent_version == fix.version]


def _drift_evidence(runs: Sequence[EvalRun]) -> list[str]:
    newest_first = sorted(runs, key=lambda r: r.started_at, reverse=True)
    flat = [
        (s.value, n, r.id, s.case_id)
        for n, (r, s) in enumerate((r, s) for r in newest_first for s in r.scores)
    ]
    worst = [f"eval:{rid}:{cid}" for _v, _n, rid, cid in sorted(flat)[:WORST_CASES]]
    return [f"eval:{r.id}" for r in newest_first] + worst


def _drift_window(runs: Sequence[EvalRun]) -> str:
    ordered = sorted(runs, key=lambda r: r.started_at)
    return (
        f"{len(ordered)} run(s) from {ordered[0].started_at.isoformat()} "
        f"to {ordered[-1].started_at.isoformat()}"
    )


def _drift_detail(
    report: Any, baseline: Baseline, runs: Sequence[EvalRun], threshold: float
) -> dict[str, Any]:
    return {
        "baseline_id": baseline.id,
        "baseline_mean": baseline.mean_score,
        "mean_score": report.mean_score,
        "delta": report.delta,
        "threshold": threshold,
        "verdict": report.verdict,
        "baseline_n": report.baseline_n,
        "t_statistic": _num(report.t_statistic),
        "ci_low": _num(report.ci_low),
        "ci_high": _num(report.ci_high),
        "runs": len(runs),
        "cases": sum(len(r.scores) for r in runs),
        "human_graded": sum(r.human_graded for r in runs),
        "models": sorted({r.model for r in runs}),
    }


def evaluate_drift_fix(
    tested: Sequence[EvalRun],
    *,
    since: int,
    fix: Fix,
    baseline: Baseline,
    baseline_run: EvalRun | None,
    threshold: float,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    """Decide a drift fix from the runs that test it: `(status, result)`.

    `since` is how many runs finished after the fix was registered at all, so
    the reason can tell "nothing has run" from "runs ran, of another build".
    """
    if not tested:
        if since == 0:
            return FIX_PENDING, {"reason": "no eval run has started since the fix was registered"}
        return FIX_PENDING, {
            "reason": f"{since} run(s) since the fix, none of version {fix.version!r}"
        }
    try:
        report = compute_drift(
            tested, baseline, threshold=threshold, baseline_run=baseline_run, rng=rng
        )
    except ValueError:
        return FIX_PENDING, {"reason": "no scored case in the runs since the fix"}
    result = {
        "verdict": report.verdict,
        "mean_score": report.mean_score,
        "delta": report.delta,
        "ci_low": _num(report.ci_low),
        "ci_high": _num(report.ci_high),
        "runs": [r.id for r in tested],
        "summary": (
            f"{len(tested)} run(s) since the fix, mean {report.mean_score:.3f}, "
            f"delta {report.delta:+.3f} against the baseline"
        ),
    }
    return (FIX_FAILED if report.verdict == "regressed" else FIX_VERIFIED), result


def process_drift(
    store: Store,
    *,
    baseline: Baseline,
    baseline_run: EvalRun | None,
    recent: Sequence[EvalRun],
    threshold: float,
    agent_id: str | None,
    now: datetime,
    rng: random.Random | None = None,
    report: DriftReport | None = None,
) -> list[str]:
    """Record, refresh, close or reopen the finding for a drift check.

    `report` is the verdict the CLI already printed for the whole window. It is
    reused when no fix has cut any run out of that window, so the finding and
    the printed verdict cannot disagree on a bootstrap interval that sits at
    zero; with runs cut out it is a different set of runs and is computed fresh.

    Returns the lines the CLI prints. See the module docstring for the rules:
    one finding per key, a fix decided only by runs after it, a verified fix
    not undone by the runs it was verified against.
    """
    lines: list[str] = []
    key = dedupe_key("drift", agent_id, baseline.id)
    horizon = store.fix_horizon(key)
    usable = [r for r in recent if horizon is None or r.started_at > horizon]
    if horizon is not None and len(usable) < len(recent):
        lines.append(
            f"{len(recent) - len(usable)} run(s) of the window predate the newest fix "
            "and are not counted toward a finding"
        )

    active = store.find_active_by_key(key)
    fix = _pending_fix(active) if active is not None else None
    if active is not None and fix is not None:
        tested = runs_that_test_a_fix(usable, fix)
        status, result = evaluate_drift_fix(
            tested,
            since=len(usable),
            fix=fix,
            baseline=baseline,
            baseline_run=baseline_run,
            threshold=threshold,
            rng=rng,
        )
        if status == FIX_FAILED:
            # The failing numbers become the finding's current ones, so the
            # reopened finding describes the regression that is there now.
            draft = _drift_finding(
                key, agent_id, tested, baseline, baseline_run, threshold, now, rng
            )
            if draft is not None:
                active.evidence = _merge_evidence(draft.evidence, active.evidence)
                active.detail = draft.detail
                active.window = draft.window
        lines.append(_apply(store, active, fix, status, result, now))
        return lines

    if not usable:
        return lines
    same_window = report if len(usable) == len(recent) else None
    draft = _drift_finding(
        key, agent_id, usable, baseline, baseline_run, threshold, now, rng, same_window
    )
    if draft is None:
        return lines
    finding, is_new = _upsert(store, draft)
    verb = "opened" if is_new else f"seen again ({finding.seen_count} sightings)"
    lines.append(f"finding {_short(finding.id)} {verb}: drift, {finding.suspect_note}")
    return lines


def _drift_finding(
    key: str,
    agent_id: str | None,
    runs: Sequence[EvalRun],
    baseline: Baseline,
    baseline_run: EvalRun | None,
    threshold: float,
    now: datetime,
    rng: random.Random | None,
    precomputed: DriftReport | None = None,
) -> Finding | None:
    """A draft finding for `runs` against `baseline`, or None when they are not
    regressed (or hold no scored case, so there is nothing to say)."""
    if not runs:
        return None
    report = precomputed
    if report is None:
        try:
            report = compute_drift(
                runs, baseline, threshold=threshold, baseline_run=baseline_run, rng=rng
            )
        except ValueError:
            return None
    if report.verdict != "regressed":
        return None
    examined = list(runs) + ([baseline_run] if baseline_run is not None else [])
    return _new_finding(
        key=key,
        agent_id=agent_id,
        kind=FINDING_DRIFT,
        sli=None,
        window=_drift_window(runs),
        evidence=_drift_evidence(runs),
        detail=_drift_detail(report, baseline, runs, threshold),
        suspect=find_suspect(examined),
        now=now,
    )


# ------------------------------------------------------------------
# Error budgets
# ------------------------------------------------------------------


def _since(rows: Sequence[Mapping[str, Any]], when: datetime) -> list[Mapping[str, Any]]:
    """Rows that are provably after `when`: a row with no timestamp is left out,
    because it cannot be placed on either side of the line."""
    cutoff = int(when.timestamp() * 1000)
    return [
        r for r in rows if r.get("last_ts_millis") is not None and int(r["last_ts_millis"]) > cutoff
    ]


def _measurement(report: SloReport, agent_id: str | None, sli: str | None) -> SloMeasurement | None:
    for m in report.subjects.get(agent_id or "", []):
        if m.sli == sli:
            return m
    return None


def _breaches(m: SloMeasurement | None) -> bool:
    return (
        m is not None
        and m.trigger in (slo.TRIGGER_EXHAUSTED, slo.TRIGGER_FAST_BURN)
        and slo.breach_is_established(m)
    )


def evaluate_slo_fix(m: SloMeasurement | None) -> tuple[str, dict[str, Any]]:
    """Decide an error-budget fix from the indicator measured over the runs
    after it: `(status, result)`.

    `failed` needs the breach to be established again (the bar the bus and the
    findings use). `verified` needs the objective not to be shown missed AND
    not to be missed by the point estimate either. In between, a point estimate
    below the target with an interval that still covers it is inconclusive and
    the fix stays pending, because "not proven bad" is not "fixed".
    """
    if m is None:
        return FIX_PENDING, {"reason": "no run for this agent since the fix"}
    if not m.measured:
        return FIX_PENDING, {"reason": f"not measured since the fix: {m.unmeasured_reason}"}
    result = {
        "observed": m.observed,
        "target": m.target,
        "ci_low": m.ci_low,
        "ci_high": m.ci_high,
        "events": m.events,
        "trigger": m.trigger,
        "budget_remaining": _num(m.remaining),
        "summary": (
            f"{m.events} run(s) since the fix, {m.sli} {m.observed:.4f} "
            f"against a target of {m.target}"
        ),
    }
    if _breaches(m):
        return FIX_FAILED, result
    if m.trigger in (slo.TRIGGER_EXHAUSTED, slo.TRIGGER_FAST_BURN):
        return FIX_PENDING, {
            **result,
            "reason": (
                f"inconclusive: {m.sli} is {m.observed:.4f} against {m.target} but the "
                f"interval [{m.ci_low:.4f}, {m.ci_high:.4f}] still covers the target"
            ),
        }
    return FIX_VERIFIED, result


def _slo_finding(report: SloReport, m: SloMeasurement, store: Store, now: datetime) -> Finding:
    runs = [r for r in store.list_runs() if r.agent_id == m.subject]
    suspect = (
        find_suspect(runs)
        if runs
        else Suspect(
            None,
            None,
            None,
            "no eval runs for this agent in the store: model and version cannot be compared",
        )
    )
    return _new_finding(
        key=dedupe_key("slo", m.subject, m.sli),
        agent_id=m.subject,
        kind=FINDING_SLO_EXHAUSTED if m.trigger == slo.TRIGGER_EXHAUSTED else FINDING_SLO_FAST_BURN,
        sli=m.sli,
        window=report.window,
        evidence=report.evidence.get((m.subject, m.sli), []),
        detail={
            "sli": m.sli,
            "target": m.target,
            "observed": m.observed,
            "ci_low": m.ci_low,
            "ci_high": m.ci_high,
            "events": m.events,
            "good": m.good,
            "bad": m.events - m.good,
            "budget_remaining": _num(m.remaining),
            "burn_rate": _num(m.burn_rate),
            "burn_events_seen": m.burn_events_seen,
            "trigger": m.trigger,
            "identity_field": report.identity_field,
        },
        suspect=suspect,
        now=now,
    )


def process_slo(
    store: Store,
    report: SloReport,
    *,
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    recompute: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], SloReport],
    now: datetime,
) -> list[str]:
    """Record, refresh, close or reopen the findings for an error-budget report.

    `recompute` is `compute_slo` with the caller's options bound, called over
    the rows after a fix or a horizon. Returns the lines the CLI prints.
    """
    lines: list[str] = []
    breaches = slo.established_breaches(report)
    if report.identity_field != slo.IDENTITY_AGENT_ID:
        if breaches:
            lines.append(
                f"{len(breaches)} breach(es) not recorded as a finding: the subject is a "
                f"{report.identity_field} and a finding needs an agent"
            )
        return lines

    handled: set[str] = set()
    for finding in store.list_findings((STATUS_FIX_PENDING,), None):
        fix = _pending_fix(finding)
        if finding.kind not in _SLO_KINDS or fix is None:
            continue
        sub = recompute(_since(records, fix.registered_at), _since(scores, fix.registered_at))
        m = _measurement(sub, finding.agent_id, finding.sli)
        status, result = evaluate_slo_fix(m)
        if status == FIX_FAILED and m is not None:
            fresh = _slo_finding(sub, m, store, now)
            finding.evidence = _merge_evidence(fresh.evidence, finding.evidence)
            finding.detail = fresh.detail
            finding.kind = fresh.kind
        lines.append(_apply(store, finding, fix, status, result, now))
        handled.add(finding.dedupe_key)

    for m in breaches:
        key = dedupe_key("slo", m.subject, m.sli)
        if key in handled:
            continue
        active = store.find_active_by_key(key)
        if active is not None and active.status == STATUS_FIX_PENDING:
            continue
        source_report, source = report, m
        horizon = store.fix_horizon(key)
        if horizon is not None:
            sub = recompute(_since(records, horizon), _since(scores, horizon))
            again = _measurement(sub, m.subject, m.sli)
            if not _breaches(again) or again is None:
                lines.append(
                    f"{m.subject} {m.sli}: no new finding, only runs from before the newest "
                    "fix show the breach"
                )
                continue
            source_report, source = sub, again
        finding, is_new = _upsert(store, _slo_finding(source_report, source, store, now))
        verb = "opened" if is_new else f"seen again ({finding.seen_count} sightings)"
        lines.append(f"finding {_short(finding.id)} {verb}: {finding.kind} {finding.sli}")
    return lines
