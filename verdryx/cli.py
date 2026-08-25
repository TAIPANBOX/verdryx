"""CLI entry point for Verdryx.

Usage::

    verdryx eval <evalset.json> --model MODEL [--db PATH] [--events PATH]
                                 [--agent-id ID]
    verdryx baseline <run_id> [--db PATH] [--label LABEL]
    verdryx drift --baseline ID [--db PATH] [--window N] [--threshold F]
                   [--events PATH] [--agent-id ID]
    verdryx cost-per-correct --input <ndjson-or-csv-or-parquet>
    verdryx cost-per-correct --traces <dir-of-parquet-segments>
    verdryx version

`--model stub` (eval) selects StubLLMAdapter instead of a real Anthropic
call: deterministic, no network, useful for dry-running an eval set's
structure before spending anything on it.

`baseline` is not in Verdryx's original CLI sketch, but `drift --baseline
<id>` needs some way to create the baseline it compares against, so this
adds the smallest command that can produce one: snapshot an already-stored
EvalRun's mean_score as a new Baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, NoReturn

from verdryx import slo
from verdryx.config import Config
from verdryx.costper import UNTAGGED, cost_per_outcome, load_records, load_run_records
from verdryx.drift import DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD, compute_drift
from verdryx.events import EventLog, resolve_events_path
from verdryx.graders import (
    AnthropicAdapter,
    Grader,
    LLMAdapter,
    StubLLMAdapter,
    ToolTraceGrader,
    build_graders,
)
from verdryx.models import Baseline, EvalRun, EvalSet, GraderKind, Score
from verdryx.otel import OTLPExporter, Span
from verdryx.store import Store

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _build_adapter(model: str, config: Config) -> LLMAdapter:
    """`stub` is a recognized model name: a deterministic, network-free
    adapter for dry runs and tests. Anything else is a real Anthropic model
    id, routed through TokenFuse when ANTHROPIC_BASE_URL is set. That is the
    only way to redirect it: there is deliberately no --base-url flag, so a
    proxy is a property of the environment a run happens in rather than
    something a single invocation can quietly change."""
    if model == "stub":
        return StubLLMAdapter()
    return AnthropicAdapter(
        model=model, base_url=config.anthropic_base_url, api_key=config.anthropic_api_key
    )


def _events_from_args(events_arg: str | None) -> EventLog | None:
    path = resolve_events_path(events_arg)
    return EventLog(path) if path is not None else None


def _otlp_from_config(config: Config) -> OTLPExporter | None:
    return OTLPExporter(config.otlp_endpoint) if config.otlp_endpoint else None


# ------------------------------------------------------------------
# Eval loop (the part that is not argparse plumbing; tested directly with
# a StubLLMAdapter so no network call is needed to cover it)
# ------------------------------------------------------------------


def run_eval(
    evalset: EvalSet,
    adapter: LLMAdapter,
    *,
    model: str,
    graders: dict[GraderKind, Grader | ToolTraceGrader] | None = None,
) -> EvalRun:
    """Grade every case in `evalset` and return the resulting EvalRun.

    For GraderKind.OUTCOME_TAG cases, case.prompt is treated as the raw
    outcome tag to grade (no model call: see EvalCase's docstring in
    models.py). For GraderKind.TOOL_TRACE cases, adapter.complete_with_tools
    (case.prompt, case.tools) replaces the usual adapter.complete() call: it
    returns a Completion (models.py) carrying the model's own ordered
    tool_use names alongside its tokens/cost_usd, which the tool-trace
    grader's grade_trace() (graders.py's ToolTraceGrader) scores against
    case.expected_tools; an adapter that does not implement
    complete_with_tools (e.g. a hand-rolled third-party LLMAdapter) dies
    with a clear message naming the adapter and the missing method, rather
    than failing deep in the loop with an opaque AttributeError. Every
    other case calls adapter.complete(case.prompt) first to produce the
    output that gets graded; that call's own cost_usd (real, billed usage
    for the model under evaluation, priced by AnthropicAdapter via the same
    PriceBook judge() uses -- see graders.py) is folded into Score.cost_usd
    alongside whatever the grader itself reports, so EvalRun.total_cost_usd
    reflects the run's full spend, not just an LLM_JUDGE grader's
    judge-call cost.
    """
    graders = graders if graders is not None else build_graders(judge_adapter=adapter)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(tz=UTC)
    scores: list[Score] = []

    for case in evalset.cases:
        grader = graders.get(case.grader)
        if grader is None:
            raise ValueError(
                f"no grader configured for kind {case.grader.value!r} (case_id={case.id!r})"
            )
        if case.grader == GraderKind.TOOL_TRACE:
            if not isinstance(grader, ToolTraceGrader):
                raise ValueError(
                    f"grader configured for kind 'tool_trace' is not a ToolTraceGrader "
                    f"(case_id={case.id!r})"
                )
            if not hasattr(adapter, "complete_with_tools"):
                _die(
                    f"adapter {type(adapter).__name__!r} has no complete_with_tools() method; "
                    "grading a GraderKind.TOOL_TRACE case requires an LLMAdapter that "
                    f"implements it (case_id={case.id!r})"
                )
            completion = adapter.complete_with_tools(case.prompt, case.tools or [])
            result = grader.grade_trace(case, completion)
            scores.append(
                Score(
                    case_id=case.id,
                    value=result.value,
                    tokens=completion.tokens + result.tokens,
                    cost_usd=completion.cost_usd + result.cost_usd,
                )
            )
            continue
        if case.grader == GraderKind.OUTCOME_TAG:
            output, completion_tokens, completion_cost_usd = case.prompt, 0, 0.0
        else:
            output, completion_tokens, completion_cost_usd = adapter.complete(case.prompt)
        result = grader.grade(case, output)
        scores.append(
            Score(
                case_id=case.id,
                value=result.value,
                tokens=completion_tokens + result.tokens,
                cost_usd=completion_cost_usd + result.cost_usd,
            )
        )

    finished_at = datetime.now(tz=UTC)
    return EvalRun(
        id=run_id, model=model, started_at=started_at, finished_at=finished_at, scores=scores
    )


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace, config: Config) -> None:
    try:
        evalset = EvalSet.load(args.evalset)
    except ValueError as e:
        _die(str(e))
    adapter = _build_adapter(args.model, config)
    run = run_eval(evalset, adapter, model=args.model)

    db_path = args.db or config.db_path
    with Store.open(db_path) as store:
        store.save_run(run)

    events = _events_from_args(args.events)
    if events is not None:
        for score in run.scores:
            events.emit(
                "quality_score",
                args.agent_id,
                {
                    "case_id": score.case_id,
                    "value": score.value,
                    "tokens": score.tokens,
                    "cost_usd": score.cost_usd,
                },
                run_id=run.id,
            )
        events.emit(
            "eval_run",
            args.agent_id,
            {
                "model": run.model,
                "cases": len(run.scores),
                "mean_score": run.mean_score,
                "total_tokens": run.total_tokens,
                "total_cost_usd": run.total_cost_usd,
            },
            run_id=run.id,
        )

    otlp = _otlp_from_config(config)
    if otlp is not None:
        otlp.export(
            Span(
                name="eval_run",
                run_id=run.id,
                agent_id=args.agent_id,
                attributes={
                    "model": run.model,
                    "cases": len(run.scores),
                    "mean_score": run.mean_score,
                    "total_tokens": run.total_tokens,
                    "total_cost_usd": run.total_cost_usd,
                },
                timestamp_ns=time.time_ns(),
            )
        )

    try:
        print(f"\nEval run {run.id}  (model={run.model}, db={db_path})\n")
        if not run.scores:
            print("  (no cases)\n")
            return
        for score in run.scores:
            print(f"  [{score.value:.2f}] {score.case_id}")
        print(
            f"\n  mean score: {run.mean_score:.3f}   cases: {len(run.scores)}   tokens: {run.total_tokens}\n"
        )
    finally:
        # A one-shot CLI process exits as soon as this handler returns, on
        # every path including the early return above -- an exported span
        # not yet joined here would be silently killed mid-flight along
        # with its daemon thread. See verdryx.otel's module docstring.
        if otlp is not None:
            otlp.wait()


def _cmd_baseline(args: argparse.Namespace, config: Config) -> None:
    db_path = args.db or config.db_path
    with Store.open(db_path) as store:
        run = store.load_run(args.run_id)
        if run is None:
            _die(f"no such eval run: {args.run_id!r}")
        baseline = Baseline(
            id=str(uuid.uuid4()),
            eval_run_id=run.id,
            mean_score=run.mean_score,
            created_at=datetime.now(tz=UTC),
            label=args.label or "",
        )
        store.set_baseline(baseline)

    print(f"\nBaseline {baseline.id}  (run={run.id}, mean_score={baseline.mean_score:.3f})\n")


def _cmd_drift(args: argparse.Namespace, config: Config) -> None:
    db_path = args.db or config.db_path
    with Store.open(db_path) as store:
        baseline = store.get_baseline(args.baseline)
        if baseline is None:
            _die(f"no such baseline: {args.baseline!r}")

        baseline_run = store.load_run(baseline.eval_run_id)
        if baseline_run is None:
            _die(
                f"baseline {args.baseline!r} references eval run "
                f"{baseline.eval_run_id!r}, which no longer exists in the store; "
                "refusing to compare against an unfiltered pool of runs across "
                "every model"
            )
        recent = store.list_runs(model=baseline_run.model, limit=args.window)
        if not recent:
            _die("no eval runs found to compare against the baseline")

        report = compute_drift(
            recent, baseline, threshold=args.threshold, baseline_run=baseline_run
        )

    print(f"\nDrift vs baseline {report.baseline_id}  (window={report.window})\n")
    print(f"  mean score: {report.mean_score:.3f}")
    print(f"  baseline:   {baseline.mean_score:.3f}")
    print(f"  delta:      {report.delta:+.3f}")
    if report.baseline_n:
        t_display = f"{report.t_statistic:.2f}" if report.t_statistic is not None else "n/a"
        print(f"  t-statistic: {t_display}  (n={report.baseline_n} baseline cases)")
        print(
            f"  {DEFAULT_CONFIDENCE:.0%} CI on delta: [{report.ci_low:+.3f}, {report.ci_high:+.3f}]"
        )
    print(f"  verdict:    {report.verdict}\n")

    if report.verdict == "regressed":
        events = _events_from_args(args.events)
        if events is not None:
            events.emit(
                "quality_drift",
                args.agent_id,
                {
                    "baseline_id": report.baseline_id,
                    "window": report.window,
                    "mean_score": report.mean_score,
                    "delta": report.delta,
                    "verdict": report.verdict,
                    "baseline_n": report.baseline_n,
                    "t_statistic": report.t_statistic,
                    "ci_low": report.ci_low,
                    "ci_high": report.ci_high,
                },
                run_id=recent[-1].id,
            )

    # Exported for every check regardless of verdict, unlike the governance-
    # alert quality_drift event above: a trace collector needs the full
    # on-track/regressed activity pattern, not just the regressions.
    otlp = _otlp_from_config(config)
    if otlp is not None:
        otlp.export(
            Span(
                name="quality_drift",
                run_id=recent[-1].id,
                agent_id=args.agent_id,
                attributes={
                    "baseline_id": report.baseline_id,
                    "window": report.window,
                    "mean_score": report.mean_score,
                    "delta": report.delta,
                    "verdict": report.verdict,
                    "baseline_n": report.baseline_n,
                    "t_statistic": report.t_statistic,
                    "ci_low": report.ci_low,
                    "ci_high": report.ci_high,
                },
                timestamp_ns=time.time_ns(),
            )
        )
        # A one-shot CLI process exits as soon as this handler returns; an
        # exported span not yet joined here would be silently killed mid-
        # flight along with its daemon thread. See verdryx.otel's module
        # docstring.
        otlp.wait()


def _cmd_cost_per_correct(args: argparse.Namespace, _config: Config) -> None:
    source = args.traces if args.traces else args.input
    records = load_records(source)
    report = cost_per_outcome(records)

    print(f"\nCost per outcome -- {source}\n")
    print(f"  {'OUTCOME':<20} {'COUNT':>6} {'TOTAL':>12} {'MEAN':>10}")
    # Alphabetical, but UNTAGGED always last -- matches tokenfuse-core's own
    # compute_outcomes row order (a real outcome tag before "(untagged)",
    # which would otherwise sort first on its leading "(").
    for outcome in sorted(report.by_outcome, key=lambda o: (o == UNTAGGED, o)):
        row = report.by_outcome[outcome]
        print(
            f"  {row.outcome:<20} {row.count:>6} ${row.total_cost_usd:>10.2f} ${row.mean_cost_usd:>8.4f}"
        )
    print(f"  {'-' * 52}")
    print(
        f"  {report.overall.outcome:<20} {report.overall.count:>6} "
        f"${report.overall.total_cost_usd:>10.2f} ${report.overall.mean_cost_usd:>8.4f}\n"
    )


def _cmd_slo(args: argparse.Namespace, config: Config) -> None:
    source = args.traces if args.traces else args.input
    records = load_run_records(source)
    targets = {}
    for name in slo.SLI_NAMES:
        value = getattr(args, f"target_{name}", None)
        if value is not None:
            targets[name] = value
    report = slo.compute_slo(
        records,
        identity_field=args.identity_field,
        window=args.window,
        targets=targets or None,
        confidence=args.confidence,
        min_events=args.min_events,
        good_outcomes=[o.strip() for o in args.good_outcomes.split(",") if o.strip()],
        quality_floor=args.quality_floor,
        cost_multiple=args.cost_multiple,
    )

    print(f"\nAgent error budget -- {source}")
    print(
        f"  window={report.window}  subjects={len(report.subjects)}  "
        f"runs={report.total_runs}  identity={report.identity_field}"
    )
    print(f"  ({slo.IDENTITY_NOTES[report.identity_field]})")
    if report.cost_reference_usd is not None:
        print(f"  fleet median run cost: ${report.cost_reference_usd:.6f}")
    if report.unattributed_runs:
        # Printed whether or not anything else is wrong, and near the top. A
        # fleet scores better the less of it is identified, so the coverage
        # has to arrive beside the figure rather than under it.
        share = report.unattributed_runs / report.total_runs if report.total_runs else 0.0
        print(
            f"  {report.unattributed_runs} run(s) ({share:.1%}) carry no "
            f"{report.identity_field} and are in no subject's numbers"
        )

    for subject, measurements in report.subjects.items():
        print(f"\n-- {subject}")
        for m in measurements:
            if not m.measured:
                print(f"     {m.sli:<16} not measured: {m.unmeasured_reason}")
                continue
            flag = f"  [{m.trigger.upper()}]" if m.trigger else ""
            if m.trigger in (slo.TRIGGER_EXHAUSTED, slo.TRIGGER_FAST_BURN) and not (
                slo.breach_is_established(m)
            ):
                # Said out loud, because a trigger with no event beside it
                # otherwise reads as a bug in the emitter.
                flag += " (not on the bus: the interval still covers the target)"
            print(
                f"     {m.sli:<16} {m.observed:.4f} (target {m.target}, "
                f"ci [{m.ci_low:.4f},{m.ci_high:.4f}], n={m.events})"
            )
            print(f"     {'':<16} budget {m.remaining:+.1%} left, burn {m.burn_rate:.2f}x{flag}")

    if report.blind_spots:
        print("\n  blind to:")
        for line in report.blind_spots:
            print(f"    {line}")

    payloads = slo.burn_events(report)
    print(
        f"\n  {len(payloads)} slo_burn event(s) to emit "
        f"(exhausted and fast burn only; a slow burn is reported, never alerted)"
    )
    if args.events or config.events_path:
        log = EventLog(resolve_events_path(args.events, config))
        sent = 0
        refused = 0
        for payload in payloads:
            subject = payload.pop("_subject")
            if not slo.subject_is_emittable(subject):
                refused += 1
                continue
            log.emit("slo_burn", agent_id=subject, data=payload)
            sent += 1
        print(f"  {sent} emitted to {resolve_events_path(args.events, config)}")
        if refused:
            print(
                f"  {refused} not emitted: the subject is a {report.identity_field} "
                f"and the envelope's only subject field is an agent id"
            )
    print()


def _get_version() -> str:
    try:
        from verdryx import __version__

        return __version__
    except Exception:
        return "unknown"


def _cmd_version(_args: argparse.Namespace, _config: Config) -> None:
    print(_get_version())


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verdryx", description="Quality-evaluation and drift plane for AI agents."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("eval", help="run graders over an eval set and store an EvalRun")
    p_eval.add_argument("evalset", help="path to an eval set JSON file")
    p_eval.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Anthropic model id to evaluate, or 'stub' for a network-free dry run",
    )
    p_eval.add_argument(
        "--db", default=None, metavar="PATH", help="SQLite store path (default: $VERDRYX_DB)"
    )
    p_eval.add_argument(
        "--events",
        default=None,
        metavar="PATH",
        help="opt-in NDJSON event log path (default: $VERDRYX_EVENTS_PATH, unset disables events)",
    )
    p_eval.add_argument(
        "--agent-id",
        default=None,
        metavar="ID",
        help="evaluated agent's Passport id (agent://...); required for events to be emitted",
    )

    p_baseline = sub.add_parser(
        "baseline", help="snapshot an eval run's mean score as a new baseline"
    )
    p_baseline.add_argument("run_id", help="eval run id to snapshot")
    p_baseline.add_argument("--db", default=None, metavar="PATH")
    p_baseline.add_argument("--label", default="", metavar="LABEL")

    p_drift = sub.add_parser("drift", help="compare recent eval runs against a stored baseline")
    p_drift.add_argument("--baseline", required=True, metavar="ID", help="baseline id")
    p_drift.add_argument("--db", default=None, metavar="PATH")
    p_drift.add_argument(
        "--window",
        type=int,
        default=1,
        metavar="N",
        help="number of most-recent eval runs to pool (default: 1)",
    )
    p_drift.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="F",
        help=f"minimum score drop counted as regression (default: {DEFAULT_THRESHOLD})",
    )
    p_drift.add_argument("--events", default=None, metavar="PATH")
    p_drift.add_argument("--agent-id", default=None, metavar="ID")

    p_cost = sub.add_parser(
        "cost-per-correct", help="cost-per-outcome unit economics from an outcome+cost export"
    )
    p_cost_source = p_cost.add_mutually_exclusive_group(required=True)
    p_cost_source.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "NDJSON (.ndjson/.jsonl), CSV (.csv), or Parquet (.parquet) file of "
            "{outcome, cost_usd} records"
        ),
    )
    p_cost_source.add_argument(
        "--traces",
        metavar="DIR",
        help="directory of tokenfuse Parquet trace segments (TOKENFUSE_DATA_DIR)",
    )

    p_slo = sub.add_parser(
        "slo",
        help="error budgets and burn rate over a tokenfuse trace (measurement only)",
    )
    p_slo_source = p_slo.add_mutually_exclusive_group(required=True)
    p_slo_source.add_argument(
        "--input", metavar="PATH", help="NDJSON, CSV or Parquet of per-run records"
    )
    p_slo_source.add_argument(
        "--traces",
        metavar="DIR",
        help="directory of tokenfuse Parquet trace segments (TOKENFUSE_DATA_DIR)",
    )
    p_slo.add_argument(
        "--identity-field",
        choices=list(slo.IDENTITY_FIELDS),
        default=slo.IDENTITY_AGENT_ID,
        help=(
            "which field groups a subject. agent_id is a client-supplied "
            "header and sound only for attribution; key_id is resolved by the "
            "gateway from the presented credential and is empty unless client "
            "keys are configured (default: agent_id)"
        ),
    )
    p_slo.add_argument("--window", default="28d", help="the window label to report (default: 28d)")
    p_slo.add_argument(
        "--confidence",
        type=float,
        default=slo.DEFAULT_CONFIDENCE,
        help=f"confidence for the Wilson interval (default: {slo.DEFAULT_CONFIDENCE})",
    )
    p_slo.add_argument(
        "--min-events",
        type=int,
        default=slo.DEFAULT_MIN_EVENTS,
        help=(
            "below this many eligible runs an indicator reports as not "
            f"measured rather than as a ratio (default: {slo.DEFAULT_MIN_EVENTS})"
        ),
    )
    p_slo.add_argument(
        "--good-outcomes",
        default=",".join(slo.DEFAULT_GOOD_OUTCOMES),
        help=(
            "comma-separated outcome tags that count as task success. "
            "`escalated` is deliberately NOT among the defaults: the budget "
            f"prices autonomy (default: {','.join(slo.DEFAULT_GOOD_OUTCOMES)})"
        ),
    )
    p_slo.add_argument(
        "--quality-floor",
        type=float,
        default=slo.DEFAULT_QUALITY_FLOOR,
        help=f"a run scoring at or above this meets the floor (default: {slo.DEFAULT_QUALITY_FLOOR})",
    )
    p_slo.add_argument(
        "--cost-multiple",
        type=float,
        default=slo.DEFAULT_COST_MULTIPLE,
        help=(
            "a run costing more than this multiple of the FLEET median fails "
            f"cost discipline (default: {slo.DEFAULT_COST_MULTIPLE})"
        ),
    )
    for name in slo.SLI_NAMES:
        p_slo.add_argument(
            f"--target-{name.replace('_', '-')}",
            dest=f"target_{name}",
            type=float,
            default=None,
            help=f"objective for {name} (default: {slo.DEFAULT_TARGET})",
        )
    p_slo.add_argument(
        "--events",
        metavar="PATH",
        default=None,
        help="append slo_burn events here (also VERDRYX_EVENTS_PATH)",
    )

    sub.add_parser("version", help="print the verdryx version")

    return parser


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[argparse.Namespace, Config], None]] = {
    "eval": _cmd_eval,
    "baseline": _cmd_baseline,
    "drift": _cmd_drift,
    "cost-per-correct": _cmd_cost_per_correct,
    "slo": _cmd_slo,
    "version": _cmd_version,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args: Any = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    config = Config.from_env()
    _HANDLERS[args.cmd](args, config)


if __name__ == "__main__":
    main()
