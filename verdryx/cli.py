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
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from verdryx.config import Config
from verdryx.costper import cost_per_outcome, load_records
from verdryx.drift import DEFAULT_THRESHOLD, compute_drift
from verdryx.events import EventLog, resolve_events_path
from verdryx.graders import (
    AnthropicAdapter,
    Grader,
    LLMAdapter,
    StubLLMAdapter,
    build_graders,
)
from verdryx.models import Baseline, EvalRun, EvalSet, GraderKind, Score
from verdryx.store import Store

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _build_adapter(model: str, config: Config) -> LLMAdapter:
    """`stub` is a recognized model name: a deterministic, network-free
    adapter for dry runs and tests. Anything else is a real Anthropic model
    id, routed through TokenFuse if ANTHROPIC_BASE_URL / --base-url-style
    config is set."""
    if model == "stub":
        return StubLLMAdapter()
    return AnthropicAdapter(
        model=model, base_url=config.anthropic_base_url, api_key=config.anthropic_api_key
    )


def _events_from_args(events_arg: str | None) -> EventLog | None:
    path = resolve_events_path(events_arg)
    return EventLog(path) if path is not None else None


# ------------------------------------------------------------------
# Eval loop (the part that is not argparse plumbing; tested directly with
# a StubLLMAdapter so no network call is needed to cover it)
# ------------------------------------------------------------------


def run_eval(
    evalset: EvalSet,
    adapter: LLMAdapter,
    *,
    model: str,
    graders: dict[GraderKind, Grader] | None = None,
) -> EvalRun:
    """Grade every case in `evalset` and return the resulting EvalRun.

    For GraderKind.OUTCOME_TAG cases, case.prompt is treated as the raw
    outcome tag to grade (no model call: see EvalCase's docstring in
    models.py). Every other case calls adapter.complete(case.prompt) first
    to produce the output that gets graded.
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
        if case.grader == GraderKind.OUTCOME_TAG:
            output, completion_tokens = case.prompt, 0
        else:
            output, completion_tokens = adapter.complete(case.prompt)
        result = grader.grade(case, output)
        scores.append(
            Score(
                case_id=case.id,
                value=result.value,
                tokens=completion_tokens + result.tokens,
                cost_usd=result.cost_usd,
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
    evalset = EvalSet.load(args.evalset)
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

    print(f"\nEval run {run.id}  (model={run.model}, db={db_path})\n")
    if not run.scores:
        print("  (no cases)\n")
        return
    for score in run.scores:
        print(f"  [{score.value:.2f}] {score.case_id}")
    print(
        f"\n  mean score: {run.mean_score:.3f}   cases: {len(run.scores)}   tokens: {run.total_tokens}\n"
    )


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
        model_filter = baseline_run.model if baseline_run is not None else None
        recent = store.list_runs(model=model_filter, limit=args.window)
        if not recent:
            _die("no eval runs found to compare against the baseline")

        report = compute_drift(recent, baseline, threshold=args.threshold)

    print(f"\nDrift vs baseline {report.baseline_id}  (window={report.window})\n")
    print(f"  mean score: {report.mean_score:.3f}")
    print(f"  baseline:   {baseline.mean_score:.3f}")
    print(f"  delta:      {report.delta:+.3f}")
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
                },
                run_id=recent[-1].id,
            )


def _cmd_cost_per_correct(args: argparse.Namespace, _config: Config) -> None:
    source = args.traces if args.traces else args.input
    records = load_records(source)
    report = cost_per_outcome(records)

    print(f"\nCost per outcome -- {source}\n")
    print(f"  {'OUTCOME':<20} {'COUNT':>6} {'TOTAL':>12} {'MEAN':>10}")
    for outcome in sorted(report.by_outcome):
        row = report.by_outcome[outcome]
        print(
            f"  {row.outcome:<20} {row.count:>6} ${row.total_cost_usd:>10.2f} ${row.mean_cost_usd:>8.4f}"
        )
    print(f"  {'-' * 52}")
    print(
        f"  {report.overall.outcome:<20} {report.overall.count:>6} "
        f"${report.overall.total_cost_usd:>10.2f} ${report.overall.mean_cost_usd:>8.4f}\n"
    )


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
