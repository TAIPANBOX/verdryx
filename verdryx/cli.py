"""CLI entry point for Verdryx.

Usage::

    verdryx eval <evalset.json> --model MODEL [--db PATH] [--events PATH]
                                 [--agent-id ID]
                                 [--typed-url URL [--typed-key-file PATH]
                                  [--typed-template ID] [--typed-timeout SECONDS]]
    verdryx baseline <run_id> [--db PATH] [--label LABEL]
    verdryx drift --baseline ID [--db PATH] [--window N] [--threshold F]
                   [--events PATH] [--agent-id ID]
    verdryx cost-per-correct --input <ndjson-or-csv-or-parquet>
    verdryx cost-per-correct --traces <dir-of-parquet-segments>
    verdryx slo --traces <dir-of-parquet-segments> [--scores-db PATH] [--db PATH]
    verdryx findings list [--db PATH] [--status open|closed|all] [--agent-id ID]
    verdryx findings show <id-or-prefix> [--db PATH]
    verdryx findings fix <id-or-prefix> --note TEXT [--version V] [--db PATH]
    verdryx version

`--model stub` (eval) selects StubLLMAdapter instead of a real Anthropic
call: deterministic, no network, useful for dry-running an eval set's
structure before spending anything on it.

`baseline` is not in Verdryx's original CLI sketch, but `drift --baseline
<id>` needs some way to create the baseline it compares against, so this
adds the smallest command that can produce one: snapshot an already-stored
EvalRun's mean_score as a new Baseline.

`--typed-url` is the ONLY thing that enables GraderKind.TYPED (asking
typryx, an optional separate service, for a typed verdict): no environment
variable enables it on its own, matching CLAUDE.md invariant 5's own logic
applied to a second external caller. The credential comes from
`--typed-key-file`, or else `$VERDRYX_TYPRYX_KEY_FILE` -- both a file path,
never the key on the command line, which would show in `ps`. See
graders.TypedGrader for what a typed case sends and gets back.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, NoReturn

from verdryx import findings as findings_mod
from verdryx import slo
from verdryx.config import ENV_TYPRYX_KEY_FILE, Config
from verdryx.costper import UNTAGGED, cost_per_outcome, load_records, load_run_records
from verdryx.drift import DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD, compute_drift
from verdryx.events import EventLog, resolve_events_path
from verdryx.graders import (
    AnthropicAdapter,
    Grader,
    LLMAdapter,
    StubLLMAdapter,
    ToolTraceGrader,
    TypedGrader,
    TypedUnanswered,
    TypryxClient,
    TypryxError,
    build_graders,
)
from verdryx.models import (
    FINDING_STATUSES,
    STATUS_CLOSED,
    STATUS_FIX_PENDING,
    STATUS_OPEN,
    Baseline,
    EvalRun,
    EvalSet,
    GraderKind,
    Score,
    Unanswered,
)
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


def _report_unwritten_events(events: EventLog | None) -> None:
    """Say so when an event log the operator switched on took nothing.

    `EventLog.emit` skips an event that has no agent id rather than invent
    one, and counts it. That count was read by no command, so `--events` with
    no agent id wrote an empty bus and printed a normal-looking report: a
    regression went unannounced. The skip is still right, the silence is not.
    On stderr, so a script reading the report is not disturbed.
    """
    if events is not None and events.skipped_empty_agent_id:
        print(
            f"warning: {events.skipped_empty_agent_id} event(s) not written to {events.path}: "
            "no agent id (pass --agent-id)",
            file=sys.stderr,
        )


def _positive_seconds(text: str) -> float:
    """argparse type for a wait in seconds: a finite number above zero."""
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return value


def _build_typed_client(args: argparse.Namespace, config: Config) -> TypryxClient | None:
    """`--typed-url` is the only thing that constructs a TypryxClient: no
    environment variable enables the typed grader on its own (CLAUDE.md
    invariant 5, checks 4 and 5 of scripts/no-paid-by-default.sh: its AST check
    holds both this function being the one construction site and this
    `if args.typed_url` guard). Returns None when the flag is absent, so
    build_graders() never registers GraderKind.TYPED for an ordinary run.
    """
    if args.typed_url:
        key_file = args.typed_key_file or config.typryx_key_file
        if not key_file:
            _die(
                "--typed-url was given with no credential: pass --typed-key-file PATH or set "
                f"${ENV_TYPRYX_KEY_FILE} to a file holding the typryx key. The key is never "
                "accepted on the command line, which would show in `ps`."
            )
        try:
            key = _read_key_file(key_file)
        except OSError as e:
            _die(f"--typed-key-file {key_file!r}: cannot read: {e}")
        kwargs: dict[str, Any] = {}
        if args.typed_template:
            kwargs["template"] = args.typed_template
        if args.typed_timeout is not None:
            kwargs["timeout"] = args.typed_timeout
        return TypryxClient(args.typed_url, key, **kwargs)
    return None


def _read_key_file(path: str) -> str:
    """Read and strip a credential file. A dedicated function, not an
    inline `.read_text().strip()`, so the one place this repo reads a
    typryx key from disk is easy to find and audit, mirroring how
    TypryxClient is the one place it is sent over the wire."""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ------------------------------------------------------------------
# Eval loop (the part that is not argparse plumbing; tested directly with
# a StubLLMAdapter so no network call is needed to cover it)
# ------------------------------------------------------------------


def run_eval(
    evalset: EvalSet,
    adapter: LLMAdapter,
    *,
    model: str,
    agent_id: str | None = None,
    agent_version: str | None = None,
    graders: dict[GraderKind, Grader | ToolTraceGrader] | None = None,
    typed_client: TypryxClient | None = None,
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

    `agent_id` is whose run this is, and it goes onto the EvalRun rather than
    only onto the events the run emits. That is what lets `verdryx slo` join a
    score back to a fleet subject at all: the eval store and a tokenfuse trace
    share no run identity, so the subject is the only join there is. None when
    the caller did not say, and never inferred from anything else.

    `typed_client` is passed straight to build_graders() when `graders` is
    not given directly; if it is given, it is threaded through anyway,
    setting the resulting TypedGrader's run_id to this run's own id (see
    TypedGrader.run_id) so every /v1/ask this run makes carries it and
    typryx's own record joins to this EvalRun. GraderKind.TYPED cases in an
    eval set run with no typed_client die with the same "no grader
    configured" error every other missing grader kind dies with (see
    below), naming --typed-url as the flag that enables it.

    A GraderKind.TYPED case whose grader answers unanswered raises
    TypedUnanswered (graders.py); THIS function catches it, per case, and
    appends a models.Unanswered to the returned EvalRun's own `unanswered`
    list instead of letting it end the run -- see that exception's own
    docstring for the decision and why it changed. TypryxError (a refusal,
    an unreachable typryx) is a different fact -- an infrastructure
    failure, not a verdict -- and is NOT caught here: it propagates out of
    this function, and verdryx.cli._cmd_eval is what turns it into a clean
    CLI death, before the store is ever opened.
    """
    graders = (
        graders
        if graders is not None
        else build_graders(judge_adapter=adapter, typed_client=typed_client)
    )
    run_id = str(uuid.uuid4())
    typed_grader = graders.get(GraderKind.TYPED)
    if isinstance(typed_grader, TypedGrader):
        typed_grader.run_id = run_id
    started_at = datetime.now(tz=UTC)
    scores: list[Score] = []
    unanswered: list[Unanswered] = []

    for case in evalset.cases:
        grader = graders.get(case.grader)
        if grader is None:
            hint = " (enable it with --typed-url)" if case.grader == GraderKind.TYPED else ""
            raise ValueError(
                f"no grader configured for kind {case.grader.value!r}{hint} (case_id={case.id!r})"
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
        try:
            result = grader.grade(case, output)
        except TypedUnanswered as e:
            # Counted apart, not scored: see TypedUnanswered's own
            # docstring and models.Unanswered. Only a TypedGrader can ever
            # raise this, so this except is reachable for GraderKind.TYPED
            # cases alone -- every other grader kind's grade() call passes
            # straight through unchanged.
            unanswered.append(Unanswered(case_id=e.case_id, answer_id=e.answer_id, reason=e.reason))
            continue
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
        id=run_id,
        model=model,
        started_at=started_at,
        finished_at=finished_at,
        scores=scores,
        # Carried through so the run KEEPS the subject it is already being
        # described by. `_cmd_eval` has stamped this same value onto every
        # event and span of the run since the event log existed, and dropped
        # it on the way to the store, which is why `slo`'s quality_floor had
        # nothing to join on. Never defaulted to the model or to anything
        # else: a run nobody attributed stays unattributed.
        agent_id=agent_id,
        agent_version=agent_version,
        unanswered=unanswered,
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
    typed_client = _build_typed_client(args, config)
    try:
        run = run_eval(
            evalset,
            adapter,
            model=args.model,
            agent_id=args.agent_id,
            agent_version=args.agent_version,
            typed_client=typed_client,
        )
    except TypryxError as e:
        # A refusal (its hourly cap, a bad key, an unknown template) or a
        # typryx nobody could reach is an infrastructure failure, not a
        # verdict -- run_eval does not catch this, so it is where it
        # becomes a clean CLI death instead of a raw traceback;
        # TypryxError's message never carries the key. Nothing is saved:
        # we die before ever reaching Store.open() below.
        #
        # An unanswered verdict (TypedUnanswered) is NOT here: run_eval
        # catches it per case and continues, so it reaches this point as
        # part of an ordinary, complete EvalRun -- see EvalRun.unanswered
        # and the printout below.
        _die(str(e))

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
                # The envelope's `data` object is open (agent-passport
                # SPEC.md Sec 6.2 registers only the type name and severity
                # for the "verdryx" row, no closed field list for eval_run
                # itself; tests/fixtures/agent-event.v0.2.schema.json's
                # `data` is `additionalProperties: true`), so this is an
                # addition, not a change to a fixed shape. 0 for every run
                # that has no unanswered cases, never omitted.
                "unanswered": len(run.unanswered),
            },
            run_id=run.id,
        )
        _report_unwritten_events(events)

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
        subject = f", agent={run.agent_id}" if run.agent_id else ", no agent id"
        if run.agent_version:
            subject += f", version={run.agent_version}"
        print(f"\nEval run {run.id}  (model={run.model}{subject}, db={db_path})\n")
        if not run.scores and not run.unanswered:
            print("  (no cases)\n")
            return
        for score in run.scores:
            print(f"  [{score.value:.2f}] {score.case_id}")
        if run.scores:
            print(
                f"\n  mean score: {run.mean_score:.3f}   cases: {len(run.scores)}   tokens: {run.total_tokens}\n"
            )
        else:
            # Invariant 8 (CLAUDE.md): an unmeasured indicator is never a
            # zero. Every case in this run came back unanswered, so there
            # is no mean to report -- printing 0.000 here would read as a
            # confident "every case scored zero", the opposite fact from
            # "nothing could be asked at all".
            print(f"\n  mean score: unmeasured (0 of {run.cases_asked} answered)\n")
        if run.unanswered:
            reason_counts: dict[str, int] = {}
            for u in run.unanswered:
                reason_counts[u.reason] = reason_counts.get(u.reason, 0) + 1
            reasons = ", ".join(f"{reason}: {count}" for reason, count in reason_counts.items())
            print(f"  unanswered: {len(run.unanswered)} of {run.cases_asked} asked ({reasons})")
            for u in run.unanswered:
                print(f"    {u.case_id}  (answer_id={u.answer_id})")
            print()
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
        if not run.scores:
            # A baseline snapshots mean_score, which is 0.0 for zero scored
            # cases (models.EvalRun.mean_score) -- the same float a perfect
            # run that scored every case 0.0 would produce. Snapshotting
            # that as a baseline would be a fabricated number: invariant 8
            # (CLAUDE.md), an unmeasured indicator is never a zero, applies
            # to a baseline exactly as it does to the run it comes from.
            unanswered_note = f" ({len(run.unanswered)} unanswered)" if run.unanswered else ""
            _die(
                f"eval run {run.id!r} has zero scored cases{unanswered_note}: a baseline of "
                "an unmeasured mean would be a fabricated number"
            )
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
        # `list_runs` is most recent first, so the run this check is about, the
        # one that just ran, is the head of the window. It was read off the end,
        # which named the OLDEST run of a window wider than one.
        newest = recent[0]
        # Whose drift this is. `--agent-id` wins; without it the agent the
        # newest run was evaluated as, which `verdryx eval --agent-id` already
        # stored. Never taken from anywhere else (see EventLog.emit).
        agent_id = args.agent_id or newest.agent_id

        try:
            report = compute_drift(
                recent, baseline, threshold=args.threshold, baseline_run=baseline_run
            )
            # Written down and decided here, inside the store's block, from the
            # stores alone: no model, no network. A fix registered earlier is
            # judged by this same call, whatever the window's own verdict is.
            finding_lines = findings_mod.process_drift(
                store,
                baseline=baseline,
                baseline_run=baseline_run,
                recent=recent,
                threshold=args.threshold,
                agent_id=agent_id,
                now=datetime.now(tz=UTC),
            )
        except ValueError as e:
            # compute_drift pools Scores per CASE across the window, so a
            # run that came back fully unanswered (zero Scores) already
            # contributes nothing to the mean rather than entering it as a
            # 0.0 -- see test_drift.py's
            # test_compute_drift_ignores_empty_runs_mixed_with_scored_ones.
            # This branch is for when EVERY run in the window is like that:
            # there is then no case anywhere to average, and compute_drift
            # already refuses ("no scores found ...") rather than letting a
            # mean read as 0.0. This is where that refusal becomes a clean
            # CLI death instead of a raw traceback.
            _die(str(e))

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
    for line in finding_lines:
        print(f"  {line}")
    if finding_lines:
        print()

    if report.verdict == "regressed":
        events = _events_from_args(args.events)
        if events is not None:
            events.emit(
                "quality_drift",
                agent_id,
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
                run_id=newest.id,
            )
            _report_unwritten_events(events)

    # Exported for every check regardless of verdict, unlike the governance-
    # alert quality_drift event above: a trace collector needs the full
    # on-track/regressed activity pattern, not just the regressions.
    otlp = _otlp_from_config(config)
    if otlp is not None:
        otlp.export(
            Span(
                name="quality_drift",
                run_id=newest.id,
                agent_id=agent_id,
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


def _scores_from_store(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """The eval store's scores, keyed by the subject each run was stamped with.

    Refuses outright when the report is grouped by `key_id`. That grouping is
    the sound one for anything enforced, because the gateway resolves a
    credential server-side and a caller cannot move off it, but the eval store
    records an agent and no credential: there is nothing to join on. Reporting
    quality_floor as unmeasured instead would look identical to supplying no
    scores at all, and an operator who passed the flag would have no way to
    tell that it did nothing.
    """
    if not args.scores_db:
        return None
    if args.identity_field != slo.IDENTITY_AGENT_ID:
        _die(
            f"--scores-db cannot be joined to a report grouped by "
            f"{args.identity_field!r}. Scores are stamped with the agent id "
            f"`verdryx eval --agent-id` was given; the eval store records no "
            f"credential, so there is nothing to attribute them to. Group by "
            f"{slo.IDENTITY_AGENT_ID} for this join, or drop --scores-db."
        )
    with Store.open(args.scores_db) as store:
        return store.score_records()


def _record_slo_findings(
    args: argparse.Namespace,
    report: Any,
    records: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    options: dict[str, Any],
) -> None:
    """Write the report's established breaches down as findings, and decide the
    fixes registered against them. Needs a store: with none given, an
    established breach says it was not recorded and how to record it, rather
    than leaving the operator to wonder why `findings list` is empty."""
    breaches = slo.established_breaches(report)
    if not args.db:
        if breaches:
            print(
                f"\n  {len(breaches)} established breach(es) not recorded as a finding: "
                "pass --db PATH to keep them"
            )
        return

    def recompute(rs: Any, sc: Any) -> Any:
        return slo.compute_slo(rs, scores=sc, **options)

    with Store.open(args.db) as store:
        lines = findings_mod.process_slo(
            store,
            report,
            records=records,
            scores=scores,
            recompute=recompute,
            now=datetime.now(tz=UTC),
        )
    if lines:
        print()
        for line in lines:
            print(f"  {line}")


def _cmd_slo(args: argparse.Namespace, config: Config) -> None:
    source = args.traces if args.traces else args.input
    records = load_run_records(source)
    scores = _scores_from_store(args)
    targets = {}
    for name in slo.SLI_NAMES:
        value = getattr(args, f"target_{name}", None)
        if value is not None:
            targets[name] = value
    options: dict[str, Any] = {
        "identity_field": args.identity_field,
        "window": args.window,
        "targets": targets or None,
        "confidence": args.confidence,
        "min_events": args.min_events,
        "good_outcomes": [o.strip() for o in args.good_outcomes.split(",") if o.strip()],
        "quality_floor": args.quality_floor,
        "cost_multiple": args.cost_multiple,
    }
    report = slo.compute_slo(records, scores=scores, **options)

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
    if report.total_scores:
        print(f"  {report.total_scores} score(s) joined from {args.scores_db}")
    if report.unattributed_scores:
        # The same sentence for the same reason, about the other input. A
        # fleet well identified on the gateway can be badly identified in its
        # eval store, so one line covering both would hide which.
        share = report.unattributed_scores / report.total_scores if report.total_scores else 0.0
        print(
            f"  {report.unattributed_scores} score(s) ({share:.1%}) come from a run "
            f"evaluated with no --agent-id and are in no subject's numbers"
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

    _record_slo_findings(args, report, records, scores or [], options)

    payloads = slo.burn_events(report)
    print(
        f"\n  {len(payloads)} slo_burn event(s) to emit "
        f"(exhausted and fast burn only; a slow burn is reported, never alerted)"
    )
    # Resolved once, and with the one argument the function takes. This block
    # asked the same question twice and got the arity wrong both times: it
    # guarded on `args.events or config.events_path` and then called
    # `resolve_events_path(args.events, config)`, so every invocation carrying
    # `--events` died with a TypeError immediately after printing the count of
    # events it was about to send. Nothing caught it, because no test had ever
    # passed the flag; it was found by running the command against a real
    # tokenfuse trace on 2026-08-26.
    #
    # The guard is gone with it, and not only because it was redundant.
    # `config` snapshots the environment at startup and `resolve_events_path`
    # reads it live, so the two could disagree about whether events are on at
    # all, and the shape of the disagreement is a command that says it will
    # emit and then does not.
    events_path = resolve_events_path(args.events)
    if events_path is not None:
        log = EventLog(events_path)
        sent = 0
        refused = 0
        for payload in payloads:
            subject = payload.pop("_subject")
            if not slo.subject_is_emittable(subject):
                refused += 1
                continue
            log.emit("slo_burn", agent_id=subject, data=payload)
            sent += 1
        print(f"  {sent} emitted to {events_path}")
        if refused:
            print(
                f"  {refused} not emitted: the subject is a {report.identity_field} "
                f"and the envelope's only subject field is an agent id"
            )
    print()


def _print_finding_row(f: Any) -> None:
    kind = f"{f.kind}:{f.sli}" if f.sli else f.kind
    print(
        f"  {f.id[:8]}  {f.status:<11}  {kind:<28}  {f.agent_id or '(no agent id)'}  "
        f"seen {f.seen_count}x, last {f.last_seen_at.isoformat(timespec='seconds')}"
    )


def _print_finding(f: Any) -> None:
    print(f"\nFinding {f.id}\n")
    print(f"  status:    {f.status}")
    print(f"  kind:      {f.kind}" + (f" ({f.sli})" if f.sli else ""))
    print(f"  agent:     {f.agent_id or '(no agent id)'}")
    print(f"  window:    {f.window}")
    print(
        f"  opened:    {f.opened_at.isoformat(timespec='seconds')}   "
        f"last seen: {f.last_seen_at.isoformat(timespec='seconds')}   seen {f.seen_count}x"
    )
    if f.closed_at:
        print(f"  closed:    {f.closed_at.isoformat(timespec='seconds')}")
    print(f"  suspect:   {f.suspect_note}")
    print("  numbers:")
    for name, value in f.detail.items():
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"    {name}: {shown}")
    print(f"  evidence ({len(f.evidence)}):")
    for item in f.evidence:
        print(f"    {item}")
    if f.fixes:
        print("  fixes:")
        for n, fx in enumerate(f.fixes, 1):
            version = f", version {fx.version}" if fx.version else ""
            print(
                f"    {n}. [{fx.status}] {fx.note}  (id {fx.id[:8]}{version}, registered "
                f"{fx.registered_at.isoformat(timespec='seconds')})"
            )
            detail = fx.result.get("summary") or fx.result.get("reason")
            if detail:
                print(f"       {detail}")
    print()


def _cmd_findings(args: argparse.Namespace, config: Config) -> None:
    if args.findings_cmd is None:
        print("usage: verdryx findings {list,show,fix} ...", file=sys.stderr)
        sys.exit(1)
    db_path = args.db or config.db_path
    with Store.open(db_path) as store:
        if args.findings_cmd == "list":
            statuses = {
                "open": (STATUS_OPEN, STATUS_FIX_PENDING),
                "closed": (STATUS_CLOSED,),
                "all": tuple(FINDING_STATUSES),
            }[args.status]
            rows = store.list_findings(statuses, args.agent_id)
            print(f"\nFindings ({args.status}) -- {db_path}\n")
            if not rows:
                print("  (none)")
            for f in rows:
                _print_finding_row(f)
            print()
            return
        try:
            finding_id = store.resolve_finding_id(args.finding)
        except LookupError as e:
            _die(str(e))
        if args.findings_cmd == "show":
            finding = store.get_finding(finding_id)
            assert finding is not None
            _print_finding(finding)
            return
        try:
            fix = findings_mod.register_fix(
                store,
                finding_id,
                args.note,
                version=args.version,
                now=datetime.now(tz=UTC),
            )
        except findings_mod.FixError as e:
            _die(str(e))
    print(f"\nFix {fix.id[:8]} registered for finding {finding_id[:8]}: the next run decides it\n")


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
    p_eval.add_argument(
        "--agent-version",
        default=None,
        metavar="LABEL",
        help=(
            "which build of the agent this run is (a git sha, a prompt version, any label). "
            "A finding names the version that changed, and a registered fix is tested only "
            "by runs of the version it names"
        ),
    )
    p_eval.add_argument(
        "--typed-url",
        default=None,
        metavar="URL",
        help=(
            "typryx base URL (e.g. http://127.0.0.1:4320); the ONLY thing that enables "
            "GraderKind.TYPED cases. typryx may run a paid backend, so this is opt-in "
            "and never enabled by an environment variable alone"
        ),
    )
    p_eval.add_argument(
        "--typed-key-file",
        default=None,
        metavar="PATH",
        help=(
            f"file holding the typryx credential (default: ${ENV_TYPRYX_KEY_FILE}); "
            "never accepted on the command line, which would show in `ps`"
        ),
    )
    p_eval.add_argument(
        "--typed-template",
        default=None,
        metavar="ID",
        help="typryx template id to ask (default: eval.outcome_met)",
    )
    p_eval.add_argument(
        "--typed-timeout",
        type=_positive_seconds,
        default=None,
        metavar="SECONDS",
        help=(
            "how long to wait for each typryx answer (default: 30); a model that is slow to "
            "load needs more, and typryx's own TYPRYX_TIMEOUT_MS should stay below it"
        ),
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
        "--scores-db",
        metavar="PATH",
        default=None,
        help=(
            "SQLite eval store to take per-run scores from, so quality_floor "
            "can be computed. Joined on the agent id `verdryx eval --agent-id` "
            "stamped onto each run: the trace and the eval store share no run "
            "identity, so the subject is the only join there is"
        ),
    )
    p_slo.add_argument(
        "--events",
        metavar="PATH",
        default=None,
        help="append slo_burn events here (also VERDRYX_EVENTS_PATH)",
    )

    p_slo.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=(
            "SQLite store to record findings in, and to read the fixes registered against "
            "them from. Without it an established breach is reported and not recorded; "
            "--scores-db stays read-only"
        ),
    )

    p_findings = sub.add_parser(
        "findings", help="list, read and register fixes for the failures drift and slo recorded"
    )
    findings_sub = p_findings.add_subparsers(dest="findings_cmd")
    p_f_list = findings_sub.add_parser("list", help="findings, most recently seen first")
    p_f_list.add_argument("--db", default=None, metavar="PATH")
    p_f_list.add_argument(
        "--status",
        choices=["open", "closed", "all"],
        default="open",
        help="open also lists findings with a fix pending (default: open)",
    )
    p_f_list.add_argument("--agent-id", default=None, metavar="ID")
    p_f_show = findings_sub.add_parser("show", help="one finding in full")
    p_f_show.add_argument("finding", metavar="ID", help="a finding id, or a unique prefix of one")
    p_f_show.add_argument("--db", default=None, metavar="PATH")
    p_f_fix = findings_sub.add_parser(
        "fix", help="register a fix; the next runs decide whether it worked"
    )
    p_f_fix.add_argument("finding", metavar="ID", help="a finding id, or a unique prefix of one")
    p_f_fix.add_argument("--note", required=True, metavar="TEXT", help="what was changed")
    p_f_fix.add_argument(
        "--version",
        default=None,
        metavar="LABEL",
        help="the agent version the fix ships in; only eval runs of it can test the fix",
    )
    p_f_fix.add_argument("--db", default=None, metavar="PATH")

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
    "findings": _cmd_findings,
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
