"""The bake-off driver: grades the same dataset with several judges through
verdryx's own `run_eval`, times every grade call, and writes a report.

Two grading shapes, both driven the same way as the scratchpad driver
drive2.py: `Replay` (an `LLMAdapter` that returns the dataset's own recorded
`final_answer`s in case order, so the "model under evaluation" is fixed by
the dataset and only the JUDGE varies) plus `run_eval`.

- `kind="typed"` (qwen, jev, stub): `GraderKind.TYPED` cases, a
  `TypryxClient` pointed at one already-running typryx process.
- `kind="llm_judge"` (claude): `GraderKind.LLM_JUDGE` cases, an
  `AnthropicAdapter`, and a rubric equivalent to typryx's
  `eval.outcome_met` template (see CLAUDE_RUBRIC below) so the judges are
  answering the same question.

The claude spend guard (`require_claude_confirmation`) never touches the
network either way: it estimates a cost from verdryx's own `PriceBook` and
either prints the estimate and continues (confirmed) or prints it and
raises SystemExit(1) (not confirmed) before any `AnthropicAdapter` is
constructed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from examples.bakeoff import metrics as m
from examples.bakeoff.dataset import family_sizes
from verdryx.cli import run_eval
from verdryx.graders import (
    AnthropicAdapter,
    LLMJudgeGrader,
    TypedGrader,
    TypryxClient,
)
from verdryx.models import EvalCase, EvalSet, GraderKind
from verdryx.pricing import PriceBook

# ------------------------------------------------------------------
# The rubric a claude LLM_JUDGE grades against: the same question typryx's
# eval.outcome_met template (examples/templates/eval.outcome_met.json in the
# typryx repo) asks, in prose instead of typed criteria, so the two judges
# answer the same question rather than two different ones.
# ------------------------------------------------------------------

CLAUDE_RUBRIC = (
    "The agent's final answer achieves the task described in the prompt. "
    "Score 1 if the final answer fully achieves what the task asked for. "
    "Score 0 if the final answer does not achieve what the task asked for, "
    "or only partly does."
)

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

#: A rough estimate, not a bill: the judge system prompt (graders.py's
#: _JUDGE_SYSTEM_PROMPT, ~80 tokens) plus CLAUDE_RUBRIC (~45 tokens) plus a
#: short task/final_answer pair from this harness's own dataset (arithmetic,
#: routing, and format tasks here all run well under 100 tokens). Padded up
#: for safety since an estimate that runs low defeats the point of asking
#: before spending. graders.py's judge() sets max_tokens=16 on the call, so
#: the output-token estimate is exact, not approximate.
ESTIMATED_INPUT_TOKENS_PER_CASE = 220
ESTIMATED_OUTPUT_TOKENS_PER_CASE = 16

DEFAULT_TYPED_TEMPLATE = "eval.outcome_met"


# ------------------------------------------------------------------
# The claude spend guard. Pure: no network, no AnthropicAdapter, whether or
# not it refuses.
# ------------------------------------------------------------------


def estimate_claude_cost(
    n: int, model: str, price_book: PriceBook | None = None
) -> tuple[int, int, float]:
    price_book = price_book if price_book is not None else PriceBook.default()
    input_tokens = ESTIMATED_INPUT_TOKENS_PER_CASE * n
    output_tokens = ESTIMATED_OUTPUT_TOKENS_PER_CASE * n
    cost = price_book.price(model, input_tokens, output_tokens)
    return input_tokens, output_tokens, cost


def claude_spend_message(n: int, model: str, price_book: PriceBook | None = None) -> str:
    input_tokens, output_tokens, cost = estimate_claude_cost(n, model, price_book)
    return (
        f"claude judge estimate: n={n} cases, ~{ESTIMATED_INPUT_TOKENS_PER_CASE} estimated "
        f"input tokens/case (judge system prompt + rubric + task + final_answer), model={model}, "
        f"price from verdryx.pricing.PriceBook.default() -> "
        f"~{input_tokens} input + {output_tokens} output tokens total, "
        f"~${cost:.4f} USD. This is an estimate, not the settled bill."
    )


def spend_confirmed() -> bool:
    """Whether BAKEOFF_CONFIRM_SPEND=yes is set in the environment. A
    function, not an inline os.environ check, so a test can call it after
    monkeypatching os.environ without also invoking the CLI."""
    return os.environ.get("BAKEOFF_CONFIRM_SPEND") == "yes"


def require_claude_confirmation(n: int, model: str, price_book: PriceBook | None = None) -> None:
    """Print the estimate; raise SystemExit(1) if BAKEOFF_CONFIRM_SPEND is
    not "yes". Never constructs an AnthropicAdapter and never imports
    `anthropic`: this function alone is what a test can call to prove the
    guard refuses, with no network reachable either way.

    Prints via plain `print()` (sys.stdout looked up at call time), not a
    keyword default bound to `sys.stdout` at function-definition time: the
    latter would capture the *original* stdout object before pytest's
    `capsys` ever swaps it out, and every print here would silently miss
    the test's capture.
    """
    print(claude_spend_message(n, model, price_book))
    if not spend_confirmed():
        print('refusing: BAKEOFF_CONFIRM_SPEND is not "yes". No Anthropic call was made.')
        raise SystemExit(1)
    print("BAKEOFF_CONFIRM_SPEND=yes: proceeding.")


# ------------------------------------------------------------------
# Grading plumbing shared by every judge kind.
# ------------------------------------------------------------------


class Replay:
    """LLMAdapter returning this dataset's own recorded final_answer values
    in case order (the drive2.py pattern): the model under evaluation is
    fixed by the dataset, right or a plausible wrong one by construction
    (see dataset.py); only the JUDGE varies between bake-off runs."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)

    def complete(self, prompt: str) -> tuple[str, int, float]:
        return self.answers.pop(0), 0, 0.0

    def judge(self, *args: object) -> tuple[float, int, float]:
        raise AssertionError("Replay is the model under evaluation, not a judge")

    def complete_with_tools(self, *args: object) -> object:
        raise AssertionError("the bake-off never uses GraderKind.TOOL_TRACE")


class TimingGrader:
    """Wraps a Grader's grade() call with stdlib time.perf_counter,
    recording one latency per case_id. Deliberately NOT a TypedGrader
    subclass: run_eval's own `isinstance(typed_grader, TypedGrader)` check
    (verdryx/cli.py) then finds nothing to auto-assign a run_id to, which is
    fine here because grade_judge() below sets the inner TypedGrader's
    run_id itself before the run starts."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.latencies: dict[str, float] = {}

    def grade(self, case: EvalCase, output: str) -> Any:
        t0 = time.perf_counter()
        try:
            return self.inner.grade(case, output)
        finally:
            self.latencies[case.id] = time.perf_counter() - t0


class RecordingTypryxClient(TypryxClient):
    """TypryxClient that keeps every /v1/ask response it receives, in call
    order. TypedGrader.grade() calls ask() exactly once per case here (no
    tool_trace/outcome_tag cases in this harness), in the same order
    run_eval iterates evalset.cases, so `responses[i]` corresponds to the
    i-th case -- see grade_judge()'s zip below. Kept only for reporting
    (backend/model/template_version); grading itself is unaffected."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.responses: list[dict[str, Any]] = []

    def ask(self, state: dict[str, Any], run_id: str | None) -> dict[str, Any]:
        response = super().ask(state, run_id)
        self.responses.append(response)
        return response


def build_evalset(
    cases: list[dict[str, Any]], grader_kind: GraderKind, rubric: str | None = None
) -> EvalSet:
    eval_cases = [
        EvalCase(
            id=c["id"],
            prompt=c["task"],
            expected="true" if c["truth"] else "false",
            rubric=rubric,
            grader=grader_kind,
        )
        for c in cases
    ]
    return EvalSet(id="bakeoff", cases=eval_cases)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load {id, family, task, final_answer, truth} rows from a JSONL file
    -- either dataset.py's own output or an operator's `--cases FILE` (same
    shape, per the brief)."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field_name in ("id", "family", "task", "final_answer", "truth"):
                if field_name not in row:
                    raise ValueError(f"{path}:{lineno}: missing required field {field_name!r}")
            rows.append(row)
    return rows


def grade_judge(judge_cfg: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run one judge over `cases` through verdryx's own run_eval, and return
    one result row per case: case_id, family, truth, judge, value,
    predicted, correct, cost_usd, latency_s, unanswered, reason, backend,
    model, template_version. Unanswered cases carry value=predicted=
    correct=None -- CLAUDE.md invariant 8's "an unmeasured indicator is
    never a zero" applied to this harness's own rows, matching how
    verdryx.models.Unanswered is kept apart from Score.
    """
    kind = judge_cfg["kind"]
    name = judge_cfg["name"]

    if kind == "typed":
        client = RecordingTypryxClient(
            judge_cfg["typed_url"],
            judge_cfg["typed_key"],
            template=judge_cfg.get("typed_template", DEFAULT_TYPED_TEMPLATE),
            timeout=judge_cfg.get("timeout_s", 30.0),
        )
        inner = TypedGrader(client)
        inner.run_id = str(uuid.uuid4())
        timing = TimingGrader(inner)
        evalset = build_evalset(cases, GraderKind.TYPED)
        adapter: Any = Replay([c["final_answer"] for c in cases])
        run = run_eval(
            evalset, adapter, model=f"bakeoff/{name}", graders={GraderKind.TYPED: timing}
        )
        responses_by_id = {c["id"]: r for c, r in zip(cases, client.responses, strict=False)}
        meta_source = "typryx"
    elif kind == "llm_judge":
        require_claude_confirmation(len(cases), judge_cfg["model"])
        adapter = AnthropicAdapter(model=judge_cfg["model"], base_url=judge_cfg.get("base_url"))
        inner = LLMJudgeGrader(adapter)
        timing = TimingGrader(inner)
        evalset = build_evalset(cases, GraderKind.LLM_JUDGE, rubric=CLAUDE_RUBRIC)
        model_adapter: Any = Replay([c["final_answer"] for c in cases])
        run = run_eval(
            evalset,
            model_adapter,
            model=judge_cfg["model"],
            graders={GraderKind.LLM_JUDGE: timing},
        )
        responses_by_id = {}
        meta_source = "anthropic"
    else:
        raise ValueError(f"unknown judge kind {kind!r} for judge {name!r}")

    score_by_id = {s.case_id: s for s in run.scores}
    unanswered_by_id = {u.case_id: u for u in run.unanswered}

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["id"]
        row: dict[str, Any] = {
            "case_id": case_id,
            "family": case["family"],
            "truth": case["truth"],
            "judge": name,
            "latency_s": timing.latencies.get(case_id, float("nan")),
        }
        if case_id in score_by_id:
            s = score_by_id[case_id]
            row.update(
                value=s.value,
                predicted=s.value >= 0.5,
                correct=(s.value >= 0.5) == case["truth"],
                cost_usd=s.cost_usd,
                unanswered=False,
                reason=None,
            )
        elif case_id in unanswered_by_id:
            u = unanswered_by_id[case_id]
            row.update(
                value=None,
                predicted=None,
                correct=None,
                cost_usd=0.0,
                unanswered=True,
                reason=u.reason,
            )
        else:  # pragma: no cover - defensive; run_eval always does one or the other
            raise AssertionError(f"case {case_id!r} was neither scored nor unanswered")
        if meta_source == "typryx":
            resp = responses_by_id.get(case_id, {})
            row["backend"] = resp.get("backend")
            row["model"] = resp.get("model")
            row["template_version"] = resp.get("template_version")
        else:
            row["backend"] = "anthropic"
            row["model"] = judge_cfg["model"]
            row["template_version"] = "n/a (verdryx rubric, not a typryx template)"
        rows.append(row)
    return rows


# ------------------------------------------------------------------
# Report: metrics per judge/family, cross-judge agreement, markdown + JSON.
# ------------------------------------------------------------------


def _pairs(rows: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    return [(r["value"], r["truth"]) for r in rows if not r["unanswered"]]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _pairs(rows)
    acc = m.accuracy_at(pairs)
    unanswered_rows = [r for r in rows if r["unanswered"]]
    reasons: dict[str, int] = {}
    for r in unanswered_rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    total_cost = sum(r["cost_usd"] for r in rows if not r["unanswered"])
    latencies = [r["latency_s"] for r in rows if not math.isnan(r["latency_s"])]
    return {
        "n_asked": len(rows),
        "n_answered": len(pairs),
        "n_unanswered": len(unanswered_rows),
        "unanswered_reasons": reasons,
        "accuracy": acc.accuracy,
        "passed_wrong": acc.passed_wrong,
        "failed_right": acc.failed_right,
        "mean_confidence": m.mean_confidence(pairs),
        "brier": m.brier(pairs),
        "ece": m.ece(pairs),
        "total_cost_usd": total_cost,
        "cost_per_1000_answered": m.cost_per_1000_answered(total_cost, len(pairs)),
        "latency_p50_s": m.percentile(latencies, 50),
        "latency_p95_s": m.percentile(latencies, 95),
    }


def judge_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({r["family"] for r in rows})
    return {
        "overall": summarize(rows),
        "by_family": {fam: summarize([r for r in rows if r["family"] == fam]) for fam in families},
    }


def _distinct_meta(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    def uniq(key: str) -> list[str]:
        return sorted({r[key] for r in rows if r.get(key) is not None})

    return {
        "backend": uniq("backend"),
        "model": uniq("model"),
        "template_version": uniq("template_version"),
    }


def cross_judge_agreement(
    rows_by_judge: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    preds = {
        name: {r["case_id"]: r["predicted"] for r in rows if not r["unanswered"]}
        for name, rows in rows_by_judge.items()
    }
    names = sorted(preds)
    out: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            agr = m.agreement(preds[a], preds[b])
            out[f"{a}_vs_{b}"] = {"n": agr.n, "agreement": agr.agreement}
    return out


NOT_PROVEN = [
    "Synthetic truth: unless --cases named a real labelled file, every case's truth came from "
    "this harness's own dataset generator (see dataset.py), not a human label.",
    "Latency was measured on this machine, this run, once: it is not a distribution over "
    "network conditions or provider load, and a single p50/p95 from one run is not a "
    "calibrated SLO the way verdryx's own SLO layer requires one to be.",
    "The claude estimate (if claude ran) is a pre-call estimate from a fixed tokens-per-case "
    "constant, not the settled invoice; the actual total_cost_usd reported here comes from "
    "AnthropicAdapter's own per-call accounting against verdryx's PriceBook, which is real.",
    "scripts/no-paid-by-default.sh scans verdryx/ only: this harness's own AnthropicAdapter "
    "construction in examples/bakeoff/run.py is outside that gate's subject list (see the "
    "brief and the report this harness's tests produce).",
    "Cross-judge agreement is measured only over each pair's answered intersection: a judge "
    "that answers fewer cases is not penalized in this number for the ones it declined.",
]


def render_report(
    rows_by_judge: dict[str, list[dict[str, Any]]],
    *,
    dataset_info: dict[str, Any],
    verdryx_commit: str,
    typryx_commit: str,
    judge_cost_notes: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    judge_cost_notes = judge_cost_notes or {}
    per_judge = {name: judge_report(rows) for name, rows in rows_by_judge.items()}
    per_judge_meta = {name: _distinct_meta(rows) for name, rows in rows_by_judge.items()}
    agreement = cross_judge_agreement(rows_by_judge)

    data: dict[str, Any] = {
        "dataset": dataset_info,
        "verdryx_commit": verdryx_commit,
        "typryx_commit": typryx_commit,
        "judges": {
            name: {
                "metadata": per_judge_meta[name],
                "cost_note": judge_cost_notes.get(name),
                "overall": per_judge[name]["overall"],
                "by_family": per_judge[name]["by_family"],
            }
            for name in sorted(rows_by_judge)
        },
        "cross_judge_agreement": agreement,
        "not_proven": NOT_PROVEN,
    }

    lines: list[str] = []
    lines.append("# Bake-off report")
    lines.append("")
    lines.append("No verdict sentence below: every ratio is printed beside its n.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    if dataset_info.get("source") == "external":
        lines.append(f"- external file: `{dataset_info['path']}` (n={dataset_info['n']})")
    else:
        lines.append(f"- generated: seed={dataset_info['seed']}, n={dataset_info['n']}")
        lines.append(f"- mix: {dataset_info['mix']}")
    lines.append(f"- verdryx commit: `{verdryx_commit}`")
    lines.append(f"- typryx commit: `{typryx_commit}`")
    lines.append("")

    for name in sorted(rows_by_judge):
        meta = per_judge_meta[name]
        overall = per_judge[name]["overall"]
        lines.append(f"## Judge: {name}")
        lines.append("")
        lines.append(f"- backend: {meta['backend'] or ['unknown']}")
        lines.append(f"- model: {meta['model'] or ['unknown']}")
        lines.append(f"- template/version: {meta['template_version'] or ['unknown']}")
        note = judge_cost_notes.get(name)
        if note:
            lines.append(f"- cost note: {note}")
        lines.append("")
        lines.append(
            "| scope | n asked | answered | unanswered | accuracy | passed-wrong | failed-right "
            "| mean confidence | Brier | ECE(10,p-binned) | total cost USD | cost/1000 answered "
            "| latency p50 s | latency p95 s |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        def _row(scope: str, s: dict[str, Any]) -> str:
            return (
                f"| {scope} | {s['n_asked']} | {s['n_answered']} | {s['n_unanswered']} "
                f"| {s['accuracy']:.3f} (n={s['n_answered']}) | {s['passed_wrong']} | {s['failed_right']} "
                f"| {s['mean_confidence']:.3f} | {s['brier']:.4f} | {s['ece']:.4f} "
                f"| {s['total_cost_usd']:.4f} | {s['cost_per_1000_answered']:.4f} "
                f"| {s['latency_p50_s']:.4f} | {s['latency_p95_s']:.4f} |"
            )

        lines.append(_row("overall", overall))
        for fam in sorted(per_judge[name]["by_family"]):
            lines.append(_row(fam, per_judge[name]["by_family"][fam]))
        if overall["unanswered_reasons"]:
            lines.append("")
            lines.append(f"unanswered reasons (overall): {overall['unanswered_reasons']}")
        lines.append("")

    lines.append("## Cross-judge agreement (answered intersection)")
    lines.append("")
    if agreement:
        lines.append("| pair | n | agreement |")
        lines.append("|---|---|---|")
        for pair, a in sorted(agreement.items()):
            lines.append(f"| {pair} | {a['n']} | {a['agreement']:.3f} |")
    else:
        lines.append("fewer than two judges ran: nothing to compare.")
    lines.append("")

    lines.append("## Not proven")
    lines.append("")
    for item in NOT_PROVEN:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines), data


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _cmd_claude_estimate(args: argparse.Namespace) -> int:
    try:
        require_claude_confirmation(args.n, args.model)
    except SystemExit as e:
        return int(e.code or 1)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    judges_config = json.loads(Path(args.judges_config).read_text(encoding="utf-8"))

    claude_cfgs = [j for j in judges_config if j["kind"] == "llm_judge"]
    for j in claude_cfgs:
        require_claude_confirmation(len(cases), j["model"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_judge: dict[str, list[dict[str, Any]]] = {}
    judge_cost_notes: dict[str, str] = {}
    for judge_cfg in judges_config:
        name = judge_cfg["name"]
        print(f"grading judge {name!r} ({judge_cfg['kind']}) over {len(cases)} cases...")
        rows = grade_judge(judge_cfg, cases)
        rows_by_judge[name] = rows
        write_jsonl(rows, out_dir / f"{name}.jsonl")
        if judge_cfg.get("cost_unpriced"):
            judge_cost_notes[name] = (
                "unpriced (TYPRYX_JEV_PRICE_PER_MTOK_INPUT/_OUTPUT were not set for this run; "
                "cost_usd is a real 0, not an estimate, but it is not a real price either)"
            )

    if args.external_cases:
        dataset_info = {"source": "external", "path": args.external_cases, "n": len(cases)}
    else:
        dataset_info = {
            "source": "generated",
            "seed": args.seed,
            "n": args.n,
            "mix": family_sizes(args.n) if args.n else None,
        }

    report_md, report_json = render_report(
        rows_by_judge,
        dataset_info=dataset_info,
        verdryx_commit=args.verdryx_commit,
        typryx_commit=args.typryx_commit,
        judge_cost_notes=judge_cost_notes,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'report.md'} and {out_dir / 'report.json'}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    est = sub.add_parser(
        "claude-estimate", help="print the claude spend estimate and refuse without confirmation"
    )
    est.add_argument("--n", type=int, required=True)
    est.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    est.set_defaults(func=_cmd_claude_estimate)

    run_p = sub.add_parser("run", help="grade every configured judge and write the report")
    run_p.add_argument("--judges-config", required=True, help="JSON list of judge configs")
    run_p.add_argument(
        "--cases", required=True, help="JSONL cases file (dataset.py output or --cases)"
    )
    run_p.add_argument("--out", required=True, help="output directory")
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--n", type=int, default=None)
    run_p.add_argument(
        "--external-cases", default=None, help="set when --cases named an operator file"
    )
    run_p.add_argument("--verdryx-commit", default="unknown")
    run_p.add_argument("--typryx-commit", default="unknown")
    run_p.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
