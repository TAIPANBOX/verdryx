# verdryx

**Verdryx measures whether an operator's own agents did their job correctly.
It never manipulates outputs, never crafts adversarial prompts, and never
attacks anything.** Verdryx is the quality-evaluation and drift plane of the
TAIPANBOX agent-governance stack: given an eval set, it grades a model's
outputs against expected values, regex patterns, recorded production outcome
tags, or an LLM-judged rubric, stores the results, and tells you when
quality has drifted against a baseline you set. This is entirely defensive,
self-measurement tooling for the operator running the agents, not a
red-teaming or offensive-security tool.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Table of contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [Eval set format](#eval-set-format)
- [Graders](#graders)
- [Drift](#drift)
- [Cost per outcome](#cost-per-outcome)
- [Events](#events)
- [Configuration](#configuration)
- [Relationship to the rest of the stack](#relationship-to-the-rest-of-the-stack)
- [Development](#development)
- [License](#license)

---

## What it does

1. **Eval runner** (`verdryx.cli.run_eval`, `verdryx eval`): loads an
   `EvalSet` (a list of `EvalCase` prompts), calls a model for each case,
   grades the output with the case's chosen grader, and stores the result
   as an `EvalRun` of per-case `Score`s in a local SQLite file.
2. **Four graders** (`verdryx/graders.py`): exact match, regex match, a
   `tokenfuse x-fuse-outcome` tag lookup, and an LLM-judged rubric. All four
   share one shape, so `verdryx eval` dispatches to whichever a case asks
   for without special-casing.
3. **Baselines and drift** (`verdryx/drift.py`): snapshot an `EvalRun`'s
   mean score as a `Baseline`, then compare a window of later runs against
   it. A `DriftReport` says `on-track` or `regressed` by a configurable
   threshold.
4. **Cost per outcome** (`verdryx/costper.py`): given a flat export of
   `{outcome, cost_usd}` records (for example a `tokenfuse outcomes --json`
   export converted to dollars, or an agent-event/trace outcome-tag
   extract), computes cost-per-resolved-case, cost-per-escalated,
   cost-per-abandoned, and overall.
5. **Opt-in event log** (`verdryx/events.py`): an NDJSON writer for the
   shared TAIPANBOX Agent Passport event envelope (schema
   `taipanbox.dev/agent-event/v0.2`, `source: "verdryx"`), so the rest of
   the governance stack can see `eval_run`, `quality_score`, and
   `quality_drift` events without depending on Verdryx's internals.

## Install

System Python is often externally managed (PEP 668); use a virtual
environment.

```bash
git clone https://github.com/TAIPANBOX/verdryx
cd verdryx
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For the real LLM-judge adapter (`AnthropicAdapter`), add the `anthropic`
extra. For reading tokenfuse Parquet traces directly (`read_parquet`,
`cost-per-correct --traces`), add the `traces` extra:

```bash
pip install -e '.[anthropic]'
pip install -e '.[traces]'
```

## Quick start

Write an eval set (see [Eval set format](#eval-set-format)), then:

```bash
# Dry run against a deterministic stub model -- no network, no API key.
verdryx eval evalset.json --model stub --db verdryx.db

# Snapshot that run as the reference point for future drift checks.
verdryx baseline <run-id-printed-above> --db verdryx.db --label "v1"

# Later, after re-running eval against a new prompt/model version:
verdryx eval evalset.json --model stub --db verdryx.db
verdryx drift --baseline <baseline-id> --db verdryx.db --window 3

# Unit economics from a tokenfuse outcomes export (NDJSON, CSV, or Parquet
# of {"outcome": ..., "cost_usd": ...} records):
verdryx cost-per-correct --input outcomes.ndjson

# ...or straight from a tokenfuse Parquet trace directory (requires the
# `traces` extra):
verdryx cost-per-correct --traces $TOKENFUSE_DATA_DIR
```

`--model stub` selects a deterministic, network-free adapter -- useful for
validating an eval set's structure, and it's exactly what Verdryx's own test
suite uses so CI never makes a real API call. Any other `--model` value is
treated as a real Anthropic model id (requires the `anthropic` extra and
`ANTHROPIC_API_KEY`, or `--events`/`ANTHROPIC_BASE_URL` to route through a
proxy such as TokenFuse).

The same operations are available as a plain Python API:

```python
from verdryx import EvalSet, Store, StubLLMAdapter, compute_drift
from verdryx.cli import run_eval

evalset = EvalSet.load("evalset.json")
run = run_eval(evalset, StubLLMAdapter(), model="stub")

with Store.open("verdryx.db") as store:
    store.save_run(run)
    baseline = store.get_baseline("some-baseline-id")
    if baseline is not None:
        report = compute_drift([run], baseline)
        print(report.verdict)
```

## Eval set format

A JSON file with an `id` and a list of `cases`:

```json
{
  "id": "support-tier1-v1",
  "cases": [
    {
      "id": "greets-politely",
      "prompt": "Reply to: 'my order is late'",
      "expected": "sorry",
      "grader": "regex"
    },
    {
      "id": "resolves-refund",
      "prompt": "Draft a refund confirmation for order #4471",
      "rubric": "Confirms the refund amount and a realistic timeline.",
      "grader": "llm_judge"
    },
    {
      "id": "run-8842-outcome",
      "prompt": "case_resolved",
      "grader": "outcome_tag"
    }
  ]
}
```

`id` must be stable across runs of the same eval set (it is not
auto-generated): Scores are compared case-by-case over time, so a case's id
needs to mean the same thing on every run. `grader` is one of `exact`
(default), `regex`, `outcome_tag`, or `llm_judge` -- see [Graders](#graders).
For `outcome_tag` cases, `prompt` holds the outcome tag itself (there is
nothing to send a model when grading an already-recorded production
outcome).

## Graders

| Grader | `case.expected` / `case.rubric` | Scores 1.0 when |
|---|---|---|
| `ExactGrader` | `expected`: literal string | `output == expected` |
| `RegexGrader` | `expected`: a regex pattern | `re.search(expected, output)` matches |
| `OutcomeTagGrader` | none (reads `output` itself) | `output` is a tag mapped to `1.0` in its table |
| `LLMJudgeGrader` | `rubric`: grading instructions | the injected judge scores the rubric that high |

`OutcomeTagGrader`'s default table is `{"case_resolved": 1.0, "escalated":
0.5, "abandoned": 0.0}` (`verdryx.DEFAULT_OUTCOME_SCORES`), fully overridable
via its `mapping` constructor argument. An unrecognized tag scores `0.0` by
default rather than raising.

`LLMJudgeGrader` takes an injected adapter satisfying the `LLMAdapter`
protocol (`complete()` + `judge()`). Two are provided:

- `StubLLMAdapter`: deterministic, records every call, no network. This is
  what Verdryx's own tests use, and what `--model stub` selects on the CLI.
- `AnthropicAdapter`: the real adapter, backed by the Anthropic Messages
  API. Its constructor mirrors the `AnthropicAdapter` seam in
  [Engram](https://github.com/TAIPANBOX/engram) (`engram/llm.py`): `model`,
  `base_url`, and `api_key` are accepted the same way, and `base_url` lets
  judge/completion calls route through a proxy (e.g. TokenFuse) instead of
  hitting Anthropic directly. Verdryx does not depend on the `engram`
  package; this is the same construction pattern applied locally. Its
  `judge()` calls also price themselves against `verdryx.pricing.PriceBook`
  (a Python port of TokenFuse's own default price book), so an `llm_judge`
  case's `Score.cost_usd` is a real dollar figure, not the `0.0` placeholder
  the other three graders leave in place (they make no model call, so there
  is nothing to price). Pass `price_book=` to price against your own table
  instead.

The candidate output a judge grades is wrapped in an `<output>` tag with an
instruction to treat it as inert data, the same delimited-block technique
Engram uses for episodic content (`engram/llm.py`'s `_wrap_observations`):
grading untrusted agent output is exactly the kind of place a prompt
injection would try to hijack the grader, so the judge prompt is built
defensively even though Verdryx itself never acts on what it reads.

## Drift

`compute_drift(recent_runs, baseline, threshold=0.05)` pools every case
score across a window of the most recent eval runs into one mean (not a
mean of each run's own mean, which would distort unevenly-sized runs), then
compares that pooled mean to `baseline.mean_score`:

- `delta = mean_score - baseline.mean_score`
- `verdict = "regressed"` if `delta <= -threshold`, else `"on-track"`

`verdryx drift --baseline ID --window N` fetches the baseline, filters
stored runs to the same model, takes the `N` most recent, and prints the
report. On `regressed`, and only then, it emits a `quality_drift` event
(severity `high`) if an event log is configured -- `on-track` checks are
not reported as events, matching how Engram only emits
`contradiction_found` when an actual contradiction occurs, not on every
`reflect()` call.

This is a flat threshold, not a significance test: no p-values or
confidence intervals in this MVP. A statistically-aware verdict is a
documented later enhancement.

## Cost per outcome

`cost_per_outcome(records)` takes an iterable of `{"outcome": str,
"cost_usd": float}` mappings and returns a `CostPerOutcomeReport`: one
`OutcomeCost` (count, total, mean) per outcome tag that appears in the
input, plus an `overall` row pooling everything. `.resolved`, `.escalated`,
and `.abandoned` are named accessors for the three default tags;
`.get(tag)` works for any custom tag.

`load_records(path)` reads `.ndjson`/`.jsonl`, `.csv`, a single `.parquet`
file, or a directory of `.parquet` files, and dispatches automatically;
`verdryx cost-per-correct --input <path>` (file) or `--traces <dir>`
(directory of tokenfuse Parquet segments, e.g. `TOKENFUSE_DATA_DIR`) wraps
all of them in one CLI command.

`read_parquet` (requires the `traces` extra: `pip install -e '.[traces]'`)
reads tokenfuse's `outcome` and `cost_microusd` trace columns directly
(`tokenfuse`'s `crates/gateway/src/sink.rs`), converting microdollars to
`cost_usd` and dropping untagged rows (most rows in a raw trace: tokenfuse
expects only a run's final call to carry the outcome tag). It does not
reproduce tokenfuse-core's full "last non-empty outcome tag per run wins"
aggregation (`tokenfuse`'s `crates/core/src/outcomes.rs`) -- that still
happens upstream, e.g. via `tokenfuse outcomes --json`, for a run whose
agent tags more than one of its calls. Reproducing that reduction inside
Verdryx itself remains a documented later enhancement.

## Events

Disabled by default. Set `VERDRYX_EVENTS_PATH` or pass `--events <path>` to
`eval`/`drift` to turn it on. Every event follows the shared TAIPANBOX
Agent Passport envelope (`taipanbox.dev/agent-event/v0.2`, see the
`agent-passport` repo's `SPEC.md`):

| `type` | severity | `data` |
|---|---|---|
| `eval_run` | info | `model`, `cases`, `mean_score`, `total_tokens`, `total_cost_usd` |
| `quality_score` | info | `case_id`, `value`, `tokens`, `cost_usd` |
| `quality_drift` | high | `baseline_id`, `window`, `mean_score`, `delta`, `verdict` |

Same rules as Engram's exporter: **opt-in** (no file, no thread, no
allocation unless a path is configured), **fail-open** (a write failure is
logged and swallowed, never raised into the caller's eval/drift call), and
an event is **skipped** (counted in `EventLog.skipped_empty_agent_id`)
whenever `agent_id` is empty -- Verdryx never fabricates one. Pass
`--agent-id agent://your-org.example/...` to `eval`/`drift` to identify
which agent's output is being measured.

## Configuration

Read once, at process start, into `verdryx.config.Config`:

| Variable | Meaning |
|---|---|
| `VERDRYX_DB` | Default SQLite store path (default: `verdryx.db`) |
| `VERDRYX_EVENTS_PATH` | Default NDJSON event log path (default: unset, events disabled) |
| `VERDRYX_OTLP_ENDPOINT` | Read into config for a future OTLP exporter; not wired up to anything yet |
| `ANTHROPIC_API_KEY` | API key for the real `AnthropicAdapter` |
| `ANTHROPIC_BASE_URL` | Proxy endpoint (e.g. TokenFuse) for the real `AnthropicAdapter` |

CLI flags (`--db`, `--events`) always take precedence over the environment.

## Where this fits in the stack

Verdryx is the quality plane of the TAIPANBOX agent-governance stack: it grades an agent's outputs and tells you when quality has drifted against a baseline.

```mermaid
flowchart TB
  Agent["AI agent (any framework)"] -->|"LLM call (base-URL swap)"| TF["TokenFuse proxy: spend + enforcement"]
  TF -->|"POST /v1/decide (PEP)"| WX["Wardryx: policy PDP"]
  WX -.->|"allow / deny / hold"| TF
  TF -->|"cheapest model, budget OK"| LLM[("LLM provider")]
  TF -->|"CallRecords"| CL["TokenFuse Cloud: control plane, incidents, replay, evidence, kill-switch"]
  TF ==>|"agent-event NDJSON"| BUS{{"agent-event bus + Agent Passport"}}
  WX ==> BUS
  ENG["Engram: memory"] -->|"reflect via base_url"| TF
  ENG ==> BUS
  BUS ==> IDX["Idryx: identity graph, detectors, Agent-BOM"]
  BUS ==> QX["Qryx: crypto / PQC, passport + hash-chain scan"]
  BUS ==> VX["Verdryx: quality / drift"]
  VX ==>|"quality events"| BUS
  TF -->|"outcome-tagged traces"| VX
  MX["Mockryx: pre-prod safety rehearsal"] -->|"hostile scenarios"| TF
  MX ==>|"sim events"| BUS
  TFP["terraform-provider-taipan"] -->|"budgets + passports as code"| CL
  ASG[["agent-stack-go: shared Go contract"]] -.->|imported by| IDX
  ASG -.->|imported by| WX
  ASG -.->|imported by| MX
  ASG -.->|imported by| TFP
  SPEC[["agent-passport: the spec"]] -.->|governs| BUS
```

- **Consumes**: outcome-tagged traces and records exported by **TokenFuse** (via `tokenfuse outcomes --json` or Parquet traces).
- **Produces**: cost-per-correct metrics, quality scores, drift reports, and `source: verdryx` events.
- **Talks to**: **TokenFuse** (its LLM judge can route through TokenFuse via `base_url`, and TokenFuse is the source of the traces Verdryx scores).

The full stack is TokenFuse (spend), Wardryx (policy), Engram (memory), Idryx (access), Qryx (crypto), Verdryx (quality), Mockryx (pre-prod), on the shared Agent Passport + agent-event contract (agent-stack-go / agent-passport), configured via terraform-provider-taipan.

## Relationship to the rest of the stack

Verdryx is one plane of the TAIPANBOX agent-governance stack, alongside
[Engram](https://github.com/TAIPANBOX/engram) (memory), TokenFuse (spend
governance), [Idryx](https://github.com/TAIPANBOX/idryx) (identity),
[Qryx](https://github.com/TAIPANBOX/qryx) (cryptographic evidence), and
[Wardryx](https://github.com/TAIPANBOX/wardryx) (policy decisions). Each
product is complete alone; the stack shares one identifier format
(`agent://...`) and one event envelope, defined in the
`TAIPANBOX/agent-passport` repo. Verdryx answers one question none of the
others do: *did the agent's output actually meet the bar, and is that bar
slipping?*

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,traces]'

pytest              # run the test suite
ruff check .        # lint
ruff format .       # format
```

All eval/judge network calls are behind the injected `LLMAdapter` protocol,
so the test suite runs fully offline against `StubLLMAdapter`. The `traces`
extra is optional -- without it, the Parquet-reading tests skip themselves
via `pytest.importorskip` instead of failing.

## License

Apache License 2.0. See [LICENSE](LICENSE).
