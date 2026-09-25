# Judge bake-off

Grades the same N labelled cases with several judges and scores each judge
against the known truth, so the four candidates for "what judges verdryx's
eval sets" can be compared on the same footing before any of them is picked
by default.

## What it measures, and why

An eval-set judge has to answer one question well: given a task and a final
answer, was the answer right? This harness builds a dataset where that
answer is known by construction (see `dataset.py`), asks each judge, and
scores the judge's own stated probability against the truth: accuracy at
0.5, how often it passed a wrong answer or failed a right one, how well its
stated confidence matches how often it is actually right (Brier score, and
Expected Calibration Error binned on the stated probability itself, not on
`max(p, 1-p)`), plus latency and cost. It does this once for each judge
over the identical cases, so the comparison is apples to apples.

## The four judges

| name | what it is | cost |
|---|---|---|
| `qwen` | typryx's `openai-logprobs` backend over a local Ollama `qwen2.5:7b` | free |
| `jev` | typryx's `jev` backend, TypeSafe AI's API | needs a key; paid or free per their pricing (unknown as of this writing) |
| `claude` | verdryx's own `LLMJudgeGrader` with `AnthropicAdapter` | paid, Anthropic API |
| `stub` | typryx's `stub` backend, deterministic | free |

Every judge is asked the same question, in the shape each one is built to
answer: typryx's `eval.outcome_met` template for `qwen`/`jev`/`stub`
("does `final_answer` achieve `task`", a noul/yes-or-no question), and an
equivalent rubric (`run.py`'s `CLAUDE_RUBRIC`) for `claude`'s LLM-judge
grader, so all four are answering the same question rather than four
slightly different ones.

## The two steps for the day a Jev key exists

1. `security add-generic-password -s typesafe-jev-api-key -a "$USER" -w`
   (paste the key when prompted).
2. `./bakeoff.sh --judges qwen,jev,claude,stub --n 1000` (add `claude` and
   `BAKEOFF_CONFIRM_SPEND=yes` only once you have looked at the cost
   estimate it prints and agreed to it; see below).

Everything else -- building typryx, starting a server per judge, generating
the dataset, grading, and writing the report -- is one command.

## Dry run (free, no key needed)

```sh
./bakeoff.sh --dry-run --judges qwen,jev,stub --n 200
```

Every typryx-backed judge here runs its typryx process with
`TYPRYX_BACKEND=stub` **except** `jev`, which keeps the real `jev` backend
code path but points `TYPRYX_JEV_URL` at `fake_jev.py` (a local, free,
deterministic stand-in for the Jev API started by `bakeoff.sh` itself) with
a throwaway key `fake_jev.py` was told to expect. That exercises the actual
jev integration -- the key-file plumbing, the request/response shape,
typryx's own `jev.go` -- for free, rather than skipping straight to the
trivial stub backend the way `qwen` does under `--dry-run`.

`qwen` alone against the real local Ollama is also free and needs no key:

```sh
./bakeoff.sh --judges qwen --n 200
```

(skipped with one line if Ollama is not reachable at `127.0.0.1:11434` or
`qwen2.5:7b` is not pulled: `ollama pull qwen2.5:7b`).

## What each judge costs, and the cap

- `qwen` and `stub`: free, always.
- `jev`: whatever TypeSafe AI charges, unknown as of this writing.
  `bakeoff.sh` sets `TYPRYX_MAX_CALLS_PER_HOUR` to the run's own N plus 5%
  (rounded up) on jev's typryx process -- **this is the spend cap**, printed
  before anything runs. `TYPRYX_JEV_PRICE_PER_MTOK_INPUT`/`_OUTPUT` pass
  through from your own shell environment if you set them; if you do not,
  the report marks jev's cost as **unpriced**, never as a bare `$0`, since
  unset really does mean "we don't know the price," not "it is free."
- `claude`: priced per verdryx's own `PriceBook` (the same table
  `AnthropicAdapter` uses everywhere else in this repository). `claude`
  never runs unless `--judges` names it **and** `BAKEOFF_CONFIRM_SPEND=yes`
  is set in your environment; without that, `bakeoff.sh` prints the
  estimate (N, an approximate input-token count per case, the model, and
  the priced total) and exits before building anything else, so asking for
  `claude` without confirming costs nothing and starts nothing.

## Where results land

`--out DIR` (default `examples/bakeoff/results/<UTC timestamp>`, gitignored
-- generated output is never checked in):

- `<out>/<judge>.jsonl`: one row per case for that judge (value, predicted
  label, cost, latency, whether it was unanswered and why).
- `<out>/report.md`, `<out>/report.json`: the same numbers, rendered as
  markdown and as data. Every ratio is printed next to the `n` it rests on;
  there is no verdict sentence anywhere in either file, on purpose -- this
  harness reports numbers, it does not recommend a judge.
- `<out>/<judge>/typryx.log`, `<out>/<judge>/ledger/`,
  `<out>/<judge>/events.ndjson`: each typryx process's own log, ledger
  (`answers.ndjson`/`outcomes.ndjson`), and agent-event NDJSON, kept
  separate per judge.

## Reading the report

Per judge, overall and per family (`arithmetic`, `routing`, `format`): how
many cases were asked, answered, and left unanswered (with typryx's own
reasons, e.g. `label_mass_too_low`); accuracy at the 0.5 threshold with its
own `n`; passed-wrong and failed-right counts; mean stated confidence
(`max(p, 1-p)`); Brier score; Expected Calibration Error over 10
equal-width bins of the judge's own stated probability (**not** binned on
confidence -- see `metrics.py`'s `ece()` docstring for why that distinction
is stated rather than assumed); total cost and cost per 1000 answered
cases; latency p50/p95. Below that, pairwise agreement between every two
judges that ran, measured only over the cases both of them answered.

An unanswered case is counted, never scored -- the same rule
`verdryx.models.Unanswered` already holds for `verdryx eval` itself (see
`CLAUDE.md` invariant 8 and `verdryx#43`): a probability a judge could not
give is not a confident zero, and averaging it in as one would manufacture
a fact nobody measured.

## What this does not prove

- **Synthetic truth.** Unless you pass `--cases FILE` with real labelled
  cases in the same JSONL shape (`{id, family, task, final_answer, truth}`,
  one per line), every case's truth comes from `dataset.py`'s own generator,
  not a human label. It is right by construction (a test checks the wrong
  answers really are wrong and the right ones right), but a judge that is
  well calibrated on synthetic arithmetic, routing, and word-format cases
  is not thereby proven well calibrated on your own eval sets.
- **One machine, one run.** Latency is measured here, once, on whatever
  machine ran it -- not a distribution over network conditions or provider
  load, and not the calibrated SLO shape `verdryx slo` holds itself to.
- **`fake_jev.py`'s noul answer is a hash, not a judgement.** The dry run
  proves the jev integration plumbing works end to end for free; it proves
  nothing about how well the real Jev API judges anything.
- **`scripts/no-paid-by-default.sh` does not scan this directory.** It
  checks `verdryx/` for exactly one `AnthropicAdapter` construction site;
  this harness's own site, in `run.py`, is outside that gate's subject
  list. The spend guard here (`run.py`'s `require_claude_confirmation`) is
  this harness's own, separate control, tested directly in
  `tests/test_bakeoff.py`.
