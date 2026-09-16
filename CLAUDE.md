# CLAUDE.md, working instructions for verdryx

These instructions apply to any model working in this repo. Read this file
before writing code. It holds process and invariants only: **no status.**
Status goes stale, and a stale instruction file is worse than none. For where
the code actually is, read `VALIDATION.md` and the README.

## Read before you change anything

1. `README.md`, the "Where this fits in the stack" section. Verdryx is the
   quality plane: it reads TokenFuse's outcome-tagged Parquet traces and
   computes cost per outcome and quality drift.
2. `pyproject.toml`. The dependency split there is an architectural decision,
   not packaging trivia. See invariants 1 and 2. `requirements-dev.lock` is
   the pinned resolution CI and an operator install from; see invariant 10
   and the lock file's own header before changing either.
3. `verdryx/events.py`, the canonicalization comment. It says why the RFC 8785
   implementation is delegated rather than written here.
4. `SPEC.md` in the sibling repo `TAIPANBOX/agent-passport` for the event
   envelope this service emits and consumes.

## What this service is

The quality and evaluation plane of the stack, and the one Python service in
it. It grades agent outputs, including with a priced LLM judge, computes cost
per outcome, and detects quality drift. It is the denominator under the stack's
unit economics: TokenFuse says what a run cost, Verdryx says whether the run
was worth it.

This service is defensive: it exists so an organization can evaluate its own
agents. Never describe it as tooling for acting against anyone else.

## The working loop

1. Branch off `main`, one logical increment per branch.
2. Run every gate below. All must pass locally before the push.
3. Commit with Conventional Commits. End the message with the standard
   co-author trailer naming the model that actually did the work.
4. Push the branch, open a PR with `gh`.
5. Wait for all CI checks to go green. Fix forward, do not force-push over red.
6. **Ask the user before merging.** Do not self-merge.

## Gates

```sh
ruff check .
ruff format --check .
pytest
./scripts/optional-imports.sh
./scripts/no-paid-by-default.sh
./scripts/one-runtime-dependency.sh
./scripts/readme-numbers.sh
./scripts/features-are-bound.sh
./scripts/lock-satisfies-pyproject.sh
./scripts/gates-have-teeth.sh   # invariant 7; needs a clean tree and the package installed
pip-audit --skip-editable       # CI only; no ignored advisories, needs pip-audit installed
```

`readme-numbers.sh` was missing from this list until 2026-08-09 while CI ran
it, so this instruction was strictly smaller than CI's. It also called `python`
rather than `python3`, which every other script here gets right, so on macOS it
could not run at all: it failed with `command not found` before reaching a
single comparison. Both fixed in the same change.

Note `ruff format --check`, not `ruff format`. CI checks formatting rather than
applying it, so a local run that reformats files and then passes is not the
same signal.

## Hard invariants

Each one carries how it is held today. Use `(gate: ...)`, `(test: ...)`,
`(partly gated: ...)` or `(not enforced)`, and use the weakest one that is
true. An invariant with no check, written as though it had one, is worse than
an absent invariant.

1. **The runtime core has exactly one dependency: `rfc8785`.** Everything else
   is an optional extra: `anthropic` for the priced LLM judge, `pyarrow` for
   reading Parquet traces, plus the `dev` group. Installing verdryx must not
   drag an LLM SDK or Arrow onto a machine that only needs the deterministic
   graders. Adding a runtime dependency is a decision for the user, not a
   convenience. *(gate: `scripts/one-runtime-dependency.sh`)*
2. **An optional dependency is imported lazily, inside the function that needs
   it, wrapped in `try/except ImportError` that re-raises with a pip-install
   hint.** Never at module top level. A top-level import turns an optional
   extra into a hard requirement for anyone who imports the module, and the
   failure shows up as an ImportError on an unrelated code path.
   *(gate: `scripts/optional-imports.sh`)*
3. **RFC 8785 canonicalization is delegated to `rfc8785`, never hand-rolled.**
   Two implementations of a canonical form always disagree eventually, and this
   one has to agree byte for byte with the Go side of the stack, which uses
   `gowebpki/jcs`. If canonical bytes differ, every `prev_hash` chain crossing
   the language boundary breaks. *(not enforced)*
4. **Verdryx reads traces, it does not produce them.** The outcome-tagged
   Parquet traces are TokenFuse's output and its schema. If a field is missing,
   the fix is in TokenFuse, not a locally invented column here.
   *(not enforced)*
5. **A grader that costs money never runs by default.** The LLM judge is priced
   per call. Any code path that could reach a paid provider must be explicitly
   selected by the caller, and the default configuration must be the
   deterministic graders. *(gate: `scripts/no-paid-by-default.sh`)*

6. **The test count this README states about the repository is checked
   against the repository.** The coverage figure the README also states is
   the exception, and invariant 11 says so and why. A figure on a page has no
   owner and no clock: it is right the day it is written, and the suite
   grows in commits that never open the README.
   This repository was one of four caught by that on 2026-08-05, when the seven
   figures on it-rat.com were audited against the code they describe: the page
   said **217 tests where pytest collects 292**. It was not wrong when written.
   The badge counts collected ITEMS, so a parametrised function counts once per
   case, because that is what a contributor sees when the suite runs and it is
   deliberately not the count of `def test_` lines, which is smaller and which
   nobody arrives at by running anything.
   *(gate: `scripts/readme-numbers.sh`; verified by moving the badge one test,
   which fails it and names both figures. It collects rather than executes, so a
   green badge cannot mean a red suite; `pytest` above it in CI is what says
   they pass.)*

7. **A check must be able to tell "did not fail" from "did not run", and every
   gate here has been made to fail on purpose to prove it can.** Three of the
   four gates above already refuse when their subject is absent: no declared
   dependencies at all, pytest reporting no collected count, an environment
   that could not be built. Every one of those sentences was true, was
   established by hand once in the session that wrote the script, and nothing
   re-ran them.

   A text parser does not break loudly: it stops matching and reports success.
   The mutants that proved these gates lived in commit messages and in the
   `*(gate: ...)*` markers above, which is a record of what was true once.
   *(gate: `scripts/gates-have-teeth.sh`, 19 cases: thirteen real faults each
   gate must catch, two non-faults they must not, and four subjects taken away
   entirely. It read "10 cases" until 2026-08-26 and "14 cases" until the lock
   gate's own teeth arrived, having been written once while the cases kept
   arriving each time, which is invariant 6's own failure inside
   the file that records invariant 6. Counted by running it. The non-fault cases are the ones worth keeping: an optional
   dependency imported INSIDE a function is exactly what invariant 2 allows,
   and a gate that fired on it would be deleted by whoever is unblocking CI.)*

   **What it does not cover.** It cannot test itself. It proves each gate
   catches the faults named in it, not every fault of that kind. It found no
   hole in any of the four.

8. **A figure this plane puts in front of an operator arrives with what it
   rests on, and an indicator it could not compute says so rather than
   reporting a number.** The SLO layer is where this bites hardest, because
   every one of its outputs is a small float that looks equally authoritative
   whether it rests on four hundred runs or four.

   Six shapes, each with its own way of lying quietly. It said "three"
   over four bullets when it was written and over six on 2026-08-26,
   which is the same failure as invariant 6 in the file that records
   invariant 6: a count in prose beside a list that grows. Counted by
   reading the list.

   - **A ratio travels with its interval and its `n`.** Three of four runs is
     0.75 and says nothing; three hundred of four hundred is 0.75 and says a
     great deal. The bus is held to a higher bar than the report on purpose:
     `slo.burn_events` emits only where the Wilson upper bound sits below the
     target, because a report is read by somebody who can see the sample size
     and a bus line is read by a program that cannot.
   - **An unmeasured indicator is never a zero.** `burn_rate == 0.0` with
     `burn_events_seen == 0` means no clock and no rate, which is the opposite
     fact from a fleet burning nothing, and the two are identical in the
     float. Same for an SLI below `min_events`: it reports `measured=False`
     with a reason naming where the missing input would have come from.
   - **A subject the envelope cannot carry is reported and not emitted.**
     Grouping on `key_id` is the sound choice for anything enforced and
     produces credentials, not agents; the envelope's one subject field is an
     `agent_id` with a fixed grammar. So a key-keyed breach appears in the
     report and never on the bus, rather than being written malformed for a
     consumer to reject.
   - **A run nobody can attribute is counted, never bucketed.** A fleet scores
     better the less of it is identified either way; only one of the two says
     so, and `SloReport.unattributed_runs` is printed beside the figures
     rather than under them. The same rule holds for the OTHER input:
     `unattributed_scores` counts a score from a `verdryx eval` that was given
     no `--agent-id`. Two counters and not one, because a fleet can be well
     identified on the gateway and badly identified in its eval store.
   - **A reason that has stopped being true is worse than none.**
     `quality_floor` reported itself unmeasured because `eval_runs` carried no
     subject, so a score could not be joined to a fleet at all. That column
     exists since 2026-08-26 and the reason was rewritten in the same change,
     because the old one sent a reader to close a gap already closed. When a
     limitation is removed, its `UNMEASURED_REASONS` entry moves with it.
   - **An absence is named as the absence it is.** "No run carried an outcome
     tag" is the wrong diagnosis for a subject with no runs at all, which is
     now reachable: an agent can be known to the report only from its eval
     scores. Three absences, three sentences (`slo._unmeasured_reason`).

   **Found by its own test, and worth keeping written down.** The module first
   computed the error budget and the burn rate over the same runs, which makes
   the rate exactly `1 - remaining`: exhaustion then wins every comparison and
   `fast_burn` and `slow_burn` are unreachable code. A trigger that can never
   fire reports identically to one with nothing to report. The budget now
   looks at the whole window and the rate at a recent slice of it
   (`BURN_WINDOW_FRACTION`), which is the SRE multi-window shape and, more to
   the point, is two questions instead of one asked twice.
   *(test: the twenty scenarios in `features/agent-error-budget.feature`,
   each bound to a named test by `scripts/features-are-bound.sh`; thirty-seven
   tests in `tests/test_slo.py`, six of which were verified red against a
   planted defect: the evidence bar removed, unattributed runs bucketed, the
   cost reference taken per subject instead of over the fleet, containment
   read off the nine Breaker reasons alone, the burn rate returned to one
   window, and the budget clamped at zero. The eight added on 2026-08-26 for
   the eval-store join were all red against the code before it.
   `gate: scripts/features-are-bound.sh` holds the binding in both directions; nothing mechanical can hold whether a
   scenario's prose still describes its test, and that limit is in the
   script.)*

9. **This plane measures and does not enforce.** `@yurii 2026-08-26`: "лише
   вимірювання". Not a half-built state: `slo.py` writes no wardryx policy,
   demotes nothing and gates nothing, and the reasons are structural rather
   than a matter of effort. wardryx's PDP is a pure function of (policy set,
   request), held by its own `scripts/decision-path-purity.sh`, so it cannot
   read a budget and a gate would have to WRITE policy. No per-agent autonomy
   tier exists anywhere in the estate to write into. And the budget can only
   be keyed soundly on `key_id` while wardryx policies target `agent_id`
   globs, which tokenfuse's own source calls "unsound as the key of a budget";
   the join between them is the gateway's identity map, which is off by
   default. A gate built over that seam would look identical whether it held
   or not.

   Adding enforcement is a decision for the user, not a convenience, and it
   belongs in the same list as adding a runtime dependency.
   *(not enforced; held by this file and by there being no policy-writing code
   to find)*

10. **The install path CI and an operator use resolves to exact versions, not
    whatever the day's floor-satisfying set happens to be.** `pyproject.toml`
    keeps `>=` floors as the declared surface (invariant 1);
    `requirements-dev.lock` is the pinned resolution of the dev/test
    environment (`pip install -e '.[dev,traces]'`) that CI and a developer
    actually install from. A floor moving in `pyproject.toml` with nobody
    regenerating the lock, or a dependency added to `pyproject.toml` with
    nobody regenerating the lock, both look identical from `pyproject.toml`
    alone, which only ever shows the aspiration.
    *(gate: `scripts/lock-satisfies-pyproject.sh`)*

11. **Line coverage of the `verdryx` package is measured and reported on
    every CI run and stated in the README, but it is not a merge gate.**
    `pytest --cov=verdryx --cov-report=term-missing` in the `test` job prints
    the total every run, and the README states the figure by hand at the
    point it was measured. It is not recomputed by a script the way the test
    count is (invariant 6): doing that would mean running the full suite a
    second time inside a gate, which buys nothing the `pytest` step above it
    did not already run. Whether a coverage number should ever fail a build
    is the user's decision, not made here.
    *(partly gated: CI prints the number every run; the README figure is a
    restated `@measured` snapshot, not cross-checked by a script)*

## Decisions that have no gate yet

This list is debt, and it is here to stay visible rather than to be tidy.

**Held by this file alone: invariants 3 and 4.**

Invariant 1 used to be in that list, and this section asked for a script
asserting that `[project].dependencies` contains exactly `rfc8785`. That script
was already written. `scripts/one-runtime-dependency.sh` is in the gate list
above, it parses `pyproject.toml` with `tomllib`, it fails in both directions
(an added runtime dependency and a disappeared one), and it refuses to pass
when the key is absent entirely rather than measuring nothing. Invariant 1's own
marker said `(gate: ...)` the whole time; this paragraph disagreed with it four
sections later.

Both directions of a wrong marker cost something, and this is the quieter one:
"held by prose alone" on something that has a gate sends the next session to
write a check that exists, and it understates what this repository already
refuses to let you do. Set a marker from evidence, both ways. Before writing
that an invariant has no gate, look in `scripts/`.

Invariant 5 is now `scripts/no-paid-by-default.sh`, and it checks all three of
the ways that invariant currently holds, because losing any one is enough:

1. `verdryx eval --model` is `required=True` with no default, so no invocation
   picks a model for you and none spends without being told to.
2. `build_graders()` with no `judge_adapter` registers no `LLM_JUDGE` grader,
   so the priced grader cannot appear because a caller forgot to opt out.
3. `AnthropicAdapter` has exactly one construction site, behind the explicit
   `model != "stub"` branch, so the priced path stays easy to find.

Points 1 and 3 are read from the AST; point 2 is checked by importing the
package and calling it, because a behavioural claim deserves to be run rather
than read. The script builds a throwaway venv for that import instead of
assuming a prepared machine, since a gate that only runs on one machine is a
gate that does not run.

Verified by breaking three ways: a paid default on `--model`, a second
`AnthropicAdapter` construction site, and `build_graders()` registering the
judge unconditionally.

Invariants 3 and 4 are judgement and probably stay judgement.

Invariant 11's README coverage figure is held by this file alone, same as
invariants 3 and 4: nothing cross-checks it against a fresh run the way
`scripts/readme-numbers.sh` does for the test count. Wiring that up would mean
a gate that reruns the full suite under coverage every time it is asked
whether a number is stale, which is a real cost for a number with no pass/fail
threshold riding on it. If a threshold is ever added (a user decision, not
made here), gating the README figure against it becomes worth that cost.

## Standing rule

An approved architecture decision is **not finished** until it is two things: a
numbered invariant in this file, and a gate in a script if it can be checked
structurally. Until then it is a document, and documents do not stop code.

## Escalate, do not push through

Stop and tell the user, then wait, when a task hits any of these:

- Adding anything to `[project].dependencies`.
- Any change to which graders run by default, or to anything that could reach a
  paid provider.
- Any change to canonicalization or to the event envelope.
- Anything that would make this plane ENFORCE rather than measure: writing a
  wardryx policy, demoting an agent, or gating an action on a budget.
- Cutting a release or publishing to PyPI.

Routine work: tests, doc comments, new deterministic graders, report
formatting, refactors that keep the public API identical.

## Conventions

- **No long dashes** anywhere: not in code comments, docs, commit messages, or
  PR bodies. Use a comma, a colon, parentheses, or a short hyphen.
- Nothing paid or metered gets enabled without telling the user first and
  getting agreement. In this repo that specifically includes the LLM judge.
- Do not delete or revoke keys, tokens, or certificates on your own initiative.
