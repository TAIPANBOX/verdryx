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
   not packaging trivia. See invariants 1 and 2.
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
```

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

## Decisions that have no gate yet

This list is debt, and it is here to stay visible rather than to be tidy.

**Held by this file alone: invariants 1, 3 and 4.**

Invariant 1 is mechanically checkable and should become a script: assert that
`[project].dependencies` in `pyproject.toml` contains exactly `rfc8785`. It is
one line of parsing and it would catch the most likely regression, which is
somebody moving an extra into the base list to make an import work.

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
- Cutting a release or publishing to PyPI.

Routine work: tests, doc comments, new deterministic graders, report
formatting, refactors that keep the public API identical.

## Conventions

- **No long dashes** anywhere: not in code comments, docs, commit messages, or
  PR bodies. Use a comma, a colon, parentheses, or a short hyphen.
- Nothing paid or metered gets enabled without telling the user first and
  getting agreement. In this repo that specifically includes the LLM judge.
- Do not delete or revoke keys, tokens, or certificates on your own initiative.
