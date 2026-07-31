#!/usr/bin/env bash
# Enforces invariant 5 of CLAUDE.md: a grader that costs money never runs by
# default.
#
# This is the one invariant here whose failure mode is a bill rather than an
# exception. Nothing crashes, no test goes red, and the first sign is an invoice
# for a run somebody thought was free.
#
# It holds three ways today, and this checks all three, because losing any one
# of them is enough:
#
#   1. `verdryx eval --model` is REQUIRED and has no default. There is no
#      invocation that picks a model for you, so there is no invocation that
#      spends without being told to.
#   2. `build_graders()` with no judge_adapter registers no LLM_JUDGE grader at
#      all. The priced grader cannot appear because a caller forgot to opt out.
#   3. `AnthropicAdapter`, the only thing here that makes a priced outbound
#      call, is constructed in exactly one place, behind an explicit
#      `model != "stub"` branch.
#
# Checks 1 and 3 read the source; check 2 imports the package and asks it. That
# split is deliberate: a structural claim is checked structurally, and a
# behavioural one by running it.
#
# The import needs verdryx's single runtime dependency, so the script builds a
# throwaway venv for it rather than assuming one is set up. A gate that only
# runs on a prepared machine is a gate that does not run.
#
# This file is the ONE copy of this check. The local hook and CI both call it.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

PY="python3"
if [ -x .venv/bin/python ] && .venv/bin/python -c "import rfc8785" 2>/dev/null; then
	PY=.venv/bin/python
elif ! python3 -c "import rfc8785" 2>/dev/null; then
	WORK="$(mktemp -d)"
	trap 'rm -rf "$WORK"' EXIT
	if ! python3 -m venv "$WORK/venv" >/dev/null 2>&1 ||
		! "$WORK/venv/bin/pip" install --quiet rfc8785 >/dev/null 2>&1; then
		echo "FAIL: could not build an environment with rfc8785, so the"
		echo "      behavioural half of this check measured nothing."
		exit 1
	fi
	PY="$WORK/venv/bin/python"
fi

"$PY" - <<'PY'
import ast
import pathlib
import sys

problems = []

# ------------------------------------------------------------------ 1 and 3
cli = pathlib.Path("verdryx/cli.py")
tree = ast.parse(cli.read_text())

model_arg = None
anthropic_sites = []

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        fn = node.func
        # p_eval.add_argument("--model", ...)
        if getattr(fn, "attr", None) == "add_argument" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "--model":
                model_arg = node
        # AnthropicAdapter(...)
        if getattr(fn, "id", None) == "AnthropicAdapter":
            anthropic_sites.append(node.lineno)

if model_arg is None:
    problems.append(
        "verdryx/cli.py no longer declares a --model argument, so nothing was "
        "checked. Do not delete this check to make it pass: work out what "
        "replaced it and check that."
    )
else:
    kw = {k.arg: k.value for k in model_arg.keywords}
    required = kw.get("required")
    if not (isinstance(required, ast.Constant) and required.value is True):
        problems.append(
            "verdryx/cli.py: --model is no longer required=True, so `eval` can "
            "be invoked without naming a model and something has to choose one"
        )
    if "default" in kw:
        d = kw["default"]
        val = d.value if isinstance(d, ast.Constant) else "<expression>"
        if val not in (None, "stub"):
            problems.append(
                f"verdryx/cli.py: --model has a default of {val!r}. A default "
                f"that is not 'stub' spends money for anyone who omits the flag."
            )

if len(anthropic_sites) != 1:
    problems.append(
        f"AnthropicAdapter is constructed in {len(anthropic_sites)} place(s) in "
        f"cli.py (lines {anthropic_sites or 'none'}). It is meant to have "
        f"exactly one construction site, behind the explicit model != 'stub' "
        f"branch, so the priced path stays easy to find and audit."
    )

# ---------------------------------------------------------------------- 2
try:
    sys.path.insert(0, ".")
    from verdryx.graders import build_graders
    from verdryx.models import GraderKind

    default_set = build_graders()
    if GraderKind.LLM_JUDGE in default_set:
        problems.append(
            "build_graders() with no judge_adapter registered an LLM_JUDGE "
            "grader. The priced grader must be absent unless a caller supplies "
            "an adapter on purpose."
        )
    if not default_set:
        problems.append(
            "build_graders() returned nothing at all, so this check measured "
            "nothing"
        )
except Exception as e:  # noqa: BLE001
    problems.append(f"could not import and call build_graders(): {e}")

# -------------------------------------------------------------------------
if problems:
    for p in problems:
        print(f"FAIL: {p}")
    print()
    print("The failure mode of this invariant is a bill, not an exception.")
    print("See CLAUDE.md invariant 5.")
    sys.exit(1)

kinds = sorted(k.name for k in default_set)
print(f"OK: the default grader set is {kinds} and carries no priced judge;")
print("    --model is required with no default; one AnthropicAdapter site.")
PY
