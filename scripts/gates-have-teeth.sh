#!/usr/bin/env bash
# Checks that the gates in `scripts/` still FAIL on the faults they exist to
# catch, still PASS on what they must not catch, and REFUSE to report success
# when they measured nothing at all.
#
# WHY
#
# Every gate here parses text, and a text parser does not break loudly: it
# stops matching and reports success. The mutants that proved each one existed
# as prose, in commit messages and in the `*(gate: ...)*` markers in CLAUDE.md,
# which is a record of what was true once. Nothing ran them again.
#
# A gate that has quietly stopped catching anything looks exactly like a gate
# with nothing to catch, and stays that way until the fault it guards ships.
#
# WHY THE THIRD PROPERTY IS SEPARATE FROM THE FIRST
#
# Three of these four gates already refuse when their subject is absent: no
# declared dependencies at all, pytest reporting no collected count, an
# environment that could not be built. Every one of those sentences was true,
# was established by hand once in the session that wrote the script, and
# nothing re-ran them. A check that cannot tell "did not fail" from
# "did not run" is the most expensive recurring mistake in this estate's
# tooling, and it lives in tooling rather than product code because tooling is
# where a silent pass looks like a result.
#
# HOW IT MUTATES WITHOUT LEAVING A MESS
#
# It edits tracked files in place, so it refuses to start unless the tree is
# clean, restores with `git checkout` after every case, restores again from a
# trap on any exit path including a kill, and asserts the tree is clean before
# reporting success.
#
# A MUTATION THAT DID NOT APPLY PROVES NOTHING
#
# Every edit asserts it changed the file. A case whose edit applied nothing is
# a failure here, not a pass. That is not hypothetical: five such mutations
# were caught across idryx and tokenfuse on 2026-08-09, and three of the five
# had been verified BY HAND against the same gate minutes earlier. The hand
# version and the harness version differ only in how many layers of quoting sit
# between the text and python, which is exactly the difference nobody sees.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# The gates here collect the test suite and build an environment, so they need
# the package installed. Without it they report "measured nothing", correctly,
# and every case below would fail for that reason rather than for the one it is
# about, which is a worse diagnosis than the fault.
if ! python3 -c 'import rfc8785, pytest' 2>/dev/null; then
	printf 'this repo\x27s gates need the package installed to measure anything.\n'
	printf 'they say so themselves rather than passing quietly, which is the\n'
	printf 'property this harness checks, so it cannot run without it either:\n'
	printf '    pip install -e ".[dev]"\n'
	exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
	printf 'this script mutates tracked files, so it needs a clean tree.\n'
	printf 'commit or stash first; it restores with `git checkout` and cannot\n'
	printf 'tell your edits from its own.\n'
	exit 1
fi

# Untracked files too: a mutation may RENAME a tracked file, and `git checkout`
# restores the original while leaving the new name behind. And the INDEX, since
# a gate may read `git ls-files` rather than the disk, so a mutation has to move
# the file in both. Safe because this
# script refuses to start unless the tree is clean, so anything untracked
# during a run was created by the run. `-x` is deliberately absent: ignored
# build output is not ours to delete.
restore() {
	git reset -q --hard HEAD 2>/dev/null
	git clean -fdq 2>/dev/null
}
trap restore EXIT INT TERM

failures=0
cases=0

# run_case <name> <expect: fail|pass> <gate> <python edit> [required output]
#
# The needle separates "it failed" from "it failed for the reason this case is
# about". Without it, a case expecting failure is satisfied by any failure,
# including one this harness caused itself.
run_case() {
	local name="$1" expect="$2" gate="$3" edit="$4" needle="${5:-}"
	cases=$((cases + 1))

	if ! python3 -c "$edit"; then
		printf 'BROKEN  %s\n        its mutation did not apply, so this case proved nothing\n' "$name"
		failures=$((failures + 1))
		restore
		return
	fi

	local out rc
	out=$(eval "$gate" 2>&1)
	rc=$?
	restore

	# Exit code first, then wording. Checking the needle before the expectation
	# turns "it did not fail at all" into "it failed for the wrong reason",
	# which sends the reader to look at prose when the gate is toothless.
	if [ "$expect" = fail ] && [ "$rc" -ne 0 ] && [ -n "$needle" ] &&
		! printf '%s' "$out" | grep -qF -- "$needle"; then
		printf 'WRONG REASON  %s\n              it failed, but not saying: %s\n' "$name" "$needle"
		failures=$((failures + 1))
		return
	fi
	if [ "$expect" = fail ] && [ "$rc" -eq 0 ]; then
		printf 'TOOTHLESS  %s\n           the gate passed on a fault it exists to catch\n' "$name"
		failures=$((failures + 1))
	elif [ "$expect" = pass ] && [ "$rc" -ne 0 ]; then
		printf 'OVEREAGER  %s\n           the gate failed on something it must not catch\n' "$name"
		failures=$((failures + 1))
		printf '%s\n' "$out" | head -4 | sed 's/^/           /'
	else
		printf 'ok  %-58s (%s)\n' "$name" "$expect"
	fi
}

py() { printf 'def edit(p, a, b):\n    s = open(p).read()\n    assert a in s, "pattern not found in " + p\n    open(p, "w").write(s.replace(a, b, 1))\n%s\n' "$1"; }

echo "=== faults each gate must catch ==="

# The invariant is that installing verdryx drags in nothing but rfc8785.
run_case "one-runtime-dependency: a second runtime dependency" fail \
	'./scripts/one-runtime-dependency.sh' \
	"$(py 'edit("pyproject.toml", "dependencies = [\"rfc8785>=0.1.4\"]", "dependencies = [\"rfc8785>=0.1.4\", \"pyarrow>=14.0\"]")')" \
	"is a runtime dependency"

# The reverse: the allow-list describing a package that no longer exists.
run_case "one-runtime-dependency: the one dependency gone" fail \
	'./scripts/one-runtime-dependency.sh' \
	"$(py 'edit("pyproject.toml", "dependencies = [\"rfc8785>=0.1.4\"]", "dependencies = [\"pyarrow>=14.0\"]")')" \
	"is gone from the runtime dependencies"

run_case "optional-imports: an optional dependency imported at module level" fail \
	'./scripts/optional-imports.sh' \
	"$(py 'edit("verdryx/cli.py", "import argparse", "import argparse\nimport anthropic")')" \
	"imports optional"

# A priced model reaching the default path is what this gate is for, and
# `required=True` with no default is what keeps it off that path.
run_case "no-paid-by-default: --model stops being required" fail \
	'./scripts/no-paid-by-default.sh' \
	"$(py 'edit("verdryx/cli.py", "        \"--model\",\n        required=True,", "        \"--model\",\n        required=False,")')" \
	"no longer required=True"

run_case "no-paid-by-default: --model acquires a default" fail \
	'./scripts/no-paid-by-default.sh' \
	"$(py 'edit("verdryx/cli.py", "        \"--model\",\n        required=True,", "        \"--model\",\n        required=True,\n        default=\"claude-opus-5\",")')" \
	"has a default of"

run_case "readme-numbers: a stale test badge" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'import re
s = open("README.md").read()
m = re.search(r"badge/tests-(\d+)-", s)
assert m, "no test badge in README.md"
open("README.md","w").write(s.replace(m.group(0), "badge/tests-%d-" % (int(m.group(1))+7), 1))')" \
	"badge"

echo
echo "=== and what they must NOT catch ==="

# The lazy import is the whole point of the optional-dependency invariant: the
# adapter may exist, it may not be imported until somebody asks for it.
run_case "optional-imports: an optional dependency imported inside a function" pass \
	'./scripts/optional-imports.sh' \
	"$(py 'edit("verdryx/cli.py", "import argparse", "import argparse\n\n\ndef _unused_lazy_import():\n    import anthropic\n\n    return anthropic\n")')"

run_case "one-runtime-dependency: another optional group added" pass \
	'./scripts/one-runtime-dependency.sh' \
	"$(py 'edit("pyproject.toml", "traces = [\"pyarrow>=14.0\"]", "traces = [\"pyarrow>=14.0\"]\nspare = [\"jsonschema>=4.20\"]")')"

echo
echo "=== and the one this estate learned the hard way ==="
echo "    a gate whose subject is gone must SAY so, not report OK on nothing"

run_case "one-runtime-dependency: no declared dependencies at all" fail \
	'./scripts/one-runtime-dependency.sh' \
	"$(py 'edit("pyproject.toml", "dependencies = [\"rfc8785>=0.1.4\"]", "")')" \
	"measured nothing"

run_case "readme-numbers: no badge left to compare against" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'import re
s = open("README.md").read()
m = re.search(r"badge/tests-\d+-", s)
assert m, "no test badge in README.md"
open("README.md","w").write(s.replace(m.group(0), "badge/nothing-", 1))')" \
	"nothing to compare against"

echo
if [ -n "$(git status --porcelain)" ]; then
	printf 'FAIL: this script left the tree dirty, so it cannot be trusted about anything above\n'
	git status --porcelain | head -5
	exit 1
fi

if [ "$failures" -gt 0 ]; then
	printf '%d of %d cases failed.\n' "$failures" "$cases"
	printf 'A gate that has quietly stopped catching anything looks exactly like a gate\n'
	printf 'with nothing to catch, and stays that way until the fault it guards ships.\n'
	exit 1
fi

printf 'OK: %d cases. Every gate fails on its own fault, passes on a non-fault,\n' "$cases"
printf '    and refuses to report success when it measured nothing.\n'
