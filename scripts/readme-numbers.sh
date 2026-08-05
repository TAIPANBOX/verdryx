#!/usr/bin/env bash
# Every number this README states about this repository, checked against the
# repository.
#
# WHY THIS EXISTS
#
# A number on a README is a claim with no owner. It is right the day it is
# written, and nothing tells anybody when it stops being right, because the
# suite grows in commits that never open the README.
#
# Not hypothetical, and this repository was one of the four. On 2026-08-05 the
# it-rat.com service pages were audited against the repositories they describe
# and FOUR OF SEVEN figures were stale: trailryx by 33 tests, tokenfuse by 196,
# engram by 42, and **verdryx by 75, where the page said 217 and pytest collects
# 292**. None was wrong when written. That is the whole problem.
#
# WHAT IS COUNTED, because a number needs a definition more than it needs a
# badge
#
# `pytest --collect-only -q` reports the number of collected test items. That
# counts PARAMETRISED CASES separately, because pytest does: a function decorated
# with five `@pytest.mark.parametrize` values is five items, and five is what a
# contributor sees when the suite runs. It is deliberately not the count of
# `def test_` lines, which is smaller and which nobody would arrive at by
# running anything.
#
# Collection, not execution. This says how much test code exists, not that it
# passes: `pytest` in CI is what says that, and conflating the two would let a
# green badge mean a red suite. Collection does still import every test module,
# so a syntax error or a missing dependency fails here as well, which is the
# right kind of noisy.
#
# It needs the project's own extras installed, exactly as CI installs them
# (`pip install -e .[dev,traces]`). Without them collection stops on an import
# and this check says so rather than reporting a smaller number as if it were
# the answer.

set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

readme="README.md"
problems=0

note() {
	printf '%s\n' "$1"
	problems=$((problems + 1))
}

out=$(python -m pytest --collect-only -q 2>&1)
actual=$(printf '%s\n' "$out" | grep -oE '^[0-9]+ tests? collected' | grep -oE '^[0-9]+')

if [ -z "${actual:-}" ]; then
	note "pytest did not report a collected count, so this check measured nothing"
	printf '%s\n' "$out" | tail -5
	exit 1
fi

# pytest's own summary line, not the word "error" anywhere in the output. The
# first version matched the latter and failed on a healthy run, because `-q`
# prints test IDs and this suite has tests with "error" in their names. A check
# that fires on a passing repository is worse than no check: it teaches people
# to ignore it.
if printf '%s\n' "$out" | grep -qE '^!+ Interrupted|errors? during collection|^ERROR '; then
	note "pytest reported errors during collection, so $actual is a floor and not the count"
	printf '%s\n' "$out" | grep -E 'error|ModuleNotFound' | head -3
	printf 'Install what CI installs: pip install -e ".[dev,traces]"\n'
	exit 1
fi

stated=$(grep -o 'badge/tests-[0-9]*-' "$readme" | grep -o '[0-9]*' | head -1)
if [ -z "$stated" ]; then
	note "the README carries no tests badge, so this check has nothing to compare against"
	note "add: ![tests](https://img.shields.io/badge/tests-${actual}-brightgreen)"
	exit 1
fi

[ "$stated" = "$actual" ] ||
	note "the badge says $stated tests and pytest collects $actual"

if [ "$problems" -gt 0 ]; then
	printf '\n%d number(s) the README states that this repository does not support.\n' "$problems"
	printf 'Update the badge in the same commit as the tests. That is the point: the\n'
	printf 'suite changes in a commit that never opens the README, and this is what\n'
	printf 'makes that impossible.\n'
	exit 1
fi

printf '%s tests collected, and the badge says so.\n' "$actual"
