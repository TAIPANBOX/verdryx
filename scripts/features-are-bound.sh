#!/usr/bin/env bash
# Checks that every scenario in features/ names a test that exists, and that
# every scenario names one at all.
#
# WHY BOTH DIRECTIONS
#
# A scenario with no binding is a paragraph of prose that describes what
# somebody wanted and proves nothing about what the code does. A binding
# pointing at a test that has been renamed or deleted is worse: it reads as
# held, and the reader has no way to tell without grepping.
#
# WHY NOT A BDD RUNNER
#
# pytest-bdd here, godog in the Go repos, cucumber-rs in the Rust ones:
# three runners and three step-definition styles across the estate, and the
# value asked for here is READABILITY: Given/When/Then that Yurii can read
# instead of a diff. A binding gate delivers that at a fraction of the surface.
# This is my engineering call and a deviation from a literal reading of
# "геркін-тести"; overrule it and I will wire a real runner.
#
# WHAT THIS DOES NOT DO
#
# It does not check that the test ASSERTS what the scenario says. Nothing
# mechanical can. The steps are prose and the binding is a pointer, so a
# scenario can drift from its test and this will stay green. What it catches is
# the pointer breaking, which is the failure that happens on its own.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

if [ ! -d features ]; then
	echo "no features/ directory: nothing to check, and that is not a pass." >&2
	exit 1
fi

fail=0
scenarios=0
bindings=0

# Every Scenario line must be preceded by at least one @test: within the
# lines above it, before the previous Scenario.
while IFS= read -r file; do
	pending=""
	lineno=0
	while IFS= read -r line; do
		lineno=$((lineno + 1))
		case "$line" in
		*@test:*)
			t="${line##*@test:}"
			t="${t%% *}"
			pending="$pending $t"
			bindings=$((bindings + 1))
			# tests/ is where every test in this repository lives; the
			# needle is `def name(` because these are pytest functions
			# rather than Go ones.
			if ! grep -rq "def ${t}(" tests/ 2>/dev/null; then
				printf 'DANGLING  %s:%s\n          @test:%s names no test\n' \
					"$file" "$lineno" "$t"
				fail=$((fail + 1))
			fi
			;;
		*Scenario:*)
			scenarios=$((scenarios + 1))
			if [ -z "$pending" ]; then
				printf 'UNBOUND   %s:%s\n          %s\n          no @test: above it, so it proves nothing\n' \
					"$file" "$lineno" "$(printf '%s' "$line" | sed 's/^ *//')"
				fail=$((fail + 1))
			fi
			pending=""
			;;
		esac
	done <"$file"
done < <(find features -name '*.feature' | sort)

# And the other direction that matters: a feature file with no scenarios at
# all is a file somebody started and left, and it would pass everything above.
for f in features/*.feature; do
	if ! grep -q "Scenario:" "$f"; then
		printf 'EMPTY     %s\n          a feature file with no scenarios\n' "$f"
		fail=$((fail + 1))
	fi
done

echo
if [ "$scenarios" -eq 0 ]; then
	echo "measured nothing: no scenarios found, which is a failure of this" >&2
	echo "script and not a clean bill of health." >&2
	exit 1
fi
printf 'features: %d scenarios, %d bindings, %d broken\n' "$scenarios" "$bindings" "$fail"
[ "$fail" -eq 0 ]
