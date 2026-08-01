#!/usr/bin/env bash
# Enforces invariant 1 of CLAUDE.md: the runtime core has exactly one
# dependency, rfc8785.
#
# Everything else is an optional extra: anthropic for the priced LLM judge,
# pyarrow for reading Parquet traces, plus the dev group. Installing verdryx
# must not drag an LLM SDK or Arrow onto a machine that only needs the
# deterministic graders.
#
# The regression this catches is specific and tempting: an import fails, the
# quickest fix is to move the extra into the base list, everything goes green,
# and from then on every install of this package pulls a model SDK. Nothing
# announces that.
#
# scripts/optional-imports.sh holds the other half, that an extra is imported
# lazily. This one holds the declaration.
#
# This file is the ONE copy of this check.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 - <<'PY'
import pathlib
import sys
import tomllib

ALLOWED = {"rfc8785"}

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
deps = data.get("project", {}).get("dependencies")
if deps is None:
    print("FAIL: pyproject.toml declares no [project].dependencies at all, so this check measured nothing")
    sys.exit(1)

names = {d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("~")[0].strip() for d in deps}

extra = sorted(names - ALLOWED)
missing = sorted(ALLOWED - names)
problems = False

for n in extra:
    print(f"FAIL: '{n}' is a runtime dependency. Installing verdryx must not drag")
    print( "      it onto a machine that only needs the deterministic graders.")
    print( "      Put it in [project.optional-dependencies] and import it lazily.")
    problems = True
for n in missing:
    print(f"FAIL: '{n}' is gone from the runtime dependencies. Either that was")
    print( "      deliberate, in which case update this script and CLAUDE.md")
    print( "      invariant 1 together, or it was an accident.")
    problems = True

if problems:
    print()
    print("Adding a runtime dependency is a decision for the user, not a convenience.")
    print("See CLAUDE.md invariant 1.")
    sys.exit(1)

optional = data.get("project", {}).get("optional-dependencies", {})
print(f"OK: exactly {len(names)} runtime dependency ({', '.join(sorted(names))}); "
      f"{len(optional)} optional group(s) kept out of it.")
PY
