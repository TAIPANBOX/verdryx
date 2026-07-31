#!/usr/bin/env bash
# Enforces invariant 2 of CLAUDE.md: an optional dependency is imported lazily,
# inside the function that needs it, never at module top level.
#
# Why this matters more than it looks. verdryx declares exactly one runtime
# dependency, rfc8785. anthropic and pyarrow are optional extras. A top-level
# `import pyarrow` in any module turns that extra into a hard requirement for
# everyone who imports the module, and the failure surfaces as an ImportError on
# a code path that has nothing to do with Parquet.
#
# The check uses Python's own AST, not a regexp, because indentation is the
# whole distinction here and a regexp on leading whitespace would be fooled by
# an import inside a class body or a try block at module scope.
#
# This file is the ONE copy of this check. The local hook and CI both call it.
# Two copies of one check always diverge, so do not inline it anywhere.

set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import ast
import pathlib
import sys

# Optional extras, from pyproject.toml [project.optional-dependencies].
OPTIONAL = {"anthropic", "pyarrow"}

fail = False

for path in sorted(pathlib.Path("verdryx").rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))

    # Module-level statements only. An import nested inside a function or method
    # is exactly what this invariant asks for, so we do not walk into those.
    for node in tree.body:
        # A top-level `try: import pyarrow` is still top level.
        candidates = [node]
        if isinstance(node, ast.Try):
            candidates = list(node.body) + list(node.orelse) + list(node.finalbody)

        for stmt in candidates:
            names = []
            if isinstance(stmt, ast.Import):
                names = [a.name for a in stmt.names]
            elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                names = [stmt.module]

            for name in names:
                root = name.split(".")[0]
                if root in OPTIONAL:
                    print(
                        f"FAIL: {path}:{stmt.lineno} imports optional "
                        f"dependency '{root}' at module level"
                    )
                    fail = True

if fail:
    print()
    print("Optional extras must be imported inside the function that needs them,")
    print("wrapped in try/except ImportError that re-raises with a pip hint.")
    print("See CLAUDE.md invariant 2, and costper.py / graders.py for the shape.")
    sys.exit(1)

print("OK: no optional dependency is imported at module level.")
PY
