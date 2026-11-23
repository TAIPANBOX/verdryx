#!/usr/bin/env bash
# Enforces the half of invariant 4 of CLAUDE.md that text can hold: verdryx
# reads traces, it does not produce them.
#
# WHY
#
# The outcome-tagged Parquet traces are TokenFuse's output and its schema.
# The day verdryx starts writing Parquet of its own, two producers share one
# format, and the first column one adds that the other does not know about is
# a fork nobody decided on. If a field is missing, the fix is in TokenFuse.
#
# So this reads the package with Python's own AST, not a regexp (a docstring
# that says "TokenFuse's ParquetSink writes segments" must not trip it), for
# any call that writes Parquet or Arrow data: `write_table`, `write_dataset`,
# `write_to_dataset`, `write_metadata`, `write_feather`, `to_parquet`, and any
# construction of a `ParquetWriter`. And it requires the reader to still be
# there: a `read_table` call in `verdryx/costper.py`. A gate whose subject has
# gone measured nothing, and says so.
#
# WHAT IT DOES NOT DO
#
# The other half of invariant 4, "a locally invented column", is judgement: a
# column read under a name TokenFuse never wrote is a reading mistake, not a
# write, and no text check can tell a new field from a renamed one. Tests and
# fixtures may write Parquet; only `verdryx/` is read.
#
# This file is the ONE copy of this check.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 - <<'PY'
import ast
import pathlib
import sys

WRITERS = {
    "write_table",
    "write_dataset",
    "write_to_dataset",
    "write_metadata",
    "write_feather",
    "to_parquet",
    "ParquetWriter",
}


def called_name(node):
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


problems = []
reads = 0
for path in sorted(pathlib.Path("verdryx").rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = called_name(node)
        if name in WRITERS:
            problems.append(f"{path}:{node.lineno}: calls {name}(), which writes trace data")
        if name == "read_table" and path.as_posix() == "verdryx/costper.py":
            reads += 1

if reads == 0:
    print("FAIL: verdryx/costper.py no longer calls read_table(), so this check")
    print("      has no trace reader to hold to reading, and measured nothing")
    sys.exit(1)

if problems:
    for p in problems:
        print(f"FAIL: {p}")
    print()
    print("Verdryx reads traces, it does not produce them: the Parquet traces are")
    print("TokenFuse's output and its schema. See CLAUDE.md invariant 4.")
    sys.exit(1)

print(f"OK: verdryx/ writes no trace data, and costper.py still reads it ({reads} read_table call).")
PY
