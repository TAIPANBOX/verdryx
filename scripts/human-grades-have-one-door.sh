#!/usr/bin/env bash
# Enforces invariant 14 of CLAUDE.md: a human grade is written through exactly
# one door, and that door checks who is knocking.
#
# WHY THIS IS A GATE AND NOT A TEST
#
# A test of `Store.add_human_grade` proves the door is locked. It cannot prove
# the building has no other door: a second INSERT into `human_grades`, or a
# caller that reaches the door from the upload path, passes every test of the
# first and defeats the rule. "Whoever only uploads results cannot grade" is a
# claim about every path in the package, so it is read from the package.
#
# WHAT IT CHECKS, all from the AST
#
#   1. A string that INSERTs into human_grades exists, and exists exactly once.
#   2. It sits inside `Store.add_human_grade`.
#   3. That function checks the credential (`_verified_principal`) and the role
#      (`ROLE_REVIEWER`) before it writes.
#   4. `add_human_grade` is called from `_cmd_grade` in verdryx/cli.py and from
#      nowhere else in the package: in particular not from `run_eval`,
#      `_cmd_eval` or any other path results come in by.
#
# WHAT IT DOES NOT DO
#
# It cannot see SQL assembled at run time, or the database file being written
# by hand: whoever can write the SQLite file can write anything in it, and
# that file is the trust boundary. It also cannot tell that the checks in (3)
# are correct, only that they are there; the tests hold that.
#
# Like its siblings it refuses to pass when it measured nothing: no door found
# is a failure, not a clean result.

set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import ast
import pathlib
import re
import sys

INSERT = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+human_grades", re.IGNORECASE)
DOOR = "add_human_grade"
CALLER = ("verdryx/cli.py", "_cmd_grade")

fail = False
sites: list[tuple[str, int, str | None]] = []
door_nodes: list[tuple[str, ast.FunctionDef]] = []
calls: list[tuple[str, int, str | None]] = []


def enclosing(tree: ast.AST, lineno: int) -> str | None:
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno >= best.lineno):
                best = node
    return best.name if best else None


for path in sorted(pathlib.Path("verdryx").rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if INSERT.search(node.value):
                sites.append((str(path), node.lineno, enclosing(tree, node.lineno)))
        if isinstance(node, ast.FunctionDef) and node.name == DOOR:
            door_nodes.append((str(path), node))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == DOOR
        ):
            calls.append((str(path), node.lineno, enclosing(tree, node.lineno)))

if not door_nodes or not sites:
    print(
        f"FAIL: measured nothing: found {len(door_nodes)} definition(s) of {DOOR} and "
        f"{len(sites)} INSERT into human_grades; the gate cannot say the door is the only one"
    )
    sys.exit(1)

if len(sites) > 1:
    for path, lineno, fn in sites:
        print(f"FAIL: {path}:{lineno} INSERTs into human_grades inside {fn or 'module level'}")
    print(f"FAIL: {len(sites)} INSERT sites: there must be exactly one, in {DOOR}")
    fail = True

for path, lineno, fn in sites:
    if fn != DOOR:
        print(
            f"FAIL: {path}:{lineno} INSERTs into human_grades inside {fn or 'module level'}, "
            f"a second place besides {DOOR}"
        )
        fail = True

for path, node in door_nodes:
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    if "_verified_principal" not in attrs:
        print(f"FAIL: {path}: {DOOR} does not check the credential (_verified_principal)")
        fail = True
    if "ROLE_REVIEWER" not in names:
        print(f"FAIL: {path}: {DOOR} does not check the role (ROLE_REVIEWER)")
        fail = True

for path, lineno, fn in calls:
    if (path, fn) != CALLER:
        print(
            f"FAIL: {path}:{lineno} calls {DOOR} from {fn or 'module level'}, outside "
            f"{CALLER[1]}: results come in by other paths and none of them may grade"
        )
        fail = True
if not any((p, f) == CALLER for p, _l, f in calls):
    print(f"FAIL: nothing calls {DOOR} from {CALLER[1]}: the grade command lost its door")
    fail = True

if fail:
    sys.exit(1)
print(
    f"OK: one INSERT into human_grades, inside {DOOR}; it checks the credential and the "
    f"role; its only caller is {CALLER[1]}."
)
PY
