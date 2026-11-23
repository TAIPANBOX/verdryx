#!/usr/bin/env bash
# Enforces invariant 3 of CLAUDE.md: RFC 8785 canonicalization is delegated to
# `rfc8785`, never hand-rolled.
#
# WHY
#
# The chain hash every event carries (`prev_hash`, agent-passport SPEC.md Sec
# 6.5) is sha256 over the canonical bytes of the previous event. The Go side of
# the stack computes the same bytes with gowebpki/jcs. Two implementations of a
# canonical form always disagree eventually, and the first disagreement breaks
# every chain that crosses the language boundary, silently: each side verifies
# its own events and rejects the other's.
#
# The tempting regression is a few lines long and looks harmless:
# `json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()` reads like a
# canonical form and is not one (number formatting, escaping, key order by code
# point rather than UTF-16 unit). So this reads the package with Python's own
# AST, not a regexp, for three things:
#
#   1. `verdryx/events.py` defines `canonicalize`, and it returns
#      `rfc8785.dumps(...)`.
#   2. `chain_hash` hashes `canonicalize(...)` and nothing else.
#   3. Nowhere in `verdryx/` is a hashlib digest taken over `json.dumps`
#      output, and nowhere is `json.dumps` called with both `sort_keys=True`
#      and `separators=`, the shape a hand-rolled canonical form takes.
#
# WHAT IT DOES NOT DO
#
# `json.dumps(..., sort_keys=True)` alone is allowed: `store.py` writes finding
# details that way for storage, and nothing hashes those bytes. A hand-rolled
# canonical form written some other way (a custom encoder, string building)
# would not be seen. It refuses to report OK when `canonicalize` or
# `chain_hash` is gone, because a check whose subject has vanished measured
# nothing.
#
# This file is the ONE copy of this check.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 - <<'PY'
import ast
import pathlib
import sys

problems = []


def is_call_to(node, owner, attr):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def is_json_dumps(node):
    return is_call_to(node, "json", "dumps")


def is_hashlib_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "hashlib"
    )


events = pathlib.Path("verdryx/events.py")
if not events.is_file():
    print("FAIL: verdryx/events.py is gone, so this check measured nothing")
    sys.exit(1)
tree = ast.parse(events.read_text(), filename=str(events))
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

canon = functions.get("canonicalize")
if canon is None:
    print("FAIL: verdryx/events.py defines no canonicalize(), so this check measured nothing")
    sys.exit(1)
returns = [n.value for n in ast.walk(canon) if isinstance(n, ast.Return)]
if not returns or not all(is_call_to(r, "rfc8785", "dumps") for r in returns):
    problems.append(
        "verdryx/events.py: canonicalize() does not return rfc8785.dumps(...); "
        "the canonical form is no longer delegated"
    )

chain = functions.get("chain_hash")
if chain is None:
    print("FAIL: verdryx/events.py defines no chain_hash(), so this check measured nothing")
    sys.exit(1)
digests = [n for n in ast.walk(chain) if is_hashlib_call(n)]
if not digests:
    problems.append("verdryx/events.py: chain_hash() takes no hashlib digest at all")
for call in digests:
    arg = call.args[0] if call.args else None
    if not (
        isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "canonicalize"
    ):
        problems.append(
            f"verdryx/events.py:{call.lineno}: chain_hash() hashes something other than "
            "canonicalize(...)"
        )

for path in sorted(pathlib.Path("verdryx").rglob("*.py")):
    module = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if is_json_dumps(node):
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            sort_keys = kw.get("sort_keys")
            if (
                isinstance(sort_keys, ast.Constant)
                and sort_keys.value is True
                and "separators" in kw
            ):
                problems.append(
                    f"{path}:{node.lineno}: json.dumps with sort_keys=True and separators= "
                    "is a hand-rolled canonical form"
                )
        if is_hashlib_call(node):
            for sub in ast.walk(node):
                if sub is not node and is_json_dumps(sub):
                    problems.append(
                        f"{path}:{node.lineno}: a hashlib digest over json.dumps output; "
                        "hash canonicalize(...) instead"
                    )

if problems:
    for p in problems:
        print(f"FAIL: {p}")
    print()
    print("RFC 8785 canonicalization is delegated to rfc8785, never hand-rolled:")
    print("it has to agree byte for byte with gowebpki/jcs on the Go side.")
    print("See CLAUDE.md invariant 3.")
    sys.exit(1)

print("OK: canonicalize() returns rfc8785.dumps, chain_hash() hashes canonicalize(),")
print("    and no json.dumps output is hashed or shaped as a canonical form in verdryx/.")
PY
