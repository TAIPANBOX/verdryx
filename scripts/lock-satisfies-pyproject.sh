#!/usr/bin/env bash
# Checks that requirements-dev.lock actually resolves what pyproject.toml
# declares: every direct dependency in the base [project.dependencies] plus
# the dev/test groups CI installs (dev, traces) is pinned in the lock at a
# version meeting its floor.
#
# WHY THIS EXISTS
#
# pyproject.toml keeps `>=` floors as the declared surface (invariant 1), and
# requirements-dev.lock is the exact resolution CI and an operator install
# from. Two ways for those to quietly disagree: a floor in pyproject.toml
# moves and the lock is never regenerated, so CI keeps installing a version
# that no longer satisfies what the project claims to need; or a dependency
# is added to pyproject.toml and nobody regenerates the lock, so the install
# path CI actually uses never resolves it at all. Both look identical to a
# reader of pyproject.toml, which only ever shows the aspiration.
#
# SCOPE
#
# The `anthropic` extra is deliberately excluded: CI never installs it
# (`pip install -e .[dev,traces]`), so pinning it would check a resolution
# nothing ever performs. See requirements-dev.lock's own header. Locked
# groups are the constant below; changing what CI installs means changing
# this list in the same commit.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 - <<'PY'
import pathlib
import re
import sys
import tomllib

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    print("FAIL: the 'packaging' package is not importable, so this check measured nothing")
    sys.exit(1)

LOCKED_GROUPS = ["dev", "traces"]  # must match what CI installs: .[dev,traces]
LOCK_PATH = pathlib.Path("requirements-dev.lock")
PYPROJECT_PATH = pathlib.Path("pyproject.toml")


def parse_requirement(req: str) -> tuple[str, str | None]:
    """Return (package name, floor version or None), via a real PEP 508 parse.

    A hand-rolled split on `>`, `<`, `=`, `~` mis-parses a name carrying one of
    those characters (`foo!=1.0,>=0.5` reads as the name `foo!`) and a version
    compare built the same way is blind to PEP 440 ordering: it drops anything
    after the leading digits of a chunk, so `8.0rc1` and `8.0` compare equal.
    `Requirement` and `Version` get both right.
    """
    requirement = Requirement(req)
    floor = None
    for spec in requirement.specifier:
        if spec.operator == ">=":
            floor = spec.version
    return requirement.name, floor


def version_at_least(actual: str, floor: str) -> bool:
    return Version(actual) >= Version(floor)


if not PYPROJECT_PATH.exists():
    print("FAIL: pyproject.toml is gone, so this check measured nothing")
    sys.exit(1)

data = tomllib.loads(PYPROJECT_PATH.read_text())
project = data.get("project", {})
base_deps = project.get("dependencies")
if base_deps is None:
    print("FAIL: pyproject.toml declares no [project].dependencies at all, so this check measured nothing")
    sys.exit(1)

optional = project.get("optional-dependencies", {})
missing_groups = sorted(g for g in LOCKED_GROUPS if g not in optional)
if missing_groups:
    print(f"FAIL: locked group(s) {missing_groups} are gone from [project.optional-dependencies], "
          f"so this check measured nothing for them")
    sys.exit(1)

wanted: dict[str, str | None] = {}
for req in base_deps:
    name, floor = parse_requirement(req)
    wanted[name] = floor
for group in LOCKED_GROUPS:
    for req in optional.get(group, []):
        name, floor = parse_requirement(req)
        wanted[name] = floor

if not wanted:
    print("FAIL: no dependency found across dependencies/"
          f"{LOCKED_GROUPS}, so this check measured nothing")
    sys.exit(1)

if not LOCK_PATH.exists():
    print(f"FAIL: {LOCK_PATH} does not exist, so this check measured nothing")
    sys.exit(1)

lock_text = LOCK_PATH.read_text()
pinned: dict[str, str] = {}
for line in lock_text.splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-e "):
        continue
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([0-9A-Za-z.\-]+)$", line)
    if m:
        pinned[m.group(1).lower()] = m.group(2)

if not pinned:
    print(f"FAIL: {LOCK_PATH} has no `name==version` pins at all, so this check measured nothing")
    sys.exit(1)

problems = False
for name, floor in sorted(wanted.items()):
    locked_version = pinned.get(name.lower())
    if locked_version is None:
        print(f"FAIL: '{name}' is declared in pyproject.toml but {LOCK_PATH} does not pin it")
        problems = True
        continue
    if floor is not None and not version_at_least(locked_version, floor):
        print(f"FAIL: {LOCK_PATH} pins '{name}' at {locked_version}, "
              f"which is older than pyproject.toml's floor of {floor}")
        problems = True

if problems:
    print()
    print("Regenerate requirements-dev.lock from a clean venv (see its own header")
    print("comment) after changing a floor or adding a dependency.")
    sys.exit(1)

print(f"OK: {len(wanted)} direct dependency name(s) across dependencies/{LOCKED_GROUPS}, "
      f"all present in {LOCK_PATH} and every floor satisfied.")
PY
