"""The declaration in components.json is only worth reading if this repository
proves it, and proves it by RUNNING rather than by describing.

estate-gates cannot do this. It has no Python toolchain, and building
twenty-two repositories in its CI is a matrix it does not have. This repository
already runs pytest on every push.

What is proved here is exactly the `checked` bucket and nothing else. The
`declared` bucket is not asserted against anything, on purpose: a test that
pretended to verify a sentence about purpose would be the failure this whole
design exists to avoid.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"VERDRYX_[A-Z0-9_]+")


def manifest() -> dict:
    return json.loads((ROOT / "components.json").read_text())


def components() -> list[dict]:
    cs = manifest()["components"]
    assert cs, "components.json declares nothing, so every test here measured nothing"
    return cs


def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_every_console_script_this_package_installs_is_declared_and_the_reverse():
    """THE ONE THAT CLOSES THE HOLE.

    In Python the packaging metadata IS the answer to "what does this install",
    and it is already a cross-repository contract: stack-k8s's console.Dockerfile
    installs the Python tools by reading `pyproject [project.scripts]` and says
    so in a comment. A script added here without a manifest entry is a component
    the console image would install and nothing would have declared.
    """
    scripts = pyproject()["project"].get("scripts") or {}
    assert scripts, "pyproject declares no console script, so this measured nothing"

    declared = {
        c["checked"]["console_script"] for c in components() if "console_script" in c["checked"]
    }
    assert declared, "no component declares a console script, so this measured nothing"

    for name in scripts:
        assert name in declared, (
            f"pyproject installs the script {name!r} and components.json does not "
            f"declare it. A component nobody declares is one no deployment can be "
            f"asked to install."
        )
    for name in sorted(declared):
        assert name in scripts, (
            f"components.json declares the script {name!r} and pyproject installs no such thing"
        )

    # And the entry point each component names is the one packaging would call.
    for c in components():
        if "entry_point" not in c["checked"]:
            continue
        script = c["checked"]["console_script"]
        assert c["checked"]["entry_point"] == scripts[script], (
            f"components.json says {script!r} enters at {c['checked']['entry_point']!r}; "
            f"pyproject says {scripts[script]!r}"
        )


def test_every_environment_variable_this_repository_reads_is_declared_and_the_reverse():
    """Every VERDRYX_ name in non-test source against every one declared.

    It reads STRING LITERALS rather than following os.environ, for the same
    reason its siblings do: a helper between the two hides the name from a reader
    that follows call sites.
    """
    declared: set[str] = set()
    for c in components():
        declared |= set(c["checked"].get("env", {}))
    assert declared, "no component declares an environment variable, so this measured nothing"

    in_source: set[str] = set()
    for p in ROOT.rglob("*.py"):
        s = str(p.relative_to(ROOT))
        if s.startswith((".venv", "tests/")) or "/tests/" in s:
            continue
        in_source |= {n for n in NAME.findall(p.read_text(errors="ignore")) if not n.endswith("_")}
    assert in_source, "no VERDRYX_ name found in non-test source, so this measured nothing"

    missing = sorted(in_source - declared)
    extra = sorted(declared - in_source)
    assert not missing, f"the code reads these and components.json declares none of them: {missing}"
    assert not extra, f"components.json declares these and no non-test source reads them: {extra}"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verdryx.cli", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_every_declared_subcommand_is_one_the_cli_accepts():
    """AND THE HALF NO CENTRAL FILE COULD EVER DO: ask the CLI itself.

    Not a grep for `add_parser`. A subcommand that parses is one the CLI answers
    `--help` for, and one it does not know exits with argparse's usage error
    naming the invalid choice.
    """
    subs: set[str] = set()
    for c in components():
        subs |= set(c["checked"].get("subcommands", []))
        if "subcommand" in c["checked"]:
            subs.add(c["checked"]["subcommand"])
    assert subs, "no component declares a subcommand, so this measured nothing"

    for sub in sorted(subs):
        got = _run(sub, "--help")
        assert got.returncode == 0, (
            f"components.json declares the subcommand {sub!r} and `verdryx {sub} --help` "
            f"exited {got.returncode}:\n{got.stderr}"
        )

    # The reader has to be able to tell a real subcommand from a made-up one, or
    # the loop above proves only that --help never fails.
    bogus = _run("definitely-not-a-subcommand", "--help")
    assert bogus.returncode != 0, (
        "`verdryx definitely-not-a-subcommand --help` succeeded, so the check above "
        "would pass for any string at all"
    )


def test_a_missing_required_argument_exits_with_the_declared_code():
    """The exit code a ROUTINE reads.

    stack-up's routine treats a non-zero exit as an error to show an operator, so
    the difference between "you did not give me a baseline" and "the drift check
    failed" is a cross-repository fact. mockryx and qryx use 1 and 2 to mean
    different things again; nothing makes the three agree, and this at least
    makes each say which it is.
    """
    checked = 0
    for c in components():
        want = c["checked"].get("missing_argument_exit_code")
        sub = c["checked"].get("subcommand")
        if want is None or sub is None:
            continue
        checked += 1
        got = _run(sub)
        assert got.returncode == want, (
            f"`verdryx {sub}` with no arguments exited {got.returncode}; "
            f"components.json says {want}\n{got.stderr}"
        )
        # It has to say WHICH argument, or an operator reads a usage block and guesses.
        assert "required" in got.stderr.lower(), (
            f"`verdryx {sub}` refused without saying what was required:\n{got.stderr}"
        )
    if not checked:
        pytest.fail(
            "no component declares both a subcommand and an exit code, so this measured nothing"
        )
