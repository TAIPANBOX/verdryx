"""Tests for verdryx.config, in particular where the store ends up.

The resolution order matters beyond this repo: verdryx has no server, so its
SQLite file is its machine-readable surface, and the Genaryx console reads it
from the same three candidates in the same order. A mismatch here does not
lose data, it hides it, which for a plane whose whole job is to be looked at
is the same thing.
"""

from __future__ import annotations

import os

from verdryx.config import (
    DEFAULT_DB,
    Config,
    default_taipan_home,
    resolve_db_path,
)

# ------------------------------------------------------------------
# default_taipan_home
# ------------------------------------------------------------------


def test_taipan_home_env_wins() -> None:
    env = {"TAIPAN_HOME": "/opt/taipan", "HOME": "/Users/someone"}
    assert default_taipan_home(env) == "/opt/taipan"


def test_taipan_home_defaults_under_user_home() -> None:
    env = {"HOME": "/Users/someone"}
    assert default_taipan_home(env) == os.path.join("/Users/someone", ".taipan")


# ------------------------------------------------------------------
# resolve_db_path
# ------------------------------------------------------------------


def test_explicit_db_env_wins_over_an_installed_home(tmp_path) -> None:
    """An explicit override is honoured verbatim, and is not second-guessed.

    Even with a perfectly good published home sitting right there, and even
    though the named path does not exist: an operator who names a path must
    get an honest failure on THAT path later, never a silent redirect to a
    different store.
    """
    home = tmp_path / ".taipan"
    home.mkdir()
    env = {"VERDRYX_DB": "/nowhere/custom.db", "HOME": str(tmp_path)}
    assert resolve_db_path(env) == "/nowhere/custom.db"


def test_published_home_used_when_it_exists(tmp_path) -> None:
    home = tmp_path / ".taipan"
    home.mkdir()
    env = {"HOME": str(tmp_path)}
    assert resolve_db_path(env) == str(home / "verdryx.db")


def test_taipan_home_override_is_followed(tmp_path) -> None:
    """A scratch home is the whole point of TAIPAN_HOME.

    stack-up honours the same variable so an entire install can be pointed
    somewhere disposable; verdryx has to follow it or a clean-machine test
    writes its results into the real home it was trying not to touch.
    """
    scratch = tmp_path / "scratch-home"
    scratch.mkdir()
    env = {"TAIPAN_HOME": str(scratch), "HOME": str(tmp_path)}
    assert resolve_db_path(env) == str(scratch / "verdryx.db")


def test_falls_back_to_the_working_directory_when_nothing_is_installed(
    tmp_path,
) -> None:
    """No published home means no stack installed: keep the old behaviour.

    The directory is deliberately NOT created here, so this also pins that
    resolution never creates anything as a side effect of asking a question.
    """
    env = {"HOME": str(tmp_path)}
    assert resolve_db_path(env) == DEFAULT_DB
    assert not (tmp_path / ".taipan").exists()


def test_a_home_that_is_a_file_is_not_a_home(tmp_path) -> None:
    """`isdir`, not `exists`: a stray file named .taipan is not an install."""
    (tmp_path / ".taipan").write_text("not a directory")
    env = {"HOME": str(tmp_path)}
    assert resolve_db_path(env) == DEFAULT_DB


# ------------------------------------------------------------------
# Config.from_env threads it through
# ------------------------------------------------------------------


def test_config_from_env_uses_the_resolved_path(tmp_path) -> None:
    home = tmp_path / ".taipan"
    home.mkdir()
    cfg = Config.from_env({"HOME": str(tmp_path)})
    assert cfg.db_path == str(home / "verdryx.db")


def test_config_from_env_still_honours_the_explicit_var(tmp_path) -> None:
    cfg = Config.from_env({"VERDRYX_DB": "/tmp/x.db", "HOME": str(tmp_path)})
    assert cfg.db_path == "/tmp/x.db"
