"""Namespace package so `examples.bakeoff` is importable from tests/
(tests/test_bakeoff.py) without adding a runtime dependency or a package
install step. Nothing under examples/ ships in the verdryx wheel
(pyproject.toml's [tool.hatch.build.targets.wheel] packages only
"verdryx"): this is example/tooling code, not the library.
"""
