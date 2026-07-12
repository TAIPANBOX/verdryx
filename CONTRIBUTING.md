# Contributing to verdryx

## Development

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

```sh
pytest                # run tests
ruff check .           # lint
ruff format .          # format
```

Before every commit, this must be clean:

```sh
ruff check .
ruff format --check .
pytest
```

## Conventions

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`.
- One logical change per commit.
- `ruff check`, `ruff format --check`, and `pytest` must pass before a PR.
- Grade the outcome, not the call: cost-per-call numbers are meaningless
  without a quality denominator, so a new grader should report a quality
  signal, not just pass/fail.

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities privately.
