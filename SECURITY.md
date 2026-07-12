# Security Policy

verdryx grades agent outputs and reports quality drift, which feeds into
budget and rollout decisions elsewhere in the stack, so a spoofable grade is
itself a security-relevant bug. This document covers how to report a
vulnerability.

## Reporting a vulnerability

Please report security issues privately, not in public issues or PRs:

- Open a **GitHub private security advisory**:
  <https://github.com/TAIPANBOX/verdryx/security/advisories/new>

Include the affected version/commit, a description, and a minimal reproduction.
We aim to acknowledge within a few days and to fix high-severity issues before
any public disclosure. There is no bug-bounty program; we credit reporters in
the advisory unless you prefer otherwise.

## Supported versions

verdryx is pre-1.0; only `main` is supported. Fixes land on `main` and are
not backported.

## Verifying a build

Every change must pass the full gate before merge: `ruff check .`,
`ruff format --check .`, and `pytest`. See [CONTRIBUTING.md](CONTRIBUTING.md).
