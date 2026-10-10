# Changelog

## [0.1.0] - 2026-10-10

### Added

- Initial Python package scaffold.
- `mlbisect --version` command.
- Ruff, mypy, pytest, and GitHub Actions configuration.

- YAML manifest models and validation.
- Configuration identifiers, local and fake executors, and a JSONL run store.
- `mlbisect init` and `mlbisect validate` commands.
- Regression Lab synthetic classification generator and cases L01 through L06.
- Fixed-budget one-at-a-time, exhaustive, and ordered-bisect baselines.
- Benchmark runner and Markdown scorer.
- Deterministic bootstrap confidence intervals and adaptive budget-aware sampling.
- `mlbisect noise` A/A noise check.
- Memoized ddmin search with prechecks, verification, multi-culprit rounds, and
  incomplete budget-exhaustion results.
- Markdown and JSON reports with naive-run comparisons.
- `mlbisect run` with run-budget overrides, resume, dry-run, exit codes, and
  TTY-aware progress output.
- Interaction effect decomposition with non-additive effects in search reports.
- Regression Lab benchmark integration for `mlbisect` and 20-trial output.
- v0.1 usability documentation, CI templates, issue forms, and contribution
  guidance.
- Resume safety fixes for historical budgets, failed retries, and manifest
  fingerprints.
- Reports now identify the metric and distinguish tentative from confirmed
  culprit sets.
