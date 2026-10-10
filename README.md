# mlbisect

Find the smallest set of changes that caused a noisy machine-learning
regression.

Your model can get four points worse in a week while data, dependencies, a
prompt, a checkpoint, and configuration all change. `git bisect` only walks
one ordered line of commits. Testing every mix costs `2^k` configurations, and
testing one axis at a time misses interactions.

`mlbisect` samples old/new mixes, stops evaluations when the statistical
verdict is clear, and reports the culprit set, confidence, run log, and
interaction effects.

## Quickstart

The repository includes an offline interaction case that runs on a laptop.

```console
# From a checkout of this repository:
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[lab]"
mlbisect run lab/cases/L03_interaction/mlbisect.yaml --max-runs 40 --out /tmp/mlbisect-report.md --json /tmp/mlbisect-report.json
```

The command runs the case's `eval.py` with environment variables for each
configuration. It reports `data + deps` as an interaction and writes a
resumable run log beside the case manifest. On Windows, use a normal
PowerShell line or replace `/tmp/...` with a path such as
`$env:TEMP\mlbisect-report.md`.

Typical terminal output:

```text
run 1/40: {} seed 0 metric=0.96
run 2/40: {} seed 1 metric=0.95
...
wrote Markdown report to /tmp/mlbisect-report.md
wrote JSON report to /tmp/mlbisect-report.json
```

For a package-only install, use `python -m pip install mlbisect` and create a
manifest with `mlbisect init`. The evaluation command must print a final line
such as `{"metric": 0.9123}` or write that JSON to `$MLBISECT_OUTPUT`.

## Commands

```console
mlbisect init                         # write a commented manifest template
mlbisect validate mlbisect.yaml       # validate and show run bounds
mlbisect noise mlbisect.yaml          # run an A/A noise check
mlbisect run mlbisect.yaml            # search and write Markdown/JSON reports
mlbisect run mlbisect.yaml --resume   # reuse completed runs
mlbisect run mlbisect.yaml --dry-run  # show the plan without executing
```

`run` writes `report.md`, `report.json`, and
`.mlbisect/runs.jsonl` by default. Use `--out`, `--json`, and `--max-runs` to
change the report paths or enforce a smaller global budget. Exit codes are 0
for a culprit, 1 for no reproducible regression, 2 for inconclusive or budget
exhaustion, and 3 for usage or manifest errors.

## Regression Lab

Regression Lab is an offline benchmark with six reproducible cases:

| Case | Ground truth | Purpose |
|---|---|---|
| L01 data skew | `{data}` | Single culprit |
| L02 preprocessing bump | `{deps}` | Dependency/preprocessing change |
| L03 interaction | `{data, deps}` | Non-additive interaction |
| L04 small effect | Inconclusive | Noise honesty |
| L05 decoy axes | `{axis_3}` | Larger axis set |
| L06 no regression | No regression | False-alarm control |

Run the complete benchmark and regenerate the checked-in report with one
command:

```console
python -m pip install -e ".[dev,lab]"
python -m lab.run_benchmark
```

The current 20-trial results are in
[`lab/results/benchmark.md`](lab/results/benchmark.md). The headline result is
that one-at-a-time testing misses L03, while `mlbisect` identifies it with
fewer runs than exhaustive search. L04 is intentionally difficult and shows
where fixed-threshold methods produce false accusations.

## How It Fits

| Tool | Strength | Relationship to mlbisect |
|---|---|---|
| MLflow or W&B | Records and compares experiment runs | Complementary: trackers show what ran; mlbisect chooses which mixes to run |
| Optuna or sweeps | Searches for good hyperparameters | Different goal: mlbisect searches for the cause of a regression |
| `git bisect` | Finds a bad commit on one ordered axis | Generalized by mlbisect to multiple axes and noisy metrics |
| Monitoring and drift tools | Detects that performance changed | Upstream signal; mlbisect investigates which changes caused it |

## Documentation

- [Manifest reference](docs/manifest.md)
- [Design and algorithm](docs/design.md)
- [Contributing and adding lab cases](CONTRIBUTING.md)
- [GitHub Action](ci/github-action/action.yml)
- [GitLab CI template](ci/gitlab/mlbisect.gitlab-ci.yml)

## Limitations

- The evaluation command must be repeatable enough to sample.
- Axes must be independently swappable by the user's pipeline.
- Search cost grows with the number of axes and evaluation time.
- The v0.1 local executor does not orchestrate training jobs or clusters.
- Paired seeds, SPRT, Docker execution, git-ref axes, and tracker integrations
  are planned for later versions.
- Results are evidence, not an automatic fix for the regression.

## Development

```console
python -m pip install -e ".[dev,lab]"
python -m ruff check .
python -m mypy
python -m pytest
```

mlbisect is licensed under the Apache License, Version 2.0.
