# mlbisect: Implementation Plan

Status: draft v1 (2026-10-10)
Working name: `mlbisect` (check PyPI and GitHub for name clashes before first release)
License: Apache 2.0
Language: Python 3.10+

This document is the single reference for the project. It is written for a human owner and for an AI coding agent that will implement it milestone by milestone. Section 15 holds the agent rules.

---

## 1. Summary

mlbisect finds the smallest set of changes that caused an ML regression, even when evaluations are noisy and compute is limited.

When a model gets worse after a week in which the data, a library, a prompt and the model version all changed, `git bisect` cannot help: there is more than one axis of change, and the test is not a clean pass or fail. mlbisect takes a manifest of all changed axes, evaluates smart mixes of old and new versions, stops each evaluation as soon as the answer is statistically clear, and reports the culprit set, its confidence, the runs used, and any interaction between changes.

It ships together with **Regression Lab**, a public benchmark of regressions we inject on purpose, so we can publish real numbers (success rate and runs used) against naive methods.

Release strategy: open source first. A hosted product is only considered later if real usage shows demand (see section 13).

---

## 2. Problem and context

Typical situation:

- Accuracy dropped 4% in a week. Nothing crashed and no test failed.
- Merged that week: a new data snapshot, a library upgrade (tokenizer or framework), a new prompt template, a new base checkpoint, new hyperparameters.
- Which change caused it?

Why existing tools fall short:

1. **Many axes change at once.** `git bisect` walks one line of commits. A model depends on code, data, dependencies, weights and prompts that move independently.
2. **Evaluations are noisy.** Random seeds, sampling and small test sets make scores wobble. The same model may score 90.4, 91.8 and 91.2 on three reruns, so a 1 point drop may be nothing.
3. **Runs are expensive.** Each check may be a training job or a GPU eval. Testing every combination does not scale (2^k configurations for k axes).
4. **Changes interact.** Two harmless changes can be harmful together.

---

## 3. Core idea

Treat the regression as a search problem over *configurations*, where a configuration is "some axes at their new value, the rest at their old value".

1. Declare every axis of change in a manifest (old value, new value).
2. Check that the problem is real: baseline (all old) must be healthy and candidate (all new) must be regressed.
3. Use delta debugging (ddmin) to shrink the set of suspect axes: test subsets, keep the part that still reproduces the drop, repeat until the set is minimal.
4. For each configuration tested, sample evals adaptively: few runs when the verdict is obvious, more when it is ambiguous, within a global run budget.
5. Report the minimal culprit set, confidence, runs saved versus naive methods, and interaction effects.

### Worked example (interaction)

| Configuration | Accuracy | Verdict |
|---|---|---|
| All old (baseline) | 91.2 | healthy |
| All new | 87.2 | regressed |
| New data only | 90.9 | healthy |
| New library only | 91.0 | healthy |
| New data + new library | 87.5 | regressed |

Neither change hurts alone, together they cost about 4 points. A one-at-a-time test blames nobody. mlbisect reports `culprit = {data, library}` flagged as an interaction.

---

## 4. Goals and non-goals

Goals (v0.1):

- A Python CLI that runs from a local machine or a CI job.
- A YAML manifest that is easy to write for an existing pipeline.
- Noise-aware search with a run budget and a reproducible run log.
- A Markdown and JSON report.
- Regression Lab with at least 6 cases and a scoring script.
- GitHub Action and GitLab CI templates.

Non-goals (v0.1):

- No web UI or dashboard.
- No hosted service, accounts or billing.
- No automatic training orchestration (the user's command does the work).
- No cluster scheduling, no Docker or Kubernetes executor (planned for v0.2).
- No automatic fixing of the regression.

### 4.1 Positioning: how it relates to other tools

mlbisect does not replace existing MLOps tools. It answers a different question: "which change caused this drop?"

| Tool type | What it does | What it does not do | Relationship |
|---|---|---|---|
| Experiment trackers (MLflow, Weights & Biases) | Record runs: params, metrics, artifacts; compare them in a UI; model registry | Choose which runs to make, judge noise, find the culprit or interactions | Complement. v0.2: log every mlbisect run to the tracker, and optionally read baseline and candidate metrics from existing tracked runs to save reruns |
| Hyperparameter search (Optuna, sweeps) | Search for the best settings | Explain why a model got worse after a set of changes | Different goal: they search for the best, mlbisect searches for the cause |
| `git bisect` | Finds the first bad commit on one axis with a deterministic pass or fail test | Multiple axes, noisy metrics, run budgets, interactions | mlbisect generalizes the idea to ML |
| Drift and monitoring tools | Detect that data or performance changed | Say which of several changes is responsible | Upstream: monitoring raises the alarm, mlbisect investigates it |
| CI test frameworks | Pass or fail checks | Statistical verdicts on noisy scores | mlbisect runs inside CI and reports a verdict with confidence |

Honest caveat: a team that already ran every combination of old and new changes (and enough seeds) could read the culprit from an MLflow table. In practice almost nobody does, because it can take up to 2^k configurations times several repeats. The value of mlbisect is choosing the few runs that matter and knowing when to stop.

Messaging line for the README and launch posts: "Experiment trackers show you what each run scored. mlbisect decides which runs are worth making."

---

## 5. Key concepts

| Term | Meaning |
|---|---|
| Axis | One dimension of change, for example `data`, `deps`, `prompt`, `checkpoint`, `config`. Has an `old` and a `new` value. |
| Configuration | A subset of axes set to `new`; all others stay `old`. Identified by a frozen set of axis names. `{}` is the baseline, the full set is the candidate. |
| Run | One execution of the user's eval command for a configuration and a seed. Returns one metric value. |
| Metric | A number with a direction (`higher_is_better` or `lower_is_better`). |
| Min effect (`m`) | The smallest drop, in metric units, that counts as a regression. |
| Drop (`d`) | Baseline mean minus configuration mean (sign flipped for lower-is-better). |
| Verdict | `BAD`, `OK` or `UNDECIDED` for a configuration. |
| Culprit set | A minimal set of axes such that applying only those new values reproduces the regression. |
| Interaction | A culprit set with 2 or more axes (each axis alone is healthy). |

---

## 6. How it works

### 6.1 Flow

```
load manifest
  -> validate
  -> noise check (A/A): run baseline twice, estimate noise sd
  -> baseline sampling (n0 runs, seeds 0..n0-1)
  -> sanity: candidate (all new) must be BAD, else stop with "no reproducible regression"
  -> ddmin over axes, each test(config) uses adaptive sampling + cache
  -> verify: test(M) is BAD and test(all minus M) is OK
        if all minus M is still BAD: more culprits, run ddmin again on the remainder
  -> interaction analysis on each culprit set
  -> write report (md + json) and run log
```

### 6.2 Decision rule per configuration (v0.1)

Let `m = min_effect` and `t = m / 2` (the midpoint between "no regression" and "regression").

1. Run at least `min_runs_per_config` (default 3) evals for the configuration.
2. Estimate drop `d` against the baseline samples and compute a `(1 - alpha)` bootstrap confidence interval `[lo, hi]` for `d` (default 95%, 2000 resamples, fixed RNG seed for reproducibility).
3. If `lo >= t`: verdict `BAD`. If `hi < t`: verdict `OK`. Otherwise `UNDECIDED`.
4. If `UNDECIDED`, run another batch (`batch_size`, default 2) and repeat.
5. At `max_runs_per_config` (default 10) or when the global budget is exhausted, decide by comparing the point estimate to `t` and mark the verdict `low_confidence = true`.

Edge cases: zero variance (deterministic eval) must work without dividing by zero; fewer than 3 runs never decides; failed runs are retried once, then counted as errors and excluded.

Planned later (v0.2): SPRT as an alternative rule, paired seeds to reduce variance, and a baseline top-up when the baseline is the noisiest part.

### 6.3 Search (ddmin)

Use the classic ddmin algorithm over the list of axis names, with `test(S) = True` when configuration `S` is `BAD`.

- Precondition: `test({})` is OK by construction, `test(all)` is BAD (checked in the flow above).
- Memoize `test` by `frozenset` so no configuration is evaluated twice.
- Return a 1-minimal set: removing any single axis from it makes the configuration not BAD.
- Order of tests should be deterministic (sort axes by manifest order).
- Multiple culprits: after finding `M`, evaluate `all minus M`. If BAD, run ddmin on that remainder and repeat (cap at 3 rounds).

### 6.4 Interaction analysis

For each culprit set `M` with `|M| >= 2`:

- Evaluate each single axis in `M` (reuse cache) and the joint set.
- Report single effects `d_i`, joint effect `d_M`, and the non-additive part `d_M - sum(d_i)`.
- Label as `interaction`. For `|M| = 1`, label `single_cause`.

### 6.5 Budget

- `max_runs` is a hard global cap on eval runs.
- When the budget is nearly exhausted, stop sampling undecided configurations and return a best-effort answer with a ranked list of suspects and a clear `incomplete` flag.
- Always print runs used, and runs a naive method would need (one-at-a-time with `max_runs_per_config` per axis, and exhaustive 2^k).

---

## 7. Interfaces

### 7.1 Manifest (`mlbisect.yaml`)

```yaml
version: 1
name: sentiment-regression

metric:
  name: accuracy
  direction: higher_is_better     # or lower_is_better
  min_effect: 0.02                # absolute drop that counts as a regression

run:
  command: "python eval.py"       # executed once per run
  timeout_s: 600
  workdir: "."

axes:
  - name: data
    old: data/v1.parquet
    new: data/v2.parquet
  - name: deps
    old: lock/requirements.v1.txt
    new: lock/requirements.v2.txt
  - name: prompt
    old: prompts/v1.txt
    new: prompts/v2.txt

budget:
  max_runs: 60
  baseline_runs: 5
  min_runs_per_config: 3
  max_runs_per_config: 10
  batch_size: 2
  confidence: 0.95
```

Validation rules: unique axis names matching `[a-z][a-z0-9_]*`, `old != new`, `min_effect > 0`, at least 2 axes, `min_runs_per_config <= max_runs_per_config`, `max_runs >= baseline_runs + 2 * min_runs_per_config`.

### 7.1.1 Authoring the manifest, seeds and auto-discovery

**Who writes the manifest.** In v0.1 the user writes it. mlbisect cannot know which data file is the new snapshot or which prompt file is the new one. To keep this cheap:

- `mlbisect init` writes a commented template with the common axis names (`data`, `code`, `deps`, `checkpoint`, `prompt`, `config`) and explains the run contract inline.
- `mlbisect validate` checks that referenced files exist, that the rules in 7.1 hold, and prints the worst-case run count.

**What the user's eval command must do.** The integration cost is on the command side:

1. Read `MLBISECT_AXIS_<NAME>` to choose the old or new version of each axis.
2. Use `MLBISECT_SEED` to seed all randomness (weight init, shuffling, dropout, sampling).
3. Report the metric (see 7.2).

Docs must include three short examples: a Python script reading the environment variables, a Makefile target, and a shell wrapper that swaps files before calling an existing script.

**What a seed is (for the docs).** A seed is a number that starts a random number generator. The same seed gives the same result; a different seed gives a slightly different one. The spread between seeds is the noise mlbisect measures: it runs each configuration with several seeds and uses the spread to decide whether a drop is real. If the user's command ignores `MLBISECT_SEED`, runs may still vary naturally, but mlbisect cannot control or reproduce them, and `mlbisect noise` should warn about it when two identical runs give identical or wildly different scores. Paired seeds (the same seeds for baseline and candidate, to cancel shared noise) are planned for v0.2.

**Axes are not fixed.** The axes are whatever changed between the last good state and the current bad state, so they differ per project and per regression. The template names are only suggestions. Axes can be split for precision (for example `tokenizer_lib` and `numpy` instead of one `deps` axis); more axes give a sharper answer but need more runs. Guidance for the docs: start with 3 to 6 axes, and `validate` warns when there are more than 8.

**Auto-discovery (planned, not in v0.1).** Principle: tooling may propose axes, but never silently decides them. The user reviews and edits the generated manifest.

| Axis type | Possible automation | Target |
|---|---|---|
| Code | Axis values as git refs (`old: git:<ref>`, `new: git:<ref>`); the executor checks out a worktree per ref | v0.2 |
| Dependencies | Diff two lockfiles (requirements, poetry.lock, uv.lock) and suggest one axis per changed package or per group of packages | v0.2 |
| Prompts and config | File paths, or git refs in the same way as code | v0.2 |
| Checkpoints | Read model versions from an experiment tracker or registry (MLflow integration) | v0.2 or later |
| Data | Manual (file path or version id); DVC or lakeFS references may be added later | Later |

Proposed command: `mlbisect init --from-git <old>..<new>` lists the changed files between two refs, groups them into candidate axes by path patterns (data directories, prompt files, lockfiles, source code), and writes a draft manifest for the user to review.

### 7.2 Run contract (what the user's command must follow)

mlbisect sets environment variables for each run:

| Variable | Meaning |
|---|---|
| `MLBISECT_SEED` | Integer seed for this run |
| `MLBISECT_AXIS_<NAME>` | Value to use for each axis (old or new), name upper-cased |
| `MLBISECT_CONFIG` | Comma-separated list of axes currently at `new` (empty for baseline) |
| `MLBISECT_OUTPUT` | Optional path where the command may write the result JSON |

The command reports its metric in one of two ways:

1. Print a final stdout line: `{"metric": 0.9123}`
2. Or write `{"metric": 0.9123}` to the file at `$MLBISECT_OUTPUT`.

Exit code non-zero means a failed run.

### 7.3 CLI

```
mlbisect init                         # write a template mlbisect.yaml
mlbisect validate mlbisect.yaml       # check manifest, print a plan and worst-case run count
mlbisect noise mlbisect.yaml          # A/A test: estimate noise, warn if noise ~ min_effect
mlbisect run mlbisect.yaml [--max-runs N] [--out report.md] [--json report.json] [--resume] [--dry-run]
# Planned after run-log metadata is expanded: re-render a report from a run log
```

Exit codes: 0 culprit found, 1 no reproducible regression, 2 inconclusive or budget exhausted, 3 usage or manifest error.

### 7.4 Executor interface

```python
class Executor(Protocol):
    def run(self, config: Configuration, seed: int) -> RunResult: ...

@dataclass(frozen=True)
class RunResult:
    config: frozenset[str]
    seed: int
    metric: float | None
    ok: bool
    duration_s: float
    error: str | None = None
```

Implementations: `LocalExecutor` (subprocess, v0.1), `FakeExecutor` (simulated metric with configurable per-axis effects, interactions and noise, used in unit tests and by Regression Lab), `DockerExecutor` (v0.2).

### 7.5 Run log

Append-only `.mlbisect/runs.jsonl`, one line per run: config, seed, metric, ok, duration, timestamp. `--resume` reloads it so completed runs are not repeated and the cache survives interruptions.

### 7.6 Report (Markdown, plus the same data as JSON)

```
# mlbisect report: sentiment-regression

Result: culprit found (interaction), confidence high
Culprit set: data + deps

Baseline 91.2 (n=5, sd 0.6) | Candidate 87.2 (n=4)

| Configuration | Runs | Mean | Drop | Verdict |
|---|---|---|---|---|
| {} | 5 | 91.2 | 0.0 | OK |
| {data} | 3 | 90.9 | 0.3 | OK |
| {deps} | 3 | 91.0 | 0.2 | OK |
| {data, deps} | 4 | 87.5 | 3.7 | BAD |

Interaction: individual drops 0.3 + 0.2, joint drop 3.7 (non-additive part 3.2)

Runs used: 19 | one-at-a-time baseline: 33 | exhaustive: 80
Warnings: none
```

---

## 8. Repository layout and tooling

```
mlbisect/
  pyproject.toml
  README.md
  LICENSE
  CHANGELOG.md
  docs/
    design.md            # condensed from this plan
    manifest.md          # manifest reference
  src/mlbisect/
    __init__.py
    cli.py               # typer app
    manifest.py          # pydantic models, loading, validation
    config.py            # Configuration helpers, hashing
    executors/
      base.py            # Executor protocol, RunResult
      local.py           # subprocess executor
      fake.py            # simulated executor for tests and the lab
    store.py             # runs.jsonl persistence and cache
    stats.py             # drop estimate, bootstrap CI, decision rule
    sampler.py           # adaptive sampling per config, budget accounting
    search.py            # ddmin, multi-culprit loop, interaction analysis
    report.py            # markdown and json rendering
  lab/                   # Regression Lab (section 9)
    datagen.py
    cases/<case_id>/{case.yaml, mlbisect.yaml, eval.py}
    baselines.py         # naive methods
    run_benchmark.py
    score.py
    results/benchmark.md
  ci/
    github-action/action.yml
    gitlab/mlbisect.gitlab-ci.yml
  examples/
  tests/
```

Tooling: `uv` or `pip` with `pyproject.toml`, `typer`, `pydantic` v2, `pyyaml`, `numpy`, `rich` for output. Dev: `pytest`, `ruff`, `mypy`. Keep runtime dependencies minimal; scikit-learn only for the lab (as an optional extra `lab`). Avoid `scipy` unless needed.

---

## 9. Regression Lab (benchmark)

Purpose: give the project real, reproducible evidence, and a place for the community to contribute cases.

Principles:

- Everything runs offline on a laptop CPU in seconds per eval. No downloads.
- Data and "versions" are generated by `datagen.py` from a seed. A version is selected by environment variables, for example `MLBISECT_AXIS_DATA=v2`.
- Each case has a ground truth in `case.yaml`.
- Include cases where the tool should say "inconclusive" or "no regression". These protect the credibility of the results.

`case.yaml`:

```yaml
id: L03_interaction
description: New data adds a feature on a new scale; new preprocessing drops scaling. Each alone is fine.
axes: [data, deps]
truth:
  culprit_sets: [[data, deps]]
  expected_verdict: culprit_found   # culprit_found | inconclusive | no_regression
  kind: interaction
```

Initial cases:

| Id | What changes | Ground truth | What it tests |
|---|---|---|---|
| L01_data_skew | Data v2 shifts class balance and noise in one segment | {data} | Single culprit |
| L02_preproc_bump | "Library" v2 changes default preprocessing | {deps} | Single culprit |
| L03_interaction | Data v2 adds a feature at a new scale, deps v2 drops scaling | {data, deps} | Interaction |
| L04_small_effect | Real drop 0.5 pts, eval noise sd 1.5 pts | inconclusive | Honesty under noise |
| L05_decoy_axes | 6 axes, only one harmful | {axis_3} | Efficiency at larger k |
| L06_no_regression | All changes harmless | no_regression | No false alarms |

Later: L07_multi_culprit (two independent culprits), L08_label_leak, L09_flaky_eval, L10_prompt_change (simulated LLM eval), plus community submissions.

Methods compared (all implemented in `lab/baselines.py`):

1. `mlbisect`
2. `one_at_a_time`: test each axis alone with a fixed 10 runs, blame axes with a significant drop. Cannot find interactions.
3. `exhaustive`: all 2^k subsets with a fixed 10 runs each.
4. `ordered_bisect`: treat axes as an ordered timeline and bisect with a fixed 5 runs per step (the `git bisect` analogue).

Scoring (`score.py`): run each method on each case for 20 trials with different master seeds. Report per method and case: exact culprit match rate, false accusation rate, inconclusive rate, median and p90 eval runs. Output `lab/results/benchmark.md` with a summary table and a short honest "where it did not help" section.

---

## 10. Roadmap and milestones

Assumption: about 8 hours per week alongside a full-time job. Total about 12 weeks. Calendar weeks are indicative; milestones are the contract.

| Phase | Weeks | Milestones |
|---|---|---|
| 0 Foundations | 0 | M0 |
| 1 Benchmark first | 1 to 3 | M1, M2 |
| 2 Engine | 3 to 6 | M3, M4, M5, M6 |
| 3 Make it usable | 6 to 8 | M7 |
| 4 Launch | 8 | M8 |
| 5 Learn and iterate | 9 to 12 | M9 |

### M0: Scaffold (week 0)

Tasks:
- Create repo, `pyproject.toml`, Apache 2.0 license, README stub, `CHANGELOG.md`.
- Configure `ruff`, `mypy`, `pytest`, and a CI workflow (lint, type check, tests on Python 3.10 to 3.12).
- Add `mlbisect --version`.

Acceptance: CI is green on an empty test suite; `pip install -e .` works; `mlbisect --version` prints a version.

### M1: Manifest, executors, run store (week 1)

Tasks:
- `manifest.py` with pydantic models and all validation rules from 7.1; helpful error messages.
- `config.py` with `Configuration` (frozenset of axis names) and a stable string id.
- `executors/base.py`, `local.py` (subprocess, env vars from 7.2, timeout, metric parsing from stdout or `MLBISECT_OUTPUT`), `fake.py`.
- `store.py` with append-only JSONL and a cache lookup by `(config, seed)`.
- CLI: `init` (commented template as described in 7.1.1), `validate`.

Acceptance: unit tests for validation errors and metric parsing; `LocalExecutor` runs a sample script with the right env vars; `FakeExecutor` is deterministic for a given seed; `validate` prints worst-case run counts.

### M2: Regression Lab v0 and naive baselines (weeks 1 to 3)

Tasks:
- `datagen.py` for a tabular task (synthetic classification, scikit-learn model, metric on a held-out set). Version switches by env var.
- Cases L01, L02, L03, L04, L05, L06 with `case.yaml`, `mlbisect.yaml`, `eval.py`.
- `baselines.py` with `one_at_a_time`, `exhaustive`, `ordered_bisect` using the executor interface.
- `run_benchmark.py` and `score.py` (mlbisect not wired yet).

Acceptance: every case runs end to end in under 10 seconds per eval; ground truth is verified by a test that evaluates the true culprit set and confirms it is BAD and its complement is OK (with enough runs); baselines produce results, and one-at-a-time visibly fails L03.

### M3: Statistics and adaptive sampler (weeks 3 to 4)

Tasks:
- `stats.py`: drop estimate with direction handling, deterministic bootstrap CI, decision rule from 6.2.
- `sampler.py`: sample until decisive, respect per-config and global budgets, reuse cached runs, return a `ConfigEstimate` (n, mean, drop, ci, verdict, low_confidence).
- CLI: `noise` (A/A test).

Acceptance: property tests with `FakeExecutor`: for a known effect and noise level, the false OK and false BAD rates are below 5% at the default confidence over 500 simulated trials; zero variance works; budget is never exceeded.

### M4: Search (weeks 4 to 5)

Tasks:
- `search.py`: ddmin with memoization, deterministic ordering, 1-minimal result.
- Pre-checks: candidate must be BAD, baseline must be OK; return `no_reproducible_regression` otherwise.
- Verification step and the multi-culprit loop.
- Budget exhaustion path returns a ranked, `incomplete` result.

Acceptance: against `FakeExecutor` scenarios (single culprit, interaction, two culprits, no regression, decoys with k up to 8) the correct set is found in at least 95% of simulated trials; number of evaluated configurations is below 2^k for k >= 5.

### M5: Report and `run` command (weeks 5 to 6)

Tasks:
- `report.py` for Markdown and JSON as in 7.6, including the naive-run comparison.
- CLI `run` with `--max-runs`, `--out`, `--json`, `--resume`, `--dry-run`, and the exit codes from 7.3.
- Live progress output with `rich` (plain text when not a TTY).

Acceptance: snapshot tests for the report; `--resume` after an interrupted run repeats no completed run; `--dry-run` lists planned configurations without executing.

### M6: Interaction analysis and Lab integration (week 6)

Tasks:
- Interaction analysis from 6.4 in `search.py` and `report.py`.
- Wire `mlbisect` into `run_benchmark.py`; run all cases for 20 trials; generate `lab/results/benchmark.md`.

Acceptance: L03 is reported as an interaction with a non-additive effect; L04 is reported as inconclusive; L06 as no regression; the benchmark table is generated by one command.

### M7: Usability and release v0.1 (weeks 6 to 8)

Tasks:
- README: pain statement, 2-minute quickstart using a lab case, demo GIF or terminal recording, benchmark summary, a short "how it relates to MLflow, Optuna and git bisect" section (from 4.1), manifest reference link, limitations section.
- `docs/manifest.md` and `docs/design.md`.
- `ci/github-action/action.yml` and `ci/gitlab/mlbisect.gitlab-ci.yml` that run `mlbisect run` and upload the report as an artifact.
- Issue templates: bug report and "bring your regression".
- `CONTRIBUTING.md` explaining how to add a Regression Lab case.
- Publish to PyPI and tag `v0.1.0`.

Acceptance: a fresh virtualenv can `pip install mlbisect` and reproduce the quickstart in under 5 minutes; CI templates run on a sample repository.

### M8: Launch (week 8, human tasks)

- LinkedIn post 1: lead with the pain ("model 4% worse, six things changed"), show the benchmark table (including where it did not help), invite readers to run it and send unexplained regressions.
- Pin the "bring your regression" issue.
- Share in relevant communities (MLOps and ML engineering groups) as an experiment, not an announcement.

### M9: Learn and iterate (weeks 9 to 12)

- Convert good submissions into Regression Lab cases with contributor credit.
- Post 2 (the noise problem) and post 3 (the interaction finding).
- v0.2 candidates, in priority order: Docker executor, git-ref axes (checkout a worktree per ref), `init --from-git` and lockfile-diff axis suggestions (7.1.1), MLflow integration (log runs, reuse existing baseline and candidate runs), paired seeds, SPRT rule, more lab cases, an LLM eval case.
- Run the week 12 checkpoint (section 13).

---

## 11. Testing strategy

- **Unit tests:** manifest validation, config ids, metric parsing, store and cache, stats edge cases, ddmin on pure predicates.
- **Simulation tests:** `FakeExecutor` with known effects, interactions and noise levels; run hundreds of trials with fixed seeds to check error rates and run counts. Tests must be deterministic and fast (under 60 seconds total).
- **Integration tests:** `LocalExecutor` against tiny scripts; CLI tests with `typer.testing.CliRunner`.
- **Benchmark as acceptance test:** the Lab must show the claimed results; a regression in success rate fails CI on a nightly job (not on every PR).
- No network access in any test.

---

## 12. Release, open source and launch notes

- Apache 2.0, public repo from the first commit of M7 at the latest (earlier is fine).
- Semantic versioning, changelog entries per milestone.
- README must state limitations plainly: assumes the user's eval is repeatable enough to sample, assumes axes can be swapped independently, search cost grows with the number of axes and the run time of the eval.
- Possible open-core path later (hosted history of regressions across a team, run scheduling, integrations with experiment trackers). Do not build any of it before the signals in section 13 appear.

---

## 13. Success metrics and checkpoints

| When | Question | Good sign |
|---|---|---|
| End of M2 (about week 3) | Does the benchmark show naive methods struggling? | One-at-a-time misses the interaction; exhaustive needs many more runs |
| End of M6 (about week 6) | Does mlbisect beat the baselines on the lab? | Correct culprit in at least 90% of trials with clearly fewer runs than exhaustive and one-at-a-time |
| Week 12 | Is anyone else using it? | People outside the owner's network file issues, submit cases, or run it in CI more than once |

Signals that justify exploring a paid layer: repeated use in CI by the same teams, independent requests for shared history or hosted runners, someone asking how to pay. Stars alone are not a signal.

If M6 shows no advantage over naive bisect, publish that finding honestly and rethink the story before launch.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep (dashboards, more executors) | Hold v0.1 to CLI, manifest, search, report. Everything else goes to v0.2. |
| Benchmark looks rigged | Include L04 and L06 (honest failures and controls); publish code and seeds; invite new cases. |
| Real pipelines are not cleanly swappable by axis | Document the run contract with examples; provide a Docker executor and git-ref axes in v0.2. |
| Search cost too high for real evals | Always show the worst-case run count in `validate` and the budget in `run`; support `--resume`. |
| Limited time (full-time job) | If a week is lost, move interaction polish or CI templates to v0.2 rather than delaying the launch. |
| Name clash | Check PyPI and GitHub before M7; rename early if needed. |

---

## 15. Instructions for the AI coding agent

Read this whole document first, then implement one milestone at a time in order (M0 to M7). M8 is human work. M9 only after M8.

Rules:

1. **One milestone per session or branch.** Finish its acceptance criteria before moving on. Do not start the next milestone's tasks early.
2. **Stay in scope.** Implement only what the milestone lists. Anything else goes into a `docs/backlog.md` file as a one-line note.
3. **Tests first for algorithms.** For `stats.py`, `sampler.py` and `search.py`, write the simulation or property tests before the implementation.
4. **Deterministic by default.** All randomness takes an explicit seed or RNG. Same inputs must give the same outputs.
5. **No network access** in code or tests. The lab generates its own data.
6. **Typed, small, pure.** Type-annotate all public functions; keep search and stats logic free of I/O so it can be tested with `FakeExecutor`.
7. **Do not invent results.** Benchmark numbers in the README and docs must come from running `lab/run_benchmark.py`. If a result is bad, report it as it is.
8. **Errors for humans.** Manifest and runtime errors should say what is wrong and how to fix it.
9. **Keep dependencies minimal.** Ask before adding a new runtime dependency; record the reason in the PR or commit message.
10. **Small commits** with clear messages. Update `CHANGELOG.md` at the end of each milestone.

Definition of done for every milestone:

- [ ] All tasks implemented as listed
- [ ] Acceptance criteria met and demonstrated by tests or a command
- [ ] `ruff`, `mypy` and `pytest` pass
- [ ] Docs and `CHANGELOG.md` updated
- [ ] Open questions or deviations written down at the end of the milestone summary

Suggested first prompt to the agent:

> Read `mlbisect-plan.md`. Implement milestone M0, then stop and summarize what you did, what you decided on your own, and any open questions. Follow the rules in section 15.

---

## 16. Open decisions for the owner

1. Final project name and PyPI availability.
2. Whether Regression Lab stays inside the repo (`lab/`) or becomes its own repo at launch (default: inside, split later if it grows).
3. Which tasks to add after the tabular one (a small text classifier, then a simulated LLM eval).
4. Whether to open the repo publicly before M7 to collect early feedback (default: public from M1, with an "early development" banner).
5. How to publish the launch results (LinkedIn post only, or also a short blog post in the repo `docs/`).
