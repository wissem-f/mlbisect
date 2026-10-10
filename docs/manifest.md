# Manifest Reference

An `mlbisect.yaml` declares the old and new value for every independent axis,

## Complete Example

```yaml
version: 1
name: sentiment-regression

metric:
  name: accuracy
  direction: higher_is_better  # or lower_is_better
  min_effect: 0.02             # absolute drop that counts as a regression

run:
  command: "python eval.py"
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

## Fields

`version` must be `1`. `name` identifies the regression in reports.

`metric.name` is displayed in reports. `higher_is_better` means a lower value
is a drop; `lower_is_better` reverses that comparison. `min_effect` is the
smallest absolute effect worth calling a regression. The decision rule uses
half of this value as its midpoint threshold.

`run.command` is executed once per evaluation. `timeout_s` stops a command
that runs too long, and `workdir` is resolved relative to the manifest file.

Each axis name must match `[a-z][a-z0-9_]*`, be unique, and have different
`old` and `new` values. A manifest needs at least two axes. Values are strings;
they may be paths, version IDs, or registry references.

The budget fields mean:

| Field | Meaning | Default |
|---|---|---:|
| `max_runs` | Hard global cap across the search | 60 |
| `baseline_runs` | Baseline samples collected before searching | 5 |
| `min_runs_per_config` | Minimum samples before adaptive decisions | 3 |
| `max_runs_per_config` | Per-configuration sample ceiling | 10 |
| `batch_size` | Additional samples for an undecided configuration | 2 |
| `confidence` | Bootstrap confidence level | 0.95 |

`max_runs` must be at least `baseline_runs + 2 * min_runs_per_config`, and
`min_runs_per_config` cannot exceed `max_runs_per_config`.

## Evaluation Contract

For every run, mlbisect sets:

| Variable | Meaning |
|---|---|
| `MLBISECT_SEED` | Integer seed for all randomness in the evaluation |
| `MLBISECT_AXIS_<NAME>` | The old or new value selected for an axis |
| `MLBISECT_CONFIG` | Comma-separated new axes, empty for baseline |
| `MLBISECT_OUTPUT` | Temporary path where the command may write its result |

The command must exit zero and report one metric in one of these forms:

```text
{"metric": 0.9123}
```

as the final non-empty stdout line, or the same JSON object written to
`$MLBISECT_OUTPUT`. A non-zero exit code is a failed run. Failed runs are
retried once and then excluded from estimates.

Example Python evaluator:

```python
import json
import os

seed = int(os.environ["MLBISECT_SEED"])
data_version = os.environ["MLBISECT_AXIS_DATA"]
metric = evaluate(data_version, seed)
print(json.dumps({"metric": metric}))
```

## Authoring Guidance

Start with three to six axes. Split a broad dependency change when you need to
distinguish packages or groups. Keep axes independently swappable and make the
seed control every source of randomness. Use `mlbisect validate` before a run;
