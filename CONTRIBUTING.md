# Contributing

Thanks for contributing to mlbisect. Keep changes small, deterministic, and
offline. Do not add network access to code or tests.

## Development Setup

```console
python -m pip install -e ".[dev,lab]"
python -m ruff check .
python -m mypy
python -m pytest
```

The full suite includes the six Regression Lab cases and normally runs on a
laptop. Add tests with behavior changes; do not update benchmark numbers by
hand.

## Add a Regression Lab Case

1. Create `lab/cases/L07_name/` with `case.yaml`, `mlbisect.yaml`, and `eval.py`.
2. Keep the case offline and fast. It should run in under ten seconds per
   evaluation on a laptop.
3. Generate data from the supplied `MLBISECT_SEED` and select versions from
   `MLBISECT_AXIS_<NAME>`.
4. Declare the ground truth in `case.yaml`, including `culprit_sets` or an
   `inconclusive`/`no_regression` expected verdict.
5. Add or extend an acceptance test that checks the true culprit set and its
   complement.
6. Run `python -m lab.run_benchmark --trials 20` and inspect the generated
   `lab/results/benchmark.md`.
7. Include a short note about where the method does not help.

Use generated data rather than committing large datasets. Keep case IDs and
axis names stable once published because benchmark reports refer to them.

## Code Changes

Public functions must be typed and deterministic by default. Keep statistics
and search logic free of file and subprocess I/O. Ask before adding a new
runtime dependency. Update `CHANGELOG.md` for milestone-level changes.

## Release Checklist

Release publication is an owner action. Before publishing, confirm the name,
version, PyPI metadata, benchmark output, README quickstart, and CI templates.
Run the full checks above, then publish and tag only after explicit approval.
