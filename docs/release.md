# Release Checklist

The repository is currently kept at development version `0.0.1` by agreement.
This checklist records the remaining owner action for the v0.1 release:

- Confirm the final `mlbisect` PyPI name and metadata.
- Decide the release version and update `pyproject.toml` and `CHANGELOG.md`.
- Run `python -m lab.run_benchmark` and review the generated report.
- Verify a clean virtualenv can install the package and follow the README.
- Verify the GitHub Action and GitLab template in a sample repository.
- Build and inspect the source and wheel distributions.
- Publish to PyPI.
- Create the `v0.1.0` tag.

Publishing and tagging are deliberately not automated by this repository
change.
