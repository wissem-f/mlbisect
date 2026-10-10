"""Command-line interface for mlbisect."""

import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import typer

from mlbisect import __version__
from mlbisect.config import Configuration
from mlbisect.executors.local import LocalExecutor
from mlbisect.manifest import (
    Manifest,
    ManifestError,
    load_manifest,
    manifest_fingerprint,
)
from mlbisect.report import write_reports
from mlbisect.sampler import AdaptiveSampler
from mlbisect.search import SearchResult, find_culprits
from mlbisect.store import RunStore, StoreError

app = typer.Typer(add_completion=False, no_args_is_help=True)

_MANIFEST_TEMPLATE = """# mlbisect manifest: replace example values with your changes.
#
# Your eval command must:
# 1. Read MLBISECT_AXIS_<NAME> for each axis below.
# 2. Seed all randomness from the integer in MLBISECT_SEED.
# 3. Print a final line like {\"metric\": 0.9123}, or write it to
#    the path in MLBISECT_OUTPUT.
#
# Start with 3 to 6 axes. Split a broad dependency change into separate axes
# when you need to distinguish which package caused the regression.
version: 1
name: my-regression

metric:
  name: accuracy
  direction: higher_is_better
  min_effect: 0.02

run:
  command: \"python eval.py\"
  timeout_s: 600
  workdir: .

axes:
  - name: data
    old: data/v1.parquet
    new: data/v2.parquet
  - name: code
    old: refs/good
    new: refs/bad
  # Suggested axis names: deps, checkpoint, prompt, config.
  # Add each changed dimension as another old/new pair.

budget:
  max_runs: 60
  baseline_runs: 5
  min_runs_per_config: 3
  max_runs_per_config: 10
  batch_size: 2
  confidence: 0.95
"""


def _version_callback(value: bool) -> bool:
    if value:
        typer.echo(__version__)
        raise typer.Exit
    return value


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the mlbisect version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Find the changes that caused a noisy ML regression."""
    del version


@app.command("init")
def init_manifest(
    path: Path = typer.Argument(
        Path("mlbisect.yaml"), help="Path for the new manifest template."
    ),
) -> None:
    """Write a commented manifest template."""
    if path.exists():
        typer.echo(f"manifest already exists: {path}; choose another path", err=True)
        raise typer.Exit(code=3)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_MANIFEST_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"could not write manifest '{path}': {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"wrote manifest template to {path}")


@app.command("validate")
def validate_manifest(
    path: Path = typer.Argument(
        Path("mlbisect.yaml"), help="Path to the manifest to validate."
    ),
) -> None:
    """Validate a manifest and print its run plan."""
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc

    missing = manifest.missing_references(path.parent.resolve())
    if missing:
        typer.echo("invalid manifest references:", err=True)
        for reference in missing:
            typer.echo(f"- missing path: {reference}", err=True)
        raise typer.Exit(code=3)

    axis_count = len(manifest.axes)
    exhaustive_runs = 2**axis_count * manifest.budget.max_runs_per_config
    one_at_a_time_runs = (
        manifest.budget.baseline_runs
        + axis_count * manifest.budget.max_runs_per_config
    )
    typer.echo(f"Manifest valid: {manifest.name}")
    typer.echo(f"Axes ({axis_count}): {', '.join(manifest.axis_names())}")
    typer.echo(
        "Metric: "
        f"{manifest.metric.name} ({manifest.metric.direction}, "
        f"min effect {manifest.metric.min_effect:g})"
    )
    typer.echo(f"Configured max runs: {manifest.budget.max_runs}")
    typer.echo(
        "Worst-case runs: "
        f"{exhaustive_runs} ({2**axis_count} configurations x "
        f"{manifest.budget.max_runs_per_config} runs)"
    )
    typer.echo(f"One-at-a-time estimate: {one_at_a_time_runs} runs")
    if exhaustive_runs > manifest.budget.max_runs:
        typer.echo("Warning: exhaustive evaluation exceeds the configured max_runs.")


@app.command("noise")
def noise_check(
    path: Path = typer.Argument(
        Path("mlbisect.yaml"), help="Path to the manifest to check."
    ),
) -> None:
    """Run an A/A check and estimate baseline evaluation noise."""
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc

    runs_per_batch = manifest.budget.baseline_runs
    required_runs = 2 * runs_per_batch
    if required_runs > manifest.budget.max_runs:
        typer.echo(
            "noise check needs two baseline batches, but max_runs is too small "
            f"({required_runs} required)",
            err=True,
        )
        raise typer.Exit(code=3)

    executor = LocalExecutor(manifest, base_dir=path.parent.resolve())
    first: list[float] = []
    second: list[float] = []
    errors = 0
    for seed in range(required_runs):
        result = executor.run(Configuration(), seed)
        if not result.ok or result.metric is None:
            errors += 1
            continue
        (first if seed < runs_per_batch else second).append(result.metric)

    if len(first) < 2 or len(second) < 2:
        typer.echo("noise check did not produce enough successful runs", err=True)
        raise typer.Exit(code=2)

    pooled = first + second
    noise_sd = stdev(pooled)
    typer.echo(f"Baseline A mean: {mean(first):g} (n={len(first)})")
    typer.echo(f"Baseline A/A mean: {mean(second):g} (n={len(second)})")
    typer.echo(f"Noise sd: {noise_sd:g}")
    typer.echo(f"Min effect: {manifest.metric.min_effect:g}")
    if errors:
        typer.echo(f"Warnings: {errors} failed runs were excluded.")
    if noise_sd >= manifest.metric.min_effect / 2:
        typer.echo("Warning: noise is comparable to the configured min_effect.")


@app.command("run")
def run_manifest(
    path: Path = typer.Argument(
        Path("mlbisect.yaml"), help="Path to the manifest to run."
    ),
    max_runs: int | None = typer.Option(
        None, "--max-runs", min=1, help="Override the manifest's global run budget."
    ),
    out: Path = typer.Option(
        Path("report.md"), "--out", help="Markdown report output path."
    ),
    json_path: Path = typer.Option(
        Path("report.json"), "--json", help="JSON report output path."
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Reuse completed runs from the JSONL run log."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List planned configurations without executing."
    ),
) -> None:
    """Search for culprit changes and write Markdown and JSON reports."""
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc

    missing = manifest.missing_references(path.parent.resolve())
    if missing:
        typer.echo("invalid manifest references:", err=True)
        for reference in missing:
            typer.echo(f"- missing path: {reference}", err=True)
        raise typer.Exit(code=3)

    effective_max_runs = max_runs or manifest.budget.max_runs
    if dry_run:
        _print_dry_run(manifest, effective_max_runs)
        return

    log_path = path.parent / ".mlbisect" / "runs.jsonl"
    fingerprint = manifest_fingerprint(manifest)
    try:
        store = RunStore(
            log_path,
            load_existing=resume,
            fingerprint=fingerprint,
        )
    except StoreError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc

    executor = LocalExecutor(manifest, base_dir=path.parent.resolve())
    progress = _Progress(effective_max_runs)
    with progress:
        sampler = AdaptiveSampler(
            executor,
            manifest,
            store=store,
            max_runs=effective_max_runs,
            progress_callback=progress.callback,
            initial_runs_used=store.run_count,
        )
        result = find_culprits(executor, manifest, sampler=sampler)

    try:
        write_reports(manifest, result, out, json_path)
    except OSError as exc:
        typer.echo(f"could not write report: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"wrote Markdown report to {out}")
    typer.echo(f"wrote JSON report to {json_path}")
    typer.echo(f"run log: {log_path}")
    raise typer.Exit(code=_exit_code(result))


def _print_dry_run(manifest: Manifest, max_runs: int) -> None:
    axis_names = manifest.axis_names()
    budget = manifest.budget
    typer.echo("Dry run: no evaluations will be executed.")
    typer.echo(f"Manifest: {manifest.name}")
    typer.echo("Planned configurations:")
    typer.echo("- {} (baseline)")
    typer.echo("- {" + ",".join(axis_names) + "} (candidate)")
    for axis in axis_names:
        typer.echo(f"- {{{axis}}} (singleton probe)")
    typer.echo(f"Global run budget: {max_runs}")
    typer.echo(
        "Exhaustive upper bound: "
        f"{2 ** len(axis_names) * budget.max_runs_per_config} runs"
    )


def _exit_code(result: SearchResult) -> int:
    if result.status == "culprit_found":
        return 0
    if result.status == "no_reproducible_regression":
        return 1
    return 2


class _Progress:
    """TTY-aware progress output without making Rich mandatory at import time."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.count = 0
        self._rich = None
        self._task_id: Any = None
        if sys.stdout.isatty():
            try:
                from rich.progress import Progress

                self._rich = Progress()
            except ImportError:
                self._rich = None

    def __enter__(self) -> "_Progress":
        if self._rich is not None:
            self._rich.start()
            self._task_id = self._rich.add_task("mlbisect runs", total=self.total)
        return self

    def __exit__(self, *args: object) -> None:
        if self._rich is not None:
            self._rich.stop()

    def callback(self, config: Configuration, result: object, cached: bool) -> None:
        self.count += 1
        metric = getattr(result, "metric", None)
        suffix = " cached" if cached else ""
        message = (
            f"run {self.count}/{self.total}: {config.id} seed "
            f"{getattr(result, 'seed', '?')} metric={metric}{suffix}"
        )
        if self._rich is not None and self._task_id is not None:
            self._rich.update(self._task_id, advance=0, description=message)
            self._rich.advance(self._task_id)
        else:
            typer.echo(message)


if __name__ == "__main__":
    app()
