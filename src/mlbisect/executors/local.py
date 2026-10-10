"""Subprocess executor for local evaluation commands."""

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult
from mlbisect.manifest import Manifest


class LocalExecutor:
    """Run a manifest command with mlbisect's environment contract."""

    def __init__(self, manifest: Manifest, base_dir: str | Path | None = None) -> None:
        self.manifest = manifest
        root = Path(base_dir or Path.cwd()).resolve()
        workdir = Path(manifest.run.workdir)
        self.workdir = workdir if workdir.is_absolute() else root / workdir

    def run(self, config: Configuration, seed: int) -> RunResult:
        output_path = self._temporary_output_path()
        environment = self._environment(config, seed, output_path)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.manifest.run.command,
                shell=True,
                cwd=self.workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.manifest.run.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = RunResult(
                config=config.axes,
                seed=seed,
                metric=None,
                ok=False,
                duration_s=time.perf_counter() - started,
                error=(
                    f"command timed out after {self.manifest.run.timeout_s:g}s"
                    + self._timeout_output(exc)
                ),
            )
            output_path.unlink(missing_ok=True)
            return result
        except OSError as exc:
            result = RunResult(
                config=config.axes,
                seed=seed,
                metric=None,
                ok=False,
                duration_s=time.perf_counter() - started,
                error=f"could not start command: {exc}",
            )
            output_path.unlink(missing_ok=True)
            return result

        duration = time.perf_counter() - started
        try:
            if completed.returncode != 0:
                detail = completed.stderr.strip()
                error = f"command exited with status {completed.returncode}"
                if detail:
                    error += f": {detail}"
                return RunResult(
                    config=config.axes,
                    seed=seed,
                    metric=None,
                    ok=False,
                    duration_s=duration,
                    error=error,
                )

            metric = self._read_metric(output_path, completed.stdout)
            return RunResult(
                config=config.axes,
                seed=seed,
                metric=metric,
                ok=True,
                duration_s=duration,
            )
        except ValueError as exc:
            return RunResult(
                config=config.axes,
                seed=seed,
                metric=None,
                ok=False,
                duration_s=duration,
                error=str(exc),
            )
        finally:
            output_path.unlink(missing_ok=True)

    def _environment(
        self, config: Configuration, seed: int, output_path: Path
    ) -> dict[str, str]:
        environment = os.environ.copy()
        environment["MLBISECT_SEED"] = str(seed)
        environment["MLBISECT_CONFIG"] = ",".join(
            config.ordered(self.manifest.axis_names())
        )
        environment["MLBISECT_OUTPUT"] = str(output_path)
        for axis in self.manifest.axes:
            value = axis.new if axis.name in config.axes else axis.old
            environment[f"MLBISECT_AXIS_{axis.name.upper()}"] = value
        return environment

    @staticmethod
    def _temporary_output_path() -> Path:
        handle = tempfile.NamedTemporaryFile(
            prefix="mlbisect-", suffix=".json", delete=False
        )
        path = Path(handle.name)
        handle.close()
        path.unlink()
        return path

    @staticmethod
    def _read_metric(output_path: Path, stdout: str) -> float:
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as handle:
                    payload: Any = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not parse MLBISECT_OUTPUT: {exc}") from exc
        else:
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if not lines:
                raise ValueError("command produced no metric JSON")
            try:
                payload = json.loads(lines[-1])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "final stdout line is not metric JSON: " + lines[-1]
                ) from exc

        if not isinstance(payload, dict) or "metric" not in payload:
            raise ValueError("metric JSON must be an object containing 'metric'")
        metric = payload["metric"]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError("metric must be a finite number")
        value = float(metric)
        if not math.isfinite(value):
            raise ValueError("metric must be a finite number")
        return value

    @staticmethod
    def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
        raw_output: str | bytes | None = exc.stderr or exc.stdout
        if not raw_output:
            return ""
        if isinstance(raw_output, bytes):
            output = raw_output.decode(errors="replace")
        else:
            output = raw_output
        return f": {output.strip()}"
