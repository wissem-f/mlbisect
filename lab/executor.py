"""In-process executor used to run the lab benchmark efficiently."""

import time

from lab.datagen import evaluate_case
from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult
from mlbisect.manifest import Manifest


class CaseExecutor:
    """Implement the executor contract without spawning a process per run."""

    def __init__(self, case_id: str, manifest: Manifest) -> None:
        self.case_id = case_id
        self.manifest = manifest

    def run(self, config: Configuration, seed: int) -> RunResult:
        started = time.perf_counter()
        values = {
            axis.name: axis.new if axis.name in config.axes else axis.old
            for axis in self.manifest.axes
        }
        try:
            metric = evaluate_case(self.case_id, values, seed)
        except Exception as exc:
            return RunResult(
                config=config.axes,
                seed=seed,
                metric=None,
                ok=False,
                duration_s=time.perf_counter() - started,
                error=str(exc),
            )
        return RunResult(
            config=config.axes,
            seed=seed,
            metric=metric,
            ok=True,
            duration_s=time.perf_counter() - started,
        )
