from pathlib import Path

from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult
from mlbisect.executors.fake import FakeExecutor
from mlbisect.manifest import Manifest
from mlbisect.sampler import AdaptiveSampler
from mlbisect.store import RunStore


def _manifest(
    *, max_runs: int = 30, max_runs_per_config: int = 10
) -> Manifest:
    return Manifest.model_validate(
        {
            "version": 1,
            "name": "sampler-test",
            "metric": {
                "name": "accuracy",
                "direction": "higher_is_better",
                "min_effect": 0.1,
            },
            "run": {"command": "unused"},
            "axes": [
                {"name": "data", "old": "v1", "new": "v2"},
                {"name": "deps", "old": "v1", "new": "v2"},
            ],
            "budget": {
                "max_runs": max_runs,
                "baseline_runs": 5,
                "min_runs_per_config": 3,
                "max_runs_per_config": max_runs_per_config,
                "batch_size": 2,
                "confidence": 0.95,
            },
        }
    )


def test_sampler_decides_bad_with_minimum_runs() -> None:
    manifest = _manifest()
    sampler = AdaptiveSampler(
        FakeExecutor(baseline=1.0, effects={"data": -0.3}), manifest
    )

    estimate = sampler.estimate(Configuration.from_axes(["data"]))

    assert estimate.verdict == "BAD"
    assert estimate.n == 3
    assert sampler.runs_used == 8
    assert not estimate.low_confidence


def test_sampler_reuses_store_without_new_runs(tmp_path: Path) -> None:
    manifest = _manifest()
    store = RunStore(tmp_path / "runs.jsonl")
    executor = FakeExecutor(baseline=1.0, effects={"data": -0.3})
    first_sampler = AdaptiveSampler(
        executor, manifest, store=store
    )
    first = first_sampler.estimate(Configuration.from_axes(["data"]))
    second_sampler = AdaptiveSampler(
        executor, manifest, baseline_samples=first_sampler.baseline_samples, store=store
    )

    second = second_sampler.estimate(Configuration.from_axes(["data"]))

    assert first.verdict == second.verdict == "BAD"
    assert second_sampler.runs_used == 0


def test_sampler_marks_budget_exhaustion_low_confidence() -> None:
    manifest = _manifest(max_runs=11, max_runs_per_config=10)
    sampler = AdaptiveSampler(
        FakeExecutor(baseline=1.0, effects={"data": -0.01}, noise_sd=0.2),
        manifest,
    )

    estimate = sampler.estimate(Configuration.from_axes(["data"]))

    assert sampler.runs_used <= 11
    assert estimate.low_confidence


class _RetryExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.seeds: list[int] = []

    def run(self, config: Configuration, seed: int) -> RunResult:
        self.calls += 1
        self.seeds.append(seed)
        if self.calls == 1:
            return RunResult(config.axes, seed, None, False, 0.0, "temporary")
        return RunResult(config.axes, seed, 1.0, True, 0.0)


def test_sampler_retries_failed_runs_once() -> None:
    manifest = _manifest()
    executor = _RetryExecutor()
    sampler = AdaptiveSampler(executor, manifest, baseline_samples=[1.0] * 5)

    estimate = sampler.estimate(Configuration.from_axes(["data"]))

    assert estimate.verdict == "OK"
    assert estimate.errors == 0
    assert executor.calls == 4
    assert sampler.runs_used == 4


def test_resume_counts_historical_runs_against_global_budget(tmp_path: Path) -> None:
    manifest = _manifest(max_runs=11, max_runs_per_config=10)
    path = tmp_path / "runs.jsonl"
    first_store = RunStore(path, fingerprint="same")
    first = AdaptiveSampler(
        FakeExecutor(baseline=1.0, effects={"data": -0.3}),
        manifest,
        store=first_store,
        max_runs=11,
    )
    first.estimate(Configuration.from_axes(["data"]))
    second_store = RunStore(path, fingerprint="same")
    second = AdaptiveSampler(
        FakeExecutor(baseline=1.0, effects={"data": -0.3}),
        manifest,
        store=second_store,
        max_runs=11,
        initial_runs_used=second_store.run_count,
    )

    estimate = second.estimate(Configuration.from_axes(["data"]))

    assert second_store.run_count == 8
    assert second.runs_used == 8
    assert estimate.verdict == "BAD"


def test_cached_failed_run_is_not_retried_twice(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "runs.jsonl"
    store = RunStore(path, fingerprint="same")
    failed = RunResult(frozenset({"data"}), 0, None, False, 0.0, "failed")
    store.append(failed)
    store.append(failed)

    executor = _RetryExecutor()
    sampler = AdaptiveSampler(
        executor,
        manifest,
        baseline_samples=[1.0] * 5,
        store=RunStore(path, fingerprint="same"),
        max_runs=30,
    )
    sampler.estimate(Configuration.from_axes(["data"]))

    assert executor.calls == 4
    assert 0 not in executor.seeds
