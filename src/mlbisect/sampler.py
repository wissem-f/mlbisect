"""Adaptive, budget-aware sampling for individual configurations."""

from dataclasses import dataclass
from statistics import stdev
from typing import Callable, Iterable, Sequence

from mlbisect.config import Configuration
from mlbisect.executors.base import Executor, RunResult
from mlbisect.manifest import Manifest
from mlbisect.stats import (
    Decision,
    Verdict,
    bootstrap_ci,
    decide_by_point,
    decide_from_ci,
    estimate_drop,
    mean,
)
from mlbisect.store import RunStore

ProgressCallback = Callable[[Configuration, RunResult, bool], None]


@dataclass(frozen=True)
class ConfigEstimate:
    """Current evidence and verdict for one configuration."""

    config: frozenset[str]
    n: int
    mean: float | None
    drop: float | None
    ci: tuple[float, float] | None
    verdict: Verdict
    low_confidence: bool
    errors: int = 0
    sd: float | None = None


class AdaptiveSampler:
    """Sample configurations until their confidence interval is decisive."""

    def __init__(
        self,
        executor: Executor,
        manifest: Manifest,
        baseline_samples: Sequence[float] = (),
        store: RunStore | None = None,
        max_runs: int | None = None,
        seed_start: int = 0,
        progress_callback: ProgressCallback | None = None,
        initial_runs_used: int = 0,
    ) -> None:
        self.executor = executor
        self.manifest = manifest
        self.store = store
        self.max_runs = max_runs or manifest.budget.max_runs
        self.seed_start = seed_start
        self.progress_callback = progress_callback
        self.runs_used = initial_runs_used
        self._samples: dict[frozenset[str], list[float]] = {
            frozenset(): list(baseline_samples)
        }
        self._errors: dict[frozenset[str], int] = {frozenset(): 0}
        self._attempts: dict[frozenset[str], int] = {frozenset(): 0}
        self._attempted: dict[frozenset[str], set[int]] = {
            frozenset(): set(range(len(baseline_samples)))
        }

    @property
    def budget_exhausted(self) -> bool:
        """Whether no further uncached executor run can be made."""
        return self.runs_used >= self.max_runs

    @property
    def baseline_samples(self) -> tuple[float, ...]:
        """Return the currently available baseline metrics."""
        return tuple(self._samples.get(frozenset(), ()))

    def sample_baseline(self, runs: int | None = None) -> tuple[float, ...]:
        """Collect baseline samples, reusing any cached results."""
        target = runs if runs is not None else self.manifest.budget.baseline_runs
        self._ensure_samples(Configuration(), target)
        return self.baseline_samples

    def estimate(
        self, config: Configuration | Iterable[str]
    ) -> ConfigEstimate:
        """Adaptively estimate one configuration under all configured limits."""
        configuration = (
            config
            if isinstance(config, Configuration)
            else Configuration.from_axes(config)
        )
        if not self.baseline_samples:
            self.sample_baseline()
        if not self.baseline_samples:
            return ConfigEstimate(
                configuration.axes, 0, None, None, None, "UNDECIDED", True, 0
            )
        if not configuration.axes:
            return self._baseline_estimate(configuration)

        target = self.manifest.budget.min_runs_per_config
        maximum = self.manifest.budget.max_runs_per_config
        while True:
            self._ensure_samples(configuration, target)
            samples = self._samples.get(configuration.axes, [])
            estimate = self._estimate(configuration, samples)
            if estimate.n >= 3 and estimate.verdict != "UNDECIDED":
                return estimate
            if estimate.n >= maximum or self.budget_exhausted:
                return self._with_point_decision(estimate)
            next_target = min(maximum, target + self.manifest.budget.batch_size)
            if next_target <= target:
                return self._with_point_decision(estimate)
            target = next_target

    def sample(self, config: Configuration | Iterable[str]) -> ConfigEstimate:
        """Alias for :meth:`estimate`."""
        return self.estimate(config)

    def _ensure_samples(self, config: Configuration, target: int) -> None:
        axes = config.axes
        samples = self._samples.setdefault(axes, [])
        attempted = self._attempted.setdefault(axes, set())
        errors = self._errors.setdefault(axes, 0)
        offset = 0
        while len(samples) < target:
            while self.seed_start + offset in attempted:
                offset += 1
            if self.budget_exhausted:
                break
            seed = self.seed_start + offset
            if self._attempt_count(config) >= self._config_limit(config):
                if self.store is None or self.store.get(config, seed) is None:
                    break
            attempted.add(seed)
            result = self._run(config, seed)
            if result is None:
                break
            if result.ok and result.metric is not None:
                samples.append(result.metric)
            else:
                errors += 1
            offset += 1
            self._errors[axes] = errors

    def _config_limit(self, config: Configuration) -> int:
        if not config.axes:
            return max(
                self.manifest.budget.baseline_runs,
                self.manifest.budget.max_runs_per_config,
            )
        return self.manifest.budget.max_runs_per_config

    def _run(self, config: Configuration, seed: int) -> RunResult | None:
        if self.store is not None:
            cached = self.store.get(config, seed)
            if cached is not None:
                if self.progress_callback is not None:
                    self.progress_callback(config, cached, True)
                if cached.ok:
                    return cached
                history = self.store.history(config, seed)
                if len(history) >= 2:
                    return cached
                if (
                    self.budget_exhausted
                    or self._attempt_count(config) >= self._config_limit(config)
                ):
                    return cached
                retry = self.executor.run(config, seed)
                self.runs_used += 1
                if self.store is not None:
                    self.store.append(retry)
                if self.progress_callback is not None:
                    self.progress_callback(config, retry, False)
                return retry
        if self.budget_exhausted:
            return None
        result = self.executor.run(config, seed)
        self.runs_used += 1
        self._attempts[config.axes] = self._attempts.get(config.axes, 0) + 1
        if self.store is not None:
            self.store.append(result)
        if self.progress_callback is not None:
            self.progress_callback(config, result, False)
        if result.ok:
            return result

        if (
            self.budget_exhausted
            or self._attempt_count(config) >= self._config_limit(config)
        ):
            return result
        retry = self.executor.run(config, seed)
        self.runs_used += 1
        self._attempts[config.axes] = self._attempts.get(config.axes, 0) + 1
        if self.store is not None:
            self.store.append(retry)
        if self.progress_callback is not None:
            self.progress_callback(config, retry, False)
        return retry

    def _attempt_count(self, config: Configuration) -> int:
        if self.store is not None:
            return self.store.attempts_for_config(config)
        return self._attempts.get(config.axes, 0)

    def _estimate(
        self, config: Configuration, samples: Sequence[float]
    ) -> ConfigEstimate:
        if not samples:
            return ConfigEstimate(
                config.axes,
                0,
                None,
                None,
                None,
                "UNDECIDED",
                False,
                self._errors.get(config.axes, 0),
            )
        drop = estimate_drop(
            self.baseline_samples,
            samples,
            self.manifest.metric.direction,
        )
        ci = bootstrap_ci(
            self.baseline_samples,
            samples,
            self.manifest.metric.direction,
            confidence=self.manifest.budget.confidence,
            seed=self._bootstrap_seed(config),
        )
        decision = decide_from_ci(
            ci,
            self.manifest.metric.min_effect,
            min(len(samples), len(self.baseline_samples)),
        )
        return ConfigEstimate(
            config.axes,
            len(samples),
            mean(samples),
            drop,
            ci,
            decision.verdict,
            decision.low_confidence,
            self._errors.get(config.axes, 0),
            stdev(samples) if len(samples) >= 2 else None,
        )

    def _baseline_estimate(self, config: Configuration) -> ConfigEstimate:
        samples = self.baseline_samples
        verdict: Verdict = "OK" if len(samples) >= 3 else "UNDECIDED"
        return ConfigEstimate(
            config.axes,
            len(samples),
            mean(samples) if samples else None,
            0.0 if samples else None,
            (0.0, 0.0) if len(samples) >= 3 else None,
            verdict,
            False,
            self._errors.get(config.axes, 0),
            stdev(samples) if len(samples) >= 2 else None,
        )

    def _with_point_decision(self, estimate: ConfigEstimate) -> ConfigEstimate:
        if estimate.drop is None or len(self.baseline_samples) < 3:
            return ConfigEstimate(
                estimate.config,
                estimate.n,
                estimate.mean,
                estimate.drop,
                estimate.ci,
                "UNDECIDED",
                True,
                estimate.errors,
                estimate.sd,
            )
        decision: Decision = decide_by_point(
            estimate.drop,
            self.manifest.metric.min_effect,
            estimate.n,
        )
        return ConfigEstimate(
            estimate.config,
            estimate.n,
            estimate.mean,
            estimate.drop,
            estimate.ci,
            decision.verdict,
            True,
            estimate.errors,
            estimate.sd,
        )

    @staticmethod
    def _bootstrap_seed(config: Configuration) -> int:
        return sum(
            (index + 1) * ord(character)
            for index, character in enumerate(config.id)
        )


Sampler = AdaptiveSampler
