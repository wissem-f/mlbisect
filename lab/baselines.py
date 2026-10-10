"""Fixed-budget baseline methods for the Regression Lab."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from mlbisect.config import Configuration
from mlbisect.executors.base import Executor
from mlbisect.manifest import Manifest
from mlbisect.sampler import AdaptiveSampler
from mlbisect.search import find_culprits

Verdict = Literal["culprit_found", "no_regression", "inconclusive"]


@dataclass(frozen=True)
class BaselineResult:
    """A baseline method's answer for one case and trial."""

    method: str
    culprit_sets: tuple[frozenset[str], ...]
    verdict: Verdict
    runs_used: int
    configurations_tested: int
    interactions: tuple[dict[str, object], ...] = ()
    tentative_culprit_sets: tuple[frozenset[str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "culprit_sets": [sorted(culprit) for culprit in self.culprit_sets],
            "verdict": self.verdict,
            "runs_used": self.runs_used,
            "configurations_tested": self.configurations_tested,
            "interactions": list(self.interactions),
            "tentative_culprit_sets": [
                sorted(culprit) for culprit in self.tentative_culprit_sets
            ],
        }


def one_at_a_time(
    executor: Executor,
    manifest: Manifest,
    seed: int = 0,
    runs_per_config: int = 10,
) -> BaselineResult:
    """Test each axis alone and blame axes with a point-estimate drop."""
    baseline, baseline_runs = _evaluate(
        executor, manifest, Configuration(), runs_per_config, seed
    )
    culprit_sets: list[frozenset[str]] = []
    runs_used = baseline_runs
    configurations = 1
    for index, axis in enumerate(manifest.axis_names()):
        estimate, runs = _evaluate(
            executor,
            manifest,
            Configuration.from_axes([axis]),
            runs_per_config,
            seed + (index + 1) * 10_000,
        )
        runs_used += runs
        configurations += 1
        if _drop(manifest, baseline, estimate) >= manifest.metric.min_effect:
            culprit_sets.append(frozenset({axis}))
    verdict: Verdict = "culprit_found" if culprit_sets else "no_regression"
    return BaselineResult(
        method="one_at_a_time",
        culprit_sets=tuple(culprit_sets),
        verdict=verdict,
        runs_used=runs_used,
        configurations_tested=configurations,
    )


def exhaustive(
    executor: Executor,
    manifest: Manifest,
    seed: int = 0,
    runs_per_config: int = 10,
) -> BaselineResult:
    """Evaluate every subset and return the 1-minimal bad configurations."""
    axis_names = manifest.axis_names()
    estimates: dict[frozenset[str], float] = {}
    runs_used = 0
    for size in range(len(axis_names) + 1):
        for axes in combinations(axis_names, size):
            config = Configuration.from_axes(axes)
            mean, runs = _evaluate(
                executor,
                manifest,
                config,
                runs_per_config,
                seed + len(estimates) * 10_000,
            )
            estimates[config.axes] = mean
            runs_used += runs

    baseline = estimates[frozenset()]
    bad_sets = {
        config
        for config, mean in estimates.items()
        if config and _drop(manifest, baseline, mean) >= manifest.metric.min_effect
    }
    minimal = tuple(
        sorted(
            (
                config
                for config in bad_sets
                if not any(subset < config for subset in bad_sets)
            ),
            key=lambda config: (len(config), tuple(sorted(config))),
        )
    )
    verdict: Verdict = "culprit_found" if minimal else "no_regression"
    return BaselineResult(
        method="exhaustive",
        culprit_sets=minimal,
        verdict=verdict,
        runs_used=runs_used,
        configurations_tested=len(estimates),
    )


def ordered_bisect(
    executor: Executor,
    manifest: Manifest,
    seed: int = 0,
    runs_per_config: int = 5,
) -> BaselineResult:
    """Bisect ordered prefixes, treating axes as a single timeline."""
    axes = manifest.axis_names()
    baseline, runs_used = _evaluate(
        executor, manifest, Configuration(), runs_per_config, seed
    )
    configurations = 1
    full = Configuration.from_axes(axes)
    candidate, runs = _evaluate(
        executor, manifest, full, runs_per_config, seed + 10_000
    )
    runs_used += runs
    configurations += 1
    if _drop(manifest, baseline, candidate) < manifest.metric.min_effect:
        return BaselineResult(
            "ordered_bisect", (), "no_regression", runs_used, configurations
        )

    low = 0
    high = len(axes)
    while high - low > 1:
        midpoint = (low + high) // 2
        config = Configuration.from_axes(axes[:midpoint])
        estimate, runs = _evaluate(
            executor, manifest, config, runs_per_config, seed + (midpoint + 1) * 10_000
        )
        runs_used += runs
        configurations += 1
        if _drop(manifest, baseline, estimate) >= manifest.metric.min_effect:
            high = midpoint
        else:
            low = midpoint

    return BaselineResult(
        method="ordered_bisect",
        culprit_sets=(frozenset({axes[high - 1]}),),
        verdict="culprit_found",
        runs_used=runs_used,
        configurations_tested=configurations,
    )


def mlbisect(
    executor: Executor,
    manifest: Manifest,
    seed: int = 0,
) -> BaselineResult:
    """Run the adaptive mlbisect search against the executor interface."""
    sampler = AdaptiveSampler(executor, manifest, seed_start=seed)
    result = find_culprits(executor, manifest, sampler=sampler)
    interactions = tuple(
        {
            "culprit_set": sorted(interaction.culprit_set),
            "label": interaction.label,
            "single_effects": dict(interaction.single_effects),
            "joint_effect": interaction.joint_effect,
            "non_additive": interaction.non_additive,
        }
        for interaction in result.interactions
    )
    return BaselineResult(
        method="mlbisect",
        culprit_sets=(
            result.culprit_sets if result.status == "culprit_found" else ()
        ),
        verdict=result.status,
        runs_used=result.runs_used,
        configurations_tested=result.configurations_tested,
        interactions=interactions,
        tentative_culprit_sets=(
            () if result.status == "culprit_found" else result.culprit_sets
        ),
    )


def _evaluate(
    executor: Executor,
    manifest: Manifest,
    config: Configuration,
    runs: int,
    seed: int,
) -> tuple[float, int]:
    metrics: list[float] = []
    for offset in range(runs):
        result = executor.run(config, seed + offset)
        if result.ok and result.metric is not None:
            metrics.append(result.metric)
    if not metrics:
        raise RuntimeError(f"all runs failed for configuration {config}")
    return sum(metrics) / len(metrics), runs


def _drop(manifest: Manifest, baseline: float, estimate: float) -> float:
    if manifest.metric.direction == "higher_is_better":
        return baseline - estimate
    return estimate - baseline
