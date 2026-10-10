"""Deterministic statistics and decision rules for noisy evaluations."""

import random
from dataclasses import dataclass
from statistics import fmean
from typing import Literal, Sequence

MetricDirection = Literal["higher_is_better", "lower_is_better"]
Verdict = Literal["BAD", "OK", "UNDECIDED"]


@dataclass(frozen=True)
class Decision:
    """A statistical verdict and whether it was forced by a limit."""

    verdict: Verdict
    low_confidence: bool = False


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, with a useful empty-input error."""
    if not values:
        raise ValueError("cannot compute a mean of no values")
    return fmean(values)


def estimate_drop(
    baseline: Sequence[float],
    configuration: Sequence[float],
    direction: MetricDirection,
) -> float:
    """Estimate the regression drop relative to baseline."""
    baseline_mean = mean(baseline)
    configuration_mean = mean(configuration)
    if direction == "higher_is_better":
        return baseline_mean - configuration_mean
    return configuration_mean - baseline_mean


def bootstrap_ci(
    baseline: Sequence[float],
    configuration: Sequence[float],
    direction: MetricDirection,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the drop."""
    if not baseline or not configuration:
        raise ValueError("bootstrap samples cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    rng = random.Random(seed)
    baseline_size = len(baseline)
    configuration_size = len(configuration)
    drops: list[float] = []
    for _ in range(resamples):
        baseline_mean = sum(
            baseline[rng.randrange(baseline_size)] for _ in range(baseline_size)
        ) / baseline_size
        configuration_mean = sum(
            configuration[rng.randrange(configuration_size)]
            for _ in range(configuration_size)
        ) / configuration_size
        if direction == "higher_is_better":
            drops.append(baseline_mean - configuration_mean)
        else:
            drops.append(configuration_mean - baseline_mean)

    drops.sort()
    alpha = (1.0 - confidence) / 2.0
    return _percentile(drops, alpha), _percentile(drops, 1.0 - alpha)


def decide_from_ci(
    drop_ci: tuple[float, float], min_effect: float, sample_count: int
) -> Decision:
    """Apply the midpoint threshold to a confidence interval."""
    if sample_count < 3:
        return Decision("UNDECIDED")
    threshold = min_effect / 2.0
    low, high = drop_ci
    if low >= threshold:
        return Decision("BAD")
    if high < threshold:
        return Decision("OK")
    return Decision("UNDECIDED")


def decide_by_point(drop: float, min_effect: float, sample_count: int) -> Decision:
    """Return the mandated low-confidence point-estimate decision."""
    if sample_count < 3:
        return Decision("UNDECIDED", low_confidence=True)
    verdict: Verdict = "BAD" if drop >= min_effect / 2.0 else "OK"
    return Decision(verdict, low_confidence=True)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * fraction
