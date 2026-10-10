"""Deterministic executor for tests and future Regression Lab cases."""

import random
from collections.abc import Mapping

from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult


class FakeExecutor:
    """Simulate metric effects, interactions, and normally distributed noise."""

    def __init__(
        self,
        baseline: float = 0.9,
        effects: Mapping[str, float] | None = None,
        interactions: Mapping[frozenset[str], float] | None = None,
        noise_sd: float = 0.0,
    ) -> None:
        if noise_sd < 0:
            raise ValueError("noise_sd must be non-negative")
        self.baseline = baseline
        self.effects = dict(effects or {})
        self.interactions = {
            frozenset(axes): effect
            for axes, effect in (interactions or {}).items()
        }
        self.noise_sd = noise_sd

    def run(self, config: Configuration, seed: int) -> RunResult:
        metric = self.baseline
        metric += sum(self.effects.get(axis, 0.0) for axis in config.axes)
        metric += sum(
            effect for axes, effect in self.interactions.items() if axes <= config.axes
        )
        if self.noise_sd:
            metric += random.Random(seed).gauss(0.0, self.noise_sd)
        return RunResult(
            config=config.axes,
            seed=seed,
            metric=metric,
            ok=True,
            duration_s=0.0,
        )
