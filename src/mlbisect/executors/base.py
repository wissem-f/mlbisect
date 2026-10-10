"""Shared executor types."""

from dataclasses import dataclass
from typing import Protocol

from mlbisect.config import Configuration


@dataclass(frozen=True)
class RunResult:
    """Result of one evaluation command."""

    config: frozenset[str]
    seed: int
    metric: float | None
    ok: bool
    duration_s: float
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", frozenset(self.config))


class Executor(Protocol):
    """Interface implemented by local and simulated evaluators."""

    def run(self, config: Configuration, seed: int) -> RunResult:
        """Evaluate one configuration with one seed."""
