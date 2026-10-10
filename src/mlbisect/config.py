"""Configuration identifiers used by evaluators and the run store."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Configuration:
    """A set of axes currently using their new values."""

    axes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", frozenset(self.axes))

    @classmethod
    def from_axes(cls, axes: Iterable[str]) -> "Configuration":
        """Build a configuration from any iterable of axis names."""
        return cls(frozenset(axes))

    @property
    def id(self) -> str:
        """Return a stable, human-readable identifier for this configuration."""
        return "{" + ",".join(sorted(self.axes)) + "}"

    def ordered(self, axis_order: Iterable[str]) -> tuple[str, ...]:
        """Return selected axes in the order supplied by the manifest."""
        return tuple(axis for axis in axis_order if axis in self.axes)

    def __str__(self) -> str:
        return self.id
