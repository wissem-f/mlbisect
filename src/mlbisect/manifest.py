"""Manifest models, loading, and validation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ManifestError(ValueError):
    """Raised when a manifest cannot be loaded or does not follow the schema."""


def manifest_fingerprint(manifest: "Manifest") -> str:
    """Return a stable identity for a validated manifest."""
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Axis(BaseModel):
    """One old/new dimension of the regression."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    old: str
    new: str

    @model_validator(mode="after")
    def values_must_differ(self) -> "Axis":
        if self.old == self.new:
            raise ValueError("old and new values must differ")
        return self


class MetricConfig(BaseModel):
    """Metric direction and the smallest meaningful effect."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better"]
    min_effect: float = Field(gt=0)


class RunConfig(BaseModel):
    """Command execution settings."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    timeout_s: float = Field(default=600, gt=0)
    workdir: str = Field(default=".", min_length=1)


class BudgetConfig(BaseModel):
    """Global and per-configuration run limits."""

    model_config = ConfigDict(extra="forbid")

    max_runs: int = Field(default=60, ge=1)
    baseline_runs: int = Field(default=5, ge=1)
    min_runs_per_config: int = Field(default=3, ge=1)
    max_runs_per_config: int = Field(default=10, ge=1)
    batch_size: int = Field(default=2, ge=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_run_limits(self) -> "BudgetConfig":
        if self.min_runs_per_config > self.max_runs_per_config:
            raise ValueError(
                "min_runs_per_config must be less than or equal to "
                "max_runs_per_config"
            )
        required = self.baseline_runs + 2 * self.min_runs_per_config
        if self.max_runs < required:
            raise ValueError(
                "max_runs must be at least baseline_runs + "
                "2 * min_runs_per_config "
                f"({required})"
            )
        return self


class Manifest(BaseModel):
    """Complete mlbisect manifest."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    name: str = Field(min_length=1)
    metric: MetricConfig
    run: RunConfig
    axes: list[Axis] = Field(min_length=2)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @model_validator(mode="after")
    def axis_names_must_be_unique(self) -> "Manifest":
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            duplicate_names = ", ".join(duplicates)
            raise ValueError(
                f"axis names must be unique; duplicated: {duplicate_names}"
            )
        return self

    def axis_names(self) -> tuple[str, ...]:
        """Return axis names in manifest order."""
        return tuple(axis.name for axis in self.axes)

    def axis(self, name: str) -> Axis:
        """Return an axis by name."""
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(name)

    def missing_references(self, base_dir: Path) -> list[str]:
        """Return path-like old/new values that do not exist on disk."""
        missing: list[str] = []
        for axis in self.axes:
            for value in (axis.old, axis.new):
                if not _looks_like_path(value):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = base_dir / path
                if not path.exists():
                    missing.append(f"{axis.name}: {value}")
        return missing


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a YAML manifest from ``path``."""
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest '{manifest_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid YAML in '{manifest_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError("manifest must contain a YAML mapping at its root")

    try:
        return Manifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(_format_validation_error(exc)) from exc


def _format_validation_error(error: ValidationError) -> str:
    lines = ["invalid manifest:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def _looks_like_path(value: str) -> bool:
    if value.startswith(("git:", "http://", "https://", "registry:")):
        return False
    path = Path(value)
    return bool(path.suffix or "/" in value or "\\" in value)
