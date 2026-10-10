"""Loading and discovery helpers for Regression Lab cases."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlbisect.manifest import Manifest, load_manifest


@dataclass(frozen=True)
class Truth:
    """Ground truth recorded by a lab case."""

    culprit_sets: tuple[frozenset[str], ...]
    expected_verdict: str
    kind: str


@dataclass(frozen=True)
class LabCase:
    """A case directory and its expected result."""

    path: Path
    id: str
    description: str
    axes: tuple[str, ...]
    truth: Truth

    @property
    def manifest_path(self) -> Path:
        return self.path / "mlbisect.yaml"

    def load_manifest(self) -> Manifest:
        manifest = load_manifest(self.manifest_path)
        if set(self.axes) != set(manifest.axis_names()):
            raise ValueError(
                f"case axes do not match manifest axes for {self.id}: "
                f"case={self.axes}, manifest={manifest.axis_names()}"
            )
        return manifest


def discover_cases(root: str | Path | None = None) -> tuple[LabCase, ...]:
    """Discover case directories in deterministic order."""
    cases_root = Path(root) if root is not None else Path(__file__).parent / "cases"
    cases: list[LabCase] = []
    for case_path in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_file = case_path / "case.yaml"
        if case_file.exists():
            cases.append(load_case(case_path))
    return tuple(cases)


def load_case(path: str | Path) -> LabCase:
    """Load one case definition."""
    case_path = Path(path)
    with (case_path / "case.yaml").open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"case file must contain a mapping: {case_path / 'case.yaml'}")

    truth_raw = raw.get("truth")
    if not isinstance(truth_raw, dict):
        raise ValueError(f"case truth must be a mapping: {case_path}")
    culprit_sets = tuple(
        frozenset(str(axis) for axis in culprit_set)
        for culprit_set in truth_raw.get("culprit_sets", [])
    )
    axes = tuple(str(axis) for axis in raw.get("axes", []))
    return LabCase(
        path=case_path,
        id=str(raw["id"]),
        description=str(raw["description"]),
        axes=axes,
        truth=Truth(
            culprit_sets=culprit_sets,
            expected_verdict=str(truth_raw["expected_verdict"]),
            kind=str(truth_raw.get("kind", "none")),
        ),
    )
