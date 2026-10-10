from pathlib import Path

import pytest
from pydantic import ValidationError

from mlbisect.manifest import Manifest, ManifestError, load_manifest


def _manifest_data() -> dict[str, object]:
    return {
        "version": 1,
        "name": "test-regression",
        "metric": {
            "name": "accuracy",
            "direction": "higher_is_better",
            "min_effect": 0.02,
        },
        "run": {"command": "python eval.py"},
        "axes": [
            {"name": "data", "old": "v1", "new": "v2"},
            {"name": "deps", "old": "old", "new": "new"},
        ],
        "budget": {
            "max_runs": 20,
            "baseline_runs": 4,
            "min_runs_per_config": 3,
            "max_runs_per_config": 6,
            "batch_size": 2,
            "confidence": 0.95,
        },
    }


def test_manifest_defaults_and_axis_order() -> None:
    manifest = Manifest.model_validate(_manifest_data())

    assert manifest.axis_names() == ("data", "deps")
    assert manifest.run.timeout_s == 600
    assert manifest.budget.max_runs_per_config == 6


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("axes", [{"name": "only", "old": "a", "new": "b"}], "at least 2"),
        (
            "axes",
            [
                {"name": "data", "old": "a", "new": "b"},
                {"name": "data", "old": "c", "new": "d"},
            ],
            "unique",
        ),
        (
            "axes",
            [
                {"name": "Data", "old": "a", "new": "b"},
                {"name": "deps", "old": "c", "new": "d"},
            ],
            "pattern",
        ),
    ],
)
def test_manifest_validation_errors(
    field: str, value: object, message: str
) -> None:
    data = _manifest_data()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        Manifest.model_validate(data)


def test_manifest_rejects_equal_values_and_invalid_budget() -> None:
    equal_values = _manifest_data()
    equal_values["axes"] = [
        {"name": "data", "old": "same", "new": "same"},
        {"name": "deps", "old": "old", "new": "new"},
    ]
    with pytest.raises(ValidationError, match="must differ"):
        Manifest.model_validate(equal_values)

    invalid_budget = _manifest_data()
    invalid_budget["budget"] = {
        "max_runs": 5,
        "baseline_runs": 3,
        "min_runs_per_config": 2,
        "max_runs_per_config": 1,
    }
    with pytest.raises(ValidationError, match="less than or equal"):
        Manifest.model_validate(invalid_budget)


def test_load_manifest_formats_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "mlbisect.yaml"
    path.write_text("version: 1\naxes: [", encoding="utf-8")

    with pytest.raises(ManifestError, match="invalid YAML"):
        load_manifest(path)


def test_missing_references_only_checks_path_like_values(tmp_path: Path) -> None:
    data = _manifest_data()
    data["axes"] = [
        {"name": "data", "old": "data/v1.parquet", "new": "data/v2.parquet"},
        {"name": "deps", "old": "v1", "new": "v2"},
    ]
    manifest = Manifest.model_validate(data)

    assert manifest.missing_references(tmp_path) == [
        "data: data/v1.parquet",
        "data: data/v2.parquet",
    ]
