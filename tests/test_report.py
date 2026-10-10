import json

import pytest

from mlbisect.executors.fake import FakeExecutor
from mlbisect.manifest import Manifest
from mlbisect.report import build_report, render_json, render_markdown
from mlbisect.sampler import ConfigEstimate
from mlbisect.search import SearchResult, find_culprits


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "version": 1,
            "name": "report-test",
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
                "max_runs": 100,
                "baseline_runs": 5,
                "min_runs_per_config": 3,
                "max_runs_per_config": 3,
                "batch_size": 1,
                "confidence": 0.95,
            },
        }
    )


def test_markdown_report_snapshot() -> None:
    manifest = _manifest()
    result = find_culprits(
        FakeExecutor(baseline=1.0, effects={"data": -0.3}), manifest
    )

    markdown = render_markdown(build_report(manifest, result))

    assert markdown == (
        "# mlbisect report: report-test\n"
        "\n"
        "Metric: accuracy (higher_is_better, min effect 0.1)\n"
        "\n"
        "Result: culprit found, confidence high\n"
        "Culprit set: data\n"
        "\n"
        "Baseline 1 (n=5, sd=0) | Candidate 0.7 (n=3, sd=0)\n"
        "\n"
        "| Configuration | Runs | Mean | Drop | Verdict |\n"
        "|---|---:|---:|---:|---|\n"
        "| {} | 5 | 1 | 0 | OK |\n"
        "| {data} | 3 | 0.7 | 0.3 | BAD |\n"
        "| {deps} | 3 | 1 | 0 | OK |\n"
        "| {data,deps} | 3 | 0.7 | 0.3 | BAD |\n"
        "\n"
        "Runs used: 14 | one-at-a-time baseline: 11 | exhaustive: 12\n"
        "Warnings: none\n"
    )


def test_json_report_is_serializable_and_contains_same_result() -> None:
    manifest = _manifest()
    result = find_culprits(
        FakeExecutor(baseline=1.0, effects={"data": -0.3}), manifest
    )

    payload = json.loads(render_json(build_report(manifest, result)))

    assert payload["status"] == "culprit_found"
    assert payload["culprit_sets"] == [["data"]]
    assert payload["runs"]["exhaustive"] == 12


def test_interaction_is_rendered_with_non_additive_effect() -> None:
    manifest = _manifest()
    result = find_culprits(
        FakeExecutor(
            baseline=1.0,
            interactions={frozenset({"data", "deps"}): -0.3},
        ),
        manifest,
    )

    report = build_report(manifest, result)
    markdown = render_markdown(report)

    assert report["kind"] == "interaction"
    assert report["interactions"][0]["non_additive"] == pytest.approx(0.3)
    assert "Interaction: data 0 + deps 0" in markdown
    assert "non-additive 0.3" in markdown


def test_inconclusive_culprits_are_tentative_only() -> None:
    manifest = _manifest()
    result = SearchResult(
        status="inconclusive",
        culprit_sets=(frozenset({"data"}),),
        incomplete=True,
        runs_used=8,
        configurations_tested=2,
        estimates=(
            ConfigEstimate(frozenset(), 5, 1.0, 0.0, (0.0, 0.0), "OK", False),
        ),
    )

    report = build_report(manifest, result)

    assert report["culprit_sets"] == []
    assert report["tentative_culprit_sets"] == [["data"]]
    assert report["kind"] is None
