import pytest

from mlbisect.executors.fake import FakeExecutor
from mlbisect.manifest import Manifest
from mlbisect.search import SearchResult, ddmin, find_culprits


def _manifest(axis_names: list[str], max_runs: int = 500) -> Manifest:
    return Manifest.model_validate(
        {
            "version": 1,
            "name": "search-test",
            "metric": {
                "name": "accuracy",
                "direction": "higher_is_better",
                "min_effect": 0.1,
            },
            "run": {"command": "unused"},
            "axes": [
                {"name": name, "old": "v1", "new": "v2"}
                for name in axis_names
            ],
            "budget": {
                "max_runs": max_runs,
                "baseline_runs": 5,
                "min_runs_per_config": 3,
                "max_runs_per_config": 3,
                "batch_size": 1,
                "confidence": 0.95,
            },
        }
    )


def test_ddmin_returns_one_minimal_set_deterministically() -> None:
    calls: list[frozenset[str]] = []

    def test(config: frozenset[str]) -> bool:
        calls.append(config)
        return {"b", "d"}.issubset(config)

    result = ddmin(["a", "b", "c", "d"], test)

    assert result == frozenset({"b", "d"})
    assert all(
        not {"b", "d"}.issubset(result - {axis})
        for axis in result
    )
    assert calls == list(dict.fromkeys(calls))


def test_search_finds_single_culprit_and_beats_exhaustive() -> None:
    names = [f"axis_{index}" for index in range(1, 9)]
    manifest = _manifest(names)
    executor = FakeExecutor(baseline=1.0, effects={"axis_4": -0.3})

    result = find_culprits(executor, manifest)

    assert isinstance(result, SearchResult)
    assert result.status == "culprit_found"
    assert result.culprit_sets == (frozenset({"axis_4"}),)
    assert not result.incomplete
    assert result.configurations_tested < 2 ** len(names)


def test_search_finds_interaction() -> None:
    manifest = _manifest(["data", "deps", "prompt"])
    executor = FakeExecutor(
        baseline=1.0,
        interactions={frozenset({"data", "deps"}): -0.3},
    )

    result = find_culprits(executor, manifest)

    assert result.status == "culprit_found"
    assert result.culprit_sets == (frozenset({"data", "deps"}),)


def test_search_reports_non_additive_interaction_effect() -> None:
    manifest = _manifest(["data", "deps", "prompt"])
    executor = FakeExecutor(
        baseline=1.0,
        interactions={frozenset({"data", "deps"}): -0.3},
    )

    result = find_culprits(executor, manifest)

    interaction = result.interactions[0]
    assert interaction.label == "interaction"
    assert interaction.single_effects == (("data", 0.0), ("deps", 0.0))
    assert interaction.joint_effect == pytest.approx(0.3)
    assert interaction.non_additive == pytest.approx(0.3)


def test_search_finds_two_independent_culprits() -> None:
    manifest = _manifest(["a", "b", "c", "d"])
    executor = FakeExecutor(baseline=1.0, effects={"a": -0.3, "c": -0.25})

    result = find_culprits(executor, manifest)

    assert result.status == "culprit_found"
    assert set(result.culprit_sets) == {frozenset({"a"}), frozenset({"c"})}


def test_search_returns_no_reproducible_regression() -> None:
    manifest = _manifest(["data", "deps"])
    result = find_culprits(FakeExecutor(baseline=1.0), manifest)

    assert result.status == "no_reproducible_regression"
    assert result.culprit_sets == ()
    assert not result.incomplete


def test_search_returns_ranked_incomplete_result_when_budget_exhausts() -> None:
    manifest = _manifest(["a", "b", "c", "d"], max_runs=11)
    executor = FakeExecutor(baseline=1.0, effects={"c": -0.3})

    result = find_culprits(executor, manifest)

    assert result.incomplete
    assert result.status == "inconclusive"
    assert result.ranked_suspects
    assert result.ranked_suspects[0].axis == "c"
    assert result.runs_used <= manifest.budget.max_runs
