import random

import pytest

from mlbisect.stats import (
    bootstrap_ci,
    decide_by_point,
    decide_from_ci,
    estimate_drop,
)


def test_drop_handles_metric_direction() -> None:
    assert estimate_drop([0.9, 0.9], [0.8, 0.8], "higher_is_better") == pytest.approx(
        0.1
    )
    assert estimate_drop([0.1, 0.1], [0.2, 0.2], "lower_is_better") == pytest.approx(
        0.1
    )


def test_bootstrap_is_deterministic_and_handles_zero_variance() -> None:
    first = bootstrap_ci([1.0] * 5, [0.8] * 5, "higher_is_better")
    second = bootstrap_ci([1.0] * 5, [0.8] * 5, "higher_is_better")

    assert first == second
    assert first == (0.19999999999999996, 0.19999999999999996)
    assert decide_from_ci(first, 0.1, 5).verdict == "BAD"


def test_fewer_than_three_samples_never_decides() -> None:
    assert decide_from_ci((1.0, 1.0), 0.1, 2).verdict == "UNDECIDED"
    assert decide_by_point(1.0, 0.1, 2).verdict == "UNDECIDED"
    assert decide_by_point(1.0, 0.1, 2).low_confidence


def test_default_rule_error_rates_over_500_trials() -> None:
    rng = random.Random(2026)
    false_ok = 0
    false_bad = 0
    for _ in range(500):
        baseline = [rng.gauss(1.0, 0.02) for _ in range(5)]
        bad = [rng.gauss(0.9, 0.02) for _ in range(5)]
        no_effect = [rng.gauss(1.0, 0.02) for _ in range(5)]
        bad_decision = decide_from_ci(
            bootstrap_ci(baseline, bad, "higher_is_better"), 0.05, 5
        )
        no_effect_decision = decide_from_ci(
            bootstrap_ci(baseline, no_effect, "higher_is_better"), 0.05, 5
        )
        false_ok += bad_decision.verdict != "BAD"
        false_bad += no_effect_decision.verdict == "BAD"

    assert false_ok / 500 < 0.05
    assert false_bad / 500 < 0.05
