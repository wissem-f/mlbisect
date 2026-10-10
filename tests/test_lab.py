from lab.baselines import exhaustive, one_at_a_time
from lab.cases import discover_cases
from lab.executor import CaseExecutor

from mlbisect.config import Configuration
from mlbisect.executors.local import LocalExecutor


def _mean(executor: CaseExecutor, config: Configuration) -> float:
    metrics = [executor.run(config, seed).metric for seed in range(10)]
    return sum(metric for metric in metrics if metric is not None) / len(metrics)


def _drop(baseline: float, estimate: float) -> float:
    return baseline - estimate


def test_ground_truth_effects_are_reproducible() -> None:
    cases = discover_cases()
    assert len(cases) == 6

    for case in cases:
        manifest = case.load_manifest()
        executor = CaseExecutor(case.id, manifest)
        baseline = _mean(executor, Configuration())
        threshold = manifest.metric.min_effect
        if case.truth.culprit_sets:
            for culprit in case.truth.culprit_sets:
                effect = _drop(baseline, _mean(executor, Configuration(culprit)))
                assert effect >= threshold, case.id
                complement = set(case.axes) - culprit
                complement_effect = _drop(
                    baseline, _mean(executor, Configuration.from_axes(complement))
                )
                assert complement_effect < threshold, case.id
        else:
            full_effect = _drop(
                baseline, _mean(executor, Configuration.from_axes(case.axes))
            )
            assert full_effect < threshold, case.id


def test_case_eval_scripts_run_end_to_end_under_ten_seconds() -> None:
    for case in discover_cases():
        manifest = case.load_manifest()
        local_result = LocalExecutor(manifest, base_dir=case.path).run(
            Configuration(), seed=0
        )
        in_process_result = CaseExecutor(case.id, manifest).run(
            Configuration(), seed=0
        )

        assert local_result.ok, f"{case.id}: {local_result.error}"
        assert local_result.metric is not None
        assert local_result.metric == in_process_result.metric, case.id
        assert local_result.duration_s < 10, case.id


def test_one_at_a_time_misses_interaction() -> None:
    case = next(case for case in discover_cases() if case.id == "L03_interaction")
    manifest = case.load_manifest()
    executor = CaseExecutor(case.id, manifest)

    one_at_a_time_result = one_at_a_time(executor, manifest, seed=2026)
    exhaustive_result = exhaustive(executor, manifest, seed=2026)

    assert one_at_a_time_result.culprit_sets == ()
    assert exhaustive_result.culprit_sets == (frozenset({"data", "deps"}),)


def test_benchmark_baselines_produce_records() -> None:
    from lab.run_benchmark import run_benchmark
    from lab.score import score_records

    cases = discover_cases()
    records = run_benchmark(
        cases,
        trials=2,
        master_seed=2026,
        methods=(
            "mlbisect",
            "one_at_a_time",
            "exhaustive",
            "ordered_bisect",
        ),
    )

    assert len(records) == 48
    summaries = score_records(records, cases)
    assert len(summaries) == 24
    interaction_one_at_a_time = next(
        summary
        for summary in summaries
        if summary["method"] == "one_at_a_time"
        and summary["case_id"] == "L03_interaction"
    )
    assert interaction_one_at_a_time["exact_match_rate"] == 0


def test_ml_bisect_benchmark_handles_interaction_noise_and_control() -> None:
    from lab.run_benchmark import run_benchmark

    cases = discover_cases()
    records = run_benchmark(cases, trials=2, master_seed=2026, methods=("mlbisect",))
    by_case = {record["case_id"]: record for record in records}

    assert by_case["L03_interaction"]["culprit_sets"] == [["data", "deps"]]
    assert by_case["L03_interaction"]["interactions"][0]["non_additive"] > 0.2
    assert by_case["L04_small_effect"]["verdict"] == "inconclusive"
    assert by_case["L04_small_effect"]["culprit_sets"] == []
    assert by_case["L06_no_regression"]["verdict"] == "no_reproducible_regression"
