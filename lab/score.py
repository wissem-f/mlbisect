"""Score Regression Lab benchmark results and render Markdown."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from lab.cases import LabCase, discover_cases


def score_records(
    records: list[dict[str, Any]], cases: tuple[LabCase, ...]
) -> list[dict[str, object]]:
    """Aggregate exact-match, false-accusation, and run-count metrics."""
    cases_by_id = {case.id: case for case in cases}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), str(record["case_id"]))].append(record)

    summaries: list[dict[str, object]] = []
    for (method, case_id), group in sorted(grouped.items()):
        case = cases_by_id[case_id]
        exact = sum(_is_exact(record, case) for record in group) / len(group)
        false_accusation = sum(
            _is_false_accusation(record, case) for record in group
        ) / len(group)
        inconclusive = sum(
            record["verdict"] == "inconclusive" for record in group
        ) / len(group)
        runs = sorted(int(record["runs_used"]) for record in group)
        summaries.append(
            {
                "method": method,
                "case_id": case_id,
                "exact_match_rate": exact,
                "false_accusation_rate": false_accusation,
                "inconclusive_rate": inconclusive,
                "median_runs": median(runs),
                "p90_runs": runs[max(0, math.ceil(0.9 * len(runs)) - 1)],
                "trials": len(group),
                "interactions": next(
                    (
                        record.get("interactions", [])
                        for record in group
                        if record.get("interactions")
                    ),
                    [],
                ),
            }
        )
    return summaries


def render_markdown(
    summaries: list[dict[str, object]], cases: tuple[LabCase, ...]
) -> str:
    """Render benchmark summaries with an honest limitations section."""
    lines = [
        "# Regression Lab benchmark",
        "",
        "The benchmark compares adaptive `mlbisect` with fixed-budget baselines.",
        "",
        "| Method | Case | Exact match | False accusation | Inconclusive | "
        "Median runs | P90 runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {method} | {case_id} | {exact:.0%} | {false:.0%} | {inconclusive:.0%} | "
            "{median_runs:g} | {p90_runs} |".format(
                method=summary["method"],
                case_id=summary["case_id"],
                exact=summary["exact_match_rate"],
                false=summary["false_accusation_rate"],
                inconclusive=summary["inconclusive_rate"],
                median_runs=summary["median_runs"],
                p90_runs=summary["p90_runs"],
            )
        )
    interaction_summaries = [
        summary for summary in summaries if summary.get("interactions")
    ]
    if interaction_summaries:
        lines.extend(["", "## Interaction analysis", ""])
        for summary in interaction_summaries:
            for interaction in summary["interactions"]:
                singles = " + ".join(
                    f"{axis} {drop:g}"
                    for axis, drop in interaction["single_effects"].items()
                )
                lines.append(
                    f"- {summary['method']} {summary['case_id']}: "
                    f"{singles}; joint {interaction['joint_effect']:g}; "
                    f"non-additive {interaction['non_additive']:g}."
                )
    lines.extend(["", "## Where methods did not help", ""])
    interaction_cases = [case.id for case in cases if case.truth.kind == "interaction"]
    if interaction_cases:
        lines.append(
            "- One-at-a-time cannot identify the interaction case(s): "
            + ", ".join(interaction_cases)
            + "."
        )
    lines.append(
        "- Exhaustive search finds interactions but evaluates every configuration, "
        "which grows as 2^k."
    )
    lines.append(
        "- Inconclusive and no-regression cases are included to prevent treating "
        "every metric wobble as a culprit."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results" / "benchmark.jsonl",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).parent / "cases"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "benchmark.md",
    )
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    cases = discover_cases(args.root)
    markdown = render_markdown(score_records(records, cases), cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote report to {args.output}")


def _is_exact(record: dict[str, Any], case: LabCase) -> bool:
    verdict_matches = record["verdict"] == case.truth.expected_verdict
    if case.truth.expected_verdict == "no_regression":
        verdict_matches = record["verdict"] in {
            "no_regression",
            "no_reproducible_regression",
        }
    if not verdict_matches:
        return False
    reported = {
        frozenset(str(axis) for axis in culprit)
        for culprit in record["culprit_sets"]
    }
    return reported == set(case.truth.culprit_sets)


def _is_false_accusation(record: dict[str, Any], case: LabCase) -> bool:
    if case.truth.expected_verdict == "no_regression":
        return bool(record["culprit_sets"])
    if case.truth.expected_verdict == "inconclusive":
        return bool(record["culprit_sets"])
    return any(
        frozenset(str(axis) for axis in culprit)
        not in case.truth.culprit_sets
        for culprit in record["culprit_sets"]
    )


if __name__ == "__main__":
    main()
