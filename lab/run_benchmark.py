"""Run the fixed-budget Regression Lab baselines."""

import argparse
import json
from pathlib import Path
from typing import Callable

from lab.baselines import (
    BaselineResult,
    exhaustive,
    mlbisect,
    one_at_a_time,
    ordered_bisect,
)
from lab.cases import LabCase, discover_cases
from lab.executor import CaseExecutor
from lab.score import render_markdown, score_records

BaselineMethod = Callable[..., BaselineResult]
METHODS: dict[str, BaselineMethod] = {
    "mlbisect": mlbisect,
    "one_at_a_time": one_at_a_time,
    "exhaustive": exhaustive,
    "ordered_bisect": ordered_bisect,
}


def run_benchmark(
    cases: tuple[LabCase, ...],
    trials: int,
    master_seed: int,
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    """Run selected baseline methods and return JSON-ready records."""
    records: list[dict[str, object]] = []
    for trial in range(trials):
        for case_index, case in enumerate(cases):
            manifest = case.load_manifest()
            executor = CaseExecutor(case.id, manifest)
            trial_seed = master_seed + trial * 1_000_000 + case_index * 10_000
            for method_name in methods:
                method = METHODS[method_name]
                result = method(executor, manifest, seed=trial_seed)
                records.append(
                    {
                        "case_id": case.id,
                        "trial": trial,
                        **result.as_dict(),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "cases")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated baseline method names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "benchmark.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).parent / "results" / "benchmark.md",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")

    cases = discover_cases(args.root)
    if not cases:
        parser.error(f"no cases found under {args.root}")
    records = run_benchmark(cases, args.trials, args.master_seed, methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(score_records(records, cases), cases), encoding="utf-8"
    )
    print(f"wrote {len(records)} results to {args.output}")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
