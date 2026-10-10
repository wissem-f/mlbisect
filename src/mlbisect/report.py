"""Markdown and JSON rendering for completed searches."""

import json
from pathlib import Path
from typing import Any

from mlbisect.manifest import Manifest
from mlbisect.search import SearchResult


def build_report(manifest: Manifest, result: SearchResult) -> dict[str, Any]:
    """Build the JSON-compatible report model shared by both renderers."""
    estimates = [
        _estimate_payload(estimate)
        for estimate in sorted(
            result.estimates,
            key=lambda estimate: (len(estimate.config), tuple(sorted(estimate.config))),
        )
    ]
    by_config = {tuple(item["config"]): item for item in estimates}
    baseline = by_config.get(())
    candidate = by_config.get(tuple(sorted(manifest.axis_names())))
    confirmed_culprits = (
        result.culprit_sets if result.status == "culprit_found" else ()
    )
    tentative_culprits = (
        () if result.status == "culprit_found" else result.culprit_sets
    )
    confidence = "high" if not result.incomplete else "low"
    if result.status == "inconclusive":
        confidence = "low"
    if result.status == "culprit_found" and any(
        len(culprit) >= 2 for culprit in confirmed_culprits
    ):
        kind = "interaction"
    elif confirmed_culprits:
        kind = "single_cause"
    else:
        kind = None

    axis_count = len(manifest.axes)
    max_per_config = manifest.budget.max_runs_per_config
    return {
        "name": manifest.name,
        "status": result.status,
        "result": _result_label(result.status),
        "confidence": confidence,
        "kind": kind,
        "culprit_sets": [sorted(culprit) for culprit in confirmed_culprits],
        "tentative_culprit_sets": [
            sorted(culprit) for culprit in tentative_culprits
        ],
        "metric": {
            "name": manifest.metric.name,
            "direction": manifest.metric.direction,
            "min_effect": manifest.metric.min_effect,
        },
        "incomplete": result.incomplete,
        "baseline": baseline,
        "candidate": candidate,
        "configurations": estimates,
        "runs": {
            "used": result.runs_used,
            "configurations_tested": result.configurations_tested,
            "one_at_a_time": manifest.budget.baseline_runs
            + axis_count * max_per_config,
            "exhaustive": 2**axis_count * max_per_config,
        },
        "ranked_suspects": [
            {
                "axis": suspect.axis,
                "drop": suspect.drop,
                "verdict": suspect.verdict,
            }
            for suspect in result.ranked_suspects
        ],
        "interactions": [
            {
                "culprit_set": sorted(interaction.culprit_set),
                "label": interaction.label,
                "single_effects": [
                    {"axis": axis, "drop": drop}
                    for axis, drop in interaction.single_effects
                ],
                "joint_effect": interaction.joint_effect,
                "non_additive": interaction.non_additive,
            }
            for interaction in result.interactions
        ],
        "warnings": list(result.warnings),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report payload as Markdown."""
    lines = [
        f"# mlbisect report: {report['name']}",
        "",
        f"Metric: {report['metric']['name']} "
        f"({report['metric']['direction']}, "
        f"min effect {report['metric']['min_effect']:g})",
        "",
        f"Result: {report['result']}, confidence {report['confidence']}",
        f"Culprit set: {_culprit_label(report['culprit_sets'])}",
        *(
            [
                "Tentative culprit set: "
                + _culprit_label(report["tentative_culprit_sets"])
            ]
            if report["tentative_culprit_sets"]
            else []
        ),
        "",
        _summary_line(report),
        "",
        "| Configuration | Runs | Mean | Drop | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for estimate in report["configurations"]:
        lines.append(
            "| {config} | {n} | {mean} | {drop} | {verdict} |".format(
                config=estimate["id"],
                n=estimate["n"],
                mean=_number(estimate["mean"]),
                drop=_number(estimate["drop"]),
                verdict=estimate["verdict"],
            )
        )
    for interaction in report["interactions"]:
        singles = " + ".join(
            f"{effect['axis']} {_number(effect['drop'])}"
            for effect in interaction["single_effects"]
        )
        lines.extend(
            [
                "",
                "Interaction: "
                f"{singles}; joint {_number(interaction['joint_effect'])}; "
                f"non-additive {_number(interaction['non_additive'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Runs used: {used} | one-at-a-time baseline: {one} | exhaustive: "
            "{exhaustive}".format(
                used=report["runs"]["used"],
                one=report["runs"]["one_at_a_time"],
                exhaustive=report["runs"]["exhaustive"],
            ),
            "Warnings: " + ("; ".join(report["warnings"]) or "none"),
        ]
    )
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    """Render a report payload as formatted JSON."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_reports(
    manifest: Manifest,
    result: SearchResult,
    markdown_path: str | Path,
    json_path: str | Path,
) -> dict[str, Any]:
    """Build and write both report formats."""
    report = build_report(manifest, result)
    markdown_target = Path(markdown_path)
    json_target = Path(json_path)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.write_text(render_markdown(report), encoding="utf-8")
    json_target.write_text(render_json(report), encoding="utf-8")
    return report


def _estimate_payload(estimate: Any) -> dict[str, Any]:
    axes = sorted(estimate.config)
    return {
        "id": "{" + ",".join(axes) + "}",
        "config": axes,
        "n": estimate.n,
        "mean": estimate.mean,
        "sd": estimate.sd,
        "drop": estimate.drop,
        "ci": list(estimate.ci) if estimate.ci is not None else None,
        "verdict": estimate.verdict,
        "low_confidence": estimate.low_confidence,
        "errors": estimate.errors,
    }


def _result_label(status: str) -> str:
    return {
        "culprit_found": "culprit found",
        "no_reproducible_regression": "no reproducible regression",
        "inconclusive": "inconclusive",
    }.get(status, status)


def _culprit_label(culprit_sets: list[list[str]]) -> str:
    if not culprit_sets:
        return "none"
    return "; ".join(" + ".join(culprit) for culprit in culprit_sets)


def _summary_line(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    baseline_text = _estimate_summary(baseline)
    candidate_text = _estimate_summary(candidate)
    return f"Baseline {baseline_text} | Candidate {candidate_text}"


def _estimate_summary(estimate: dict[str, Any] | None) -> str:
    if estimate is None:
        return "n/a"
    sd = f", sd={_number(estimate['sd'])}" if estimate["sd"] is not None else ""
    return f"{_number(estimate['mean'])} (n={estimate['n']}{sd})"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"
