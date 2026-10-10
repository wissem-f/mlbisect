"""Delta-debugging search over noisy configuration estimates."""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from mlbisect.config import Configuration
from mlbisect.executors.base import Executor
from mlbisect.manifest import Manifest
from mlbisect.sampler import AdaptiveSampler, ConfigEstimate

SearchStatus = Literal[
    "culprit_found", "no_reproducible_regression", "inconclusive"
]


@dataclass(frozen=True)
class Suspect:
    """A ranked axis when the search could not finish conclusively."""

    axis: str
    drop: float | None
    verdict: str


@dataclass(frozen=True)
class InteractionAnalysis:
    """Effect decomposition for a culprit set with multiple axes."""

    culprit_set: frozenset[str]
    label: Literal["interaction"]
    single_effects: tuple[tuple[str, float | None], ...]
    joint_effect: float | None
    non_additive: float | None


@dataclass(frozen=True)
class SearchResult:
    """Search outcome and evidence collected along the way."""

    status: SearchStatus
    culprit_sets: tuple[frozenset[str], ...]
    incomplete: bool
    runs_used: int
    configurations_tested: int
    estimates: tuple[ConfigEstimate, ...]
    ranked_suspects: tuple[Suspect, ...] = ()
    warnings: tuple[str, ...] = ()
    interactions: tuple[InteractionAnalysis, ...] = ()

    @property
    def culprit_set(self) -> frozenset[str] | None:
        """Return the first culprit set for callers handling single causes."""
        return self.culprit_sets[0] if self.culprit_sets else None


def find_culprits(
    executor: Executor,
    manifest: Manifest,
    sampler: AdaptiveSampler | None = None,
    max_rounds: int = 3,
) -> SearchResult:
    """Find minimal bad sets using memoized ddmin and verification rounds."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    active_sampler = sampler or AdaptiveSampler(executor, manifest)
    estimates: dict[frozenset[str], ConfigEstimate] = {}
    undecided_seen = False
    low_confidence_seen = False
    warnings: list[str] = []

    def evaluate(config: frozenset[str]) -> ConfigEstimate:
        nonlocal low_confidence_seen, undecided_seen
        if config not in estimates:
            estimate = active_sampler.estimate(Configuration(config))
            estimates[config] = estimate
            if estimate.verdict == "UNDECIDED":
                undecided_seen = True
            if estimate.low_confidence:
                low_confidence_seen = True
        return estimates[config]

    baseline = evaluate(frozenset())
    if baseline.verdict != "OK":
        if baseline.verdict == "UNDECIDED":
            warnings.append("baseline could not be decided within the run budget")
            return _result(
                "inconclusive",
                (),
                True,
                active_sampler,
                estimates,
                manifest,
                warnings,
            )
        warnings.append("baseline is already classified as BAD")
        return _result(
            "no_reproducible_regression",
            (),
            False,
            active_sampler,
            estimates,
            manifest,
            warnings,
        )

    all_axes = frozenset(manifest.axis_names())
    candidate = evaluate(all_axes)
    if candidate.verdict != "BAD":
        if candidate.verdict == "UNDECIDED" or candidate.low_confidence:
            warnings.append("candidate could not be confirmed as BAD")
            return _result(
                "inconclusive",
                (),
                True,
                active_sampler,
                estimates,
                manifest,
                warnings,
            )
        warnings.append("candidate is not classified as BAD")
        return _result(
            "no_reproducible_regression",
            (),
            False,
            active_sampler,
            estimates,
            manifest,
            warnings,
        )

    if candidate.low_confidence:
        warnings.append("candidate BAD verdict has low confidence")

    culprit_sets: list[frozenset[str]] = []
    remaining = all_axes
    round_limit_hit = False
    for round_number in range(max_rounds):
        if not remaining:
            break
        ordered_axes = [axis for axis in manifest.axis_names() if axis in remaining]

        def is_bad(config: frozenset[str]) -> bool:
            return evaluate(config).verdict == "BAD"

        minimal = ddmin(ordered_axes, is_bad)
        minimal_estimate = evaluate(minimal)
        if minimal_estimate.verdict != "BAD":
            undecided_seen = True
            warnings.append("search could not verify a minimal culprit set")
            break
        culprit_sets.append(minimal)

        complement = remaining - minimal
        if not complement:
            break
        complement_estimate = evaluate(complement)
        if complement_estimate.verdict == "BAD":
            remaining = complement
            if round_number + 1 == max_rounds:
                warnings.append("maximum culprit search rounds reached")
                round_limit_hit = True
            continue
        if complement_estimate.verdict == "UNDECIDED":
            undecided_seen = True
            warnings.append("culprit complement could not be verified as OK")
        break

    interactions = _analyze_interactions(culprit_sets, manifest, evaluate)
    incomplete = (
        undecided_seen
        or low_confidence_seen
        or active_sampler.budget_exhausted
        or round_limit_hit
    )
    status: SearchStatus = "inconclusive" if incomplete else "culprit_found"
    return _result(
        status,
        tuple(culprit_sets),
        incomplete,
        active_sampler,
        estimates,
        manifest,
        warnings,
        tuple(interactions),
    )


def ddmin(
    items: Sequence[str], test: Callable[[frozenset[str]], bool]
) -> frozenset[str]:
    """Return a deterministic 1-minimal subset satisfying ``test``."""
    current = list(dict.fromkeys(items))
    memo: dict[frozenset[str], bool] = {}

    def check(config: frozenset[str]) -> bool:
        if config not in memo:
            memo[config] = test(config)
        return memo[config]

    if not current or not check(frozenset(current)):
        return frozenset()

    granularity = 2
    while len(current) >= 2:
        subsets = _split(current, granularity)
        reduced = False
        for subset in subsets:
            complement = frozenset(current) - frozenset(subset)
            if check(complement):
                current = [item for item in current if item not in subset]
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue

        for subset in subsets:
            candidate = frozenset(subset)
            if check(candidate):
                current = list(subset)
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return frozenset(current)


def _split(items: Sequence[str], parts: int) -> list[list[str]]:
    count = min(max(parts, 1), len(items))
    quotient, remainder = divmod(len(items), count)
    subsets: list[list[str]] = []
    start = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        subsets.append(list(items[start : start + size]))
        start += size
    return subsets


def _result(
    status: SearchStatus,
    culprit_sets: tuple[frozenset[str], ...],
    incomplete: bool,
    sampler: AdaptiveSampler,
    estimates: dict[frozenset[str], ConfigEstimate],
    manifest: Manifest,
    warnings: list[str],
    interactions: tuple[InteractionAnalysis, ...] = (),
) -> SearchResult:
    ranked = _rank_suspects(estimates, manifest)
    return SearchResult(
        status=status,
        culprit_sets=culprit_sets,
        incomplete=incomplete,
        runs_used=sampler.runs_used,
        configurations_tested=len(estimates),
        estimates=tuple(estimates.values()),
        ranked_suspects=ranked,
        warnings=tuple(warnings),
        interactions=interactions,
    )


def _analyze_interactions(
    culprit_sets: Sequence[frozenset[str]],
    manifest: Manifest,
    evaluate: Callable[[frozenset[str]], ConfigEstimate],
) -> list[InteractionAnalysis]:
    analyses: list[InteractionAnalysis] = []
    for culprit_set in culprit_sets:
        if len(culprit_set) < 2:
            continue
        single_effects: list[tuple[str, float | None]] = []
        for axis in manifest.axis_names():
            if axis in culprit_set:
                single_effects.append((axis, evaluate(frozenset({axis})).drop))
        joint_effect = evaluate(culprit_set).drop
        values = [effect for _, effect in single_effects]
        numeric_values = [effect for effect in values if effect is not None]
        non_additive = (
            joint_effect - sum(numeric_values)
            if joint_effect is not None and len(numeric_values) == len(values)
            else None
        )
        analyses.append(
            InteractionAnalysis(
                culprit_set=culprit_set,
                label="interaction",
                single_effects=tuple(single_effects),
                joint_effect=joint_effect,
                non_additive=non_additive,
            )
        )
    return analyses


def _rank_suspects(
    estimates: dict[frozenset[str], ConfigEstimate], manifest: Manifest
) -> tuple[Suspect, ...]:
    ranked: list[tuple[float, Suspect]] = []
    for axis in manifest.axis_names():
        evidence = [
            estimate
            for config, estimate in estimates.items()
            if axis in config
            and estimate.drop is not None
            and estimate.n >= 3
            and estimate.verdict != "UNDECIDED"
        ]
        best = (
            max(
                evidence,
                key=lambda estimate: (
                    (estimate.drop or float("-inf")) / len(estimate.config),
                    estimate.drop or float("-inf"),
                    -len(estimate.config),
                ),
            )
            if evidence
            else None
        )
        suspect = Suspect(
            axis=axis,
            drop=best.drop if best is not None else None,
            verdict=best.verdict if best is not None else "UNDECIDED",
        )
        quality = (
            (best.drop or float("-inf")) / len(best.config)
            if best is not None
            else float("-inf")
        )
        ranked.append((quality, suspect))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(suspect for _, suspect in ranked)
