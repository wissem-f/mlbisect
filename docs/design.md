# Design

mlbisect treats a regression as a search over configurations. A configuration
is a frozen set of axes assigned their new values; all other axes remain old.
The empty set is the baseline and the full set is the candidate.

## Flow

```text
load and validate manifest
  -> sample baseline
  -> confirm candidate is BAD
  -> ddmin over axes
  -> verify minimal sets and search remaining axes
  -> analyze interactions
  -> write run log and reports
```

## Statistics

Each configuration starts with `min_runs_per_config` samples. Its drop is
computed against the baseline, with metric direction handled explicitly. A
deterministic percentile bootstrap interval is compared with the midpoint
`min_effect / 2`:

- `lo >= midpoint`: `BAD`
- `hi < midpoint`: `OK`
- otherwise: `UNDECIDED`, so another batch is sampled

At the per-configuration or global budget limit, the point estimate is used
and the result is marked `low_confidence`. Fewer than three successful samples
never produce a confident verdict.

## Search

The search uses classic delta debugging (`ddmin`) with deterministic manifest
ordering and memoized configuration tests. It first requires an OK baseline
and a BAD candidate. It then removes chunks of axes while the regression still
reproduces until the result is 1-minimal.

After finding a set, mlbisect tests the remaining axes. If the remainder is
still BAD, another ddmin round searches it, capped at three rounds. If the
budget prevents verification, the result is `inconclusive` and includes ranked
suspects based on the strongest reliable small-configuration evidence.

## Interaction Analysis

For every culprit set with at least two axes, mlbisect reuses the cache to
measure each singleton and the joint set. It reports:

```text
non_additive = joint_drop - sum(single_drops)
```

A positive non-additive drop means the changes are worse together than the sum
of their individual effects. Single-axis culprit sets are labeled
`single_cause`.

## Persistence and Reproducibility

Every executor attempt can be appended to `.mlbisect/runs.jsonl`. The key is
`(configuration, seed)`, so `--resume` reuses completed work after an
interruption. The log also stores a manifest fingerprint; a changed manifest
cannot silently reuse an old log. Historical attempts count toward the hard
global budget, while a cached failed key receives at most one additional retry.
Randomness is controlled by explicit seeds and bootstrap RNG seeds are derived
deterministically from configuration IDs.

## Boundaries

The local executor owns subprocess execution only. It does not schedule
training, provision GPUs, check out git worktrees, or integrate with an
experiment tracker. Those are later integrations. The search and statistics
layers remain pure enough to test with `FakeExecutor`.
