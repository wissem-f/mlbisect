# Regression Lab benchmark

The benchmark compares adaptive `mlbisect` with fixed-budget baselines.

| Method | Case | Exact match | False accusation | Inconclusive | Median runs | P90 runs |
|---|---|---:|---:|---:|---:|---:|
| exhaustive | L01_data_skew | 100% | 0% | 0% | 40 | 40 |
| exhaustive | L02_preproc_bump | 100% | 0% | 0% | 40 | 40 |
| exhaustive | L03_interaction | 100% | 0% | 0% | 40 | 40 |
| exhaustive | L04_small_effect | 0% | 20% | 0% | 40 | 40 |
| exhaustive | L05_decoy_axes | 100% | 0% | 0% | 640 | 640 |
| exhaustive | L06_no_regression | 100% | 0% | 0% | 40 | 40 |
| mlbisect | L01_data_skew | 100% | 0% | 0% | 14 | 14 |
| mlbisect | L02_preproc_bump | 100% | 0% | 0% | 14 | 14 |
| mlbisect | L03_interaction | 100% | 0% | 0% | 14 | 14 |
| mlbisect | L04_small_effect | 100% | 0% | 100% | 18 | 35 |
| mlbisect | L05_decoy_axes | 100% | 0% | 0% | 20 | 20 |
| mlbisect | L06_no_regression | 100% | 0% | 0% | 8 | 8 |
| one_at_a_time | L01_data_skew | 100% | 0% | 0% | 30 | 30 |
| one_at_a_time | L02_preproc_bump | 100% | 0% | 0% | 30 | 30 |
| one_at_a_time | L03_interaction | 0% | 0% | 0% | 30 | 30 |
| one_at_a_time | L04_small_effect | 0% | 20% | 0% | 30 | 30 |
| one_at_a_time | L05_decoy_axes | 100% | 0% | 0% | 70 | 70 |
| one_at_a_time | L06_no_regression | 100% | 0% | 0% | 30 | 30 |
| ordered_bisect | L01_data_skew | 100% | 0% | 0% | 15 | 15 |
| ordered_bisect | L02_preproc_bump | 100% | 0% | 0% | 15 | 15 |
| ordered_bisect | L03_interaction | 0% | 100% | 0% | 15 | 15 |
| ordered_bisect | L04_small_effect | 0% | 30% | 0% | 10 | 15 |
| ordered_bisect | L05_decoy_axes | 100% | 0% | 0% | 25 | 25 |
| ordered_bisect | L06_no_regression | 100% | 0% | 0% | 10 | 10 |

## Interaction analysis

- mlbisect L03_interaction: data -0.004375 + deps 0.005; joint 0.431042; non-additive 0.430417.

## Where methods did not help

- One-at-a-time cannot identify the interaction case(s): L03_interaction.
- Exhaustive search finds interactions but evaluates every configuration, which grows as 2^k.
- Inconclusive and no-regression cases are included to prevent treating every metric wobble as a culprit.
