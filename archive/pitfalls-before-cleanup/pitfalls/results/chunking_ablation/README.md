# With and without action chunking

Fresh matched simulations use the existing hard-example learner and evaluation pipeline. All conditions use the same 20 perturbed held-out states, 200 scored steps, 15-step expert warm-up, no continuing noise, and training seeds 718–720. Test seed: 20261106. These differ from the earlier exploratory runs. Each seed trains 32 trajectories of 80 steps.

## What is held fixed?

For H=16 the same fitted policy is evaluated with execution lengths 1, 4, 8, and 16. Execution 1 replans every step; larger lengths execute an open-loop prefix. A separate H=1, execution-1 learner verifies the genuinely single-action baseline. Target times match across H and history, and all 18 paired baseline losses agree within numerical tolerance. Both use separable multi-output ridge regression, so extra prediction heads do not change the first-action fit.

## Results

| Training | History | No commitment MSE | Execute 16 MSE | Chunk / feedback | Stable seeds (1 / 16) |
|---|---:|---:|---:|---:|---|
| narrow | 2 | 5.95e+09 | 0.000537 | 9.03e-14 | 0/3 / 0/3 |
| narrow | 8 | 0.000705 | 8.73e-07 | 0.00124 | 0/3 / 3/3 |
| narrow | 16 | 6.18e-07 | 5.51e-07 | 0.89 | 3/3 / 3/3 |
| recovery | 2 | 6.82e-09 | 2.52e-07 | 36.9 | 3/3 / 3/3 |
| recovery | 8 | 2.13e-08 | 2.52e-07 | 11.8 | 3/3 / 3/3 |
| recovery | 16 | 2.83e-08 | 2.52e-07 | 8.91 | 3/3 / 3/3 |

A ratio below one favors chunking; above one favors replanning. These are ratios of seed-mean losses, not means of seed ratios.

## Interpretation

In these runs, narrow history-eight control is unstable without commitment and stable with execution eight or sixteen; execution sixteen lowers mean MSE by about 807 times. Recovery history-two control is stable throughout, but execution sixteen increases MSE by about 37 times. Narrow history-sixteen control is already stable without commitment, with only a modest chunking benefit. Narrow demonstrations omit corrective behavior. Committing to a decaying predicted chunk can suppress destabilizing feedback; longer history can also change the ridge-selected controller. With recovery data and sufficient history to infer hidden state, frequent feedback can correct departures sooner. Read the stability panel alongside loss: a low finite-horizon MSE does not establish stability. This comparison tests these competing mechanisms rather than presuming chunking always helps.

The mean includes every seed; shading is a range, not a confidence interval. Three seeds, a stable two-dimensional plant, small initial departures, expert initialization, and noiseless deployment limit generalization. The reference expert runs separately for scoring; the deployed learner never queries it.

## Reproduce

```sh
python3 pitfalls/compare_chunking.py
# Rebuild only the summary and figure:
python3 pitfalls/compare_chunking.py --summarize-only
```

`summary.csv` aggregates seed means; `paired_comparison.csv` preserves matched comparisons. The narrow/ and recovery/ folders retain configs, raw rollout losses, test states, and long-run diagnostics.
