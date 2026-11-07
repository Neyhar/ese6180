> Superseded independent-copy prototype. The current implementation and results use [coupled feedback loops](../coupled_higher_dimensions/README.md). The former source is archived in `archive/pitfalls-before-focus/repeated-pair-prototype/`.

# Higher-dimensional indistinguishable expert pairs

This is a direct-sum extension of the original two-dimensional construction, not a new coupled-system theorem. Dimension grows while the observability index stays two and the number of sensors/actions grows. Coordinates are ordered (hidden1,visible1,hidden2,visible2,...).

## Why the pair is hard

For n=2b, A1,A2,K1,K2 are I_b tensor their original 2D matrices. The demonstration subspace has every hidden coordinate zero. Both experts produce identical actions and identical state trajectories throughout this subspace. Each expert stabilizes its own plant, but K2 destabilizes A1 after a departure in any hidden block. These statements are verified over the entire subspace in pair_checks.csv, not just sampled demonstrations. Matrices for each dimension are saved as pair_n*.npz.

## Simulation

The shared model.py classes, matched-window ridge learner, batched rollout, and exact augmented-history stability routines are reused. Partial sensing observes every visible coordinate. Narrow data keep hidden states zero; recovery data apply known Gaussian action probes and expert continuation labels. Single-action H=1 and H=16 predictions share the same training windows and first-action fit. Execution 1 uses feedback every step, larger executions commit to an action prefix. Config.json records all settings.

Training trajectories scale as 32 per 2D block by default. Test departures have fixed TOTAL Euclidean magnitude across dimensions, tested in one hidden coordinate and distributed across hidden coordinates. Both total and per-coordinate tracking MSE are saved; per-coordinate MSE mechanically falls by n for equal total error, so use total MSE for fixed-total perturbation comparisons. Initial visible coordinates are sampled independently in [-1,1]. All controllers within a dimension share held-out cases. Expert warm-up fills history before an unannounced hidden departure. No continuing test noise or deployed expert queries are used.

## Pair verification

| n | Expert K1 rate | Expert K2 rate | Wrong K2 on A1 rate | Observability index |
|---:|---:|---:|---:|---:|
| 4 | 0.75000 | 0.75000 | 2.09375 | 2 |
| 8 | 0.75000 | 0.75000 | 2.09375 | 2 |
| 16 | 0.75000 | 0.75000 | 2.09375 | 2 |

## Selected learner comparisons

| n | Training | History | MSE execution 1 | MSE execution 16 | Stable seeds (1 / 16) |
|---:|---|---:|---:|---:|---|
| 4 | narrow | 1 | 9.21e+12 | 0.0054 | 0/3 / 0/3 |
| 4 | narrow | 2 | 6.16e+09 | 0.000555 | 0/3 / 0/3 |
| 4 | narrow | 8 | 0.000731 | 9.04e-07 | 0/3 / 3/3 |
| 4 | narrow | 16 | 6.41e-07 | 5.7e-07 | 3/3 / 3/3 |
| 4 | recovery | 1 | 1.49e+17 | 1.62 | 0/3 / 0/3 |
| 4 | recovery | 2 | 7.07e-09 | 2.61e-07 | 3/3 / 3/3 |
| 4 | recovery | 8 | 2.21e-08 | 2.61e-07 | 3/3 / 3/3 |
| 4 | recovery | 16 | 2.93e-08 | 2.61e-07 | 3/3 / 3/3 |
| 8 | narrow | 1 | 7.35e+12 | 0.00422 | 0/3 / 0/3 |
| 8 | narrow | 2 | 5e+09 | 0.000449 | 0/3 / 0/3 |
| 8 | narrow | 8 | 0.0006 | 7.43e-07 | 0/3 / 3/3 |
| 8 | narrow | 16 | 5.27e-07 | 4.69e-07 | 3/3 / 3/3 |
| 8 | recovery | 1 | 1.08e+17 | 1.98 | 0/3 / 0/3 |
| 8 | recovery | 2 | 5.81e-09 | 2.15e-07 | 3/3 / 3/3 |
| 8 | recovery | 8 | 1.81e-08 | 2.15e-07 | 3/3 / 3/3 |
| 8 | recovery | 16 | 2.41e-08 | 2.15e-07 | 3/3 / 3/3 |
| 16 | narrow | 1 | 9.89e+12 | 0.00571 | 0/3 / 0/3 |
| 16 | narrow | 2 | 6.68e+09 | 0.0006 | 0/3 / 0/3 |
| 16 | narrow | 8 | 0.000797 | 9.86e-07 | 0/3 / 3/3 |
| 16 | narrow | 16 | 6.99e-07 | 6.23e-07 | 3/3 / 3/3 |
| 16 | recovery | 1 | 4.76e+17 | 3.78 | 0/3 / 0/3 |
| 16 | recovery | 2 | 7.71e-09 | 2.85e-07 | 3/3 / 3/3 |
| 16 | recovery | 8 | 2.41e-08 | 2.85e-07 | 3/3 / 3/3 |
| 16 | recovery | 16 | 3.2e-08 | 2.85e-07 | 3/3 / 3/3 |

## Findings in this run

Across dimensions 4, 8, and 16, recovery training with history two gives stable
one-step control, with total MSE 7.07e-9, 5.81e-9, and 7.71e-9 respectively.
Narrow history-eight learners are unstable at execution one but stable at
eight and sixteen. Narrow history-sixteen learners are stable at execution one.
All no-history learners remain unstable over the tested execution grid.
These patterns reproduce the original mechanism; similar errors do not show
that higher dimension is easier, because history requirements stay fixed and
training data increase with dimension.

## Reproduce

```sh
# Historical source: archive/pitfalls-before-focus/repeated-pair-prototype/high_dimensional.py
```

Each metrics row is one policy deployment, not an independent fit. Raw episode losses and reference baselines are retained. This does not test growing observability index, unknown dynamics identification, fixed-sensor scaling, or coupling between blocks. Stability and finite-horizon tracking are separate criteria; three seeds are descriptive.
