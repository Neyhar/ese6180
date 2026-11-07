# Start here: the focused higher-dimensional experiment

## The question

Can remembering observations let the learner choose one action at a time instead of executing an action sequence before checking again?

## The setup

Use one coupled system at each dimension: 2, 4, 8, and 16. The learner sees only the last state coordinate. At deployment change x1 once, leaving all other hidden coordinates zero. The change takes n-1 steps to reach the sensor. The expert sees every state. K1 and K2 agree on demonstrations, but the wrong expert destabilizes the other system.

Noiseless training follows the expert without action changes. Noisy training adds known action perturbations and supplies exact expert actions at the visited states. All testing is noiseless after the initial change. Noise here describes training-data collection, not noisy expert answers or continuing test noise.

Use 40 held-out starting conditions and 5 training seeds. Each learner runs for 200 steps. Score the mean squared distance from an expert rollout. An unstable fit stays in the mean; red crosses identify instability. This focused view uses only the saved x1-change cases, not a mixture of disturbance types.

## Two plots

1. [History versus one-action control](figures/overview_history.png): vary the number of observations and choose one action every step. The dotted line marks enough information to reconstruct the state.
2. [Execution versus error at sufficient history](figures/overview_execution.png): give each n-state system n observations, then vary how many actions it executes before checking again.

Both plots have SVG versions for export. Lines show mean error across training seeds; there is no shading.

## Results with enough history

| States | Observations | Noiseless training, one action | Noisy training, one action | Noisy training, longest execution | Stable noisy one-action fits |
|---:|---:|---:|---:|---:|---|
| 2 | 2 | 5.046e-06 | 1.746e-08 | 1.365e-07 | 5/5 |
| 4 | 4 | 1.135e-06 | 5.616e-08 | 2.991e-07 | 5/5 |
| 8 | 8 | 1.578e-06 | 2.094e-07 | 6.588e-07 | 5/5 |
| 16 | 16 | 4.562e-06 | 8.825e-07 | 1.002e-06 | 5/5 |

## What to conclude

With noisy training and n observations, one-action control is stable for all five seeds at every dimension. Executing sixteen actions gives higher tracking error in all four systems. Before enough history is available, noisy-data fits can be unstable. Noiseless-data fits can become stable without learning the expert correction. Read the first plot for the effect of history and the second for the effect of action execution. Stable control does not by itself mean the learner reconstructed the state or learned the expert correction. Noiseless demonstrations omit corrections; noisy collection supplies them. This is one constructed family, not a universal claim about history or action chunking.

## Reproduce and inspect

```sh
python3 -m pitfalls --plot-only
```

`focused_summary.csv` and `focused_seed_metrics.csv` contain exactly the x1-only scores shown here. The underlying saved experiment contains extra exploratory cases; they are excluded from these two plots. Original Simchowitz n=2 results remain separately available in [original_n2/](original_n2/README.md). They are a reference, not another point mixed into this coupled-family curve. The broader [exploratory results](EXPLORATORY_RESULTS.md) retain the additional plots and disturbance types; they are not needed for this comparison.
