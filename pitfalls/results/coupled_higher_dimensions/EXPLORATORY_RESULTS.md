# Coupled hard-example results

Each system has one coupled n-state feedback loop and ONE scalar sensor. This is not a collection of independent 2D copies. K1 and K2 produce identical demonstrations, but each destabilizes the other plant. The history required for exact action recovery is n.

## Start with the plots

- [Construction, observation ranks, and wrong-expert failure](figures/pair_diagnostics.png)
- [One-step error versus absolute and relative history](figures/cross_dimension/one_step_history.png)
- [Cross-dimension error with enough history, including observer ratios](figures/cross_dimension/observable_context.png)
- [Paired effect of longer execution](figures/cross_dimension/chunking_effect.png)
- [Farthest versus distributed hidden departures](figures/cross_dimension/perturbation_families.png)
- [Matched short/long test-horizon check](figures/cross_dimension/test_horizon_check.png)

Every figure also has an SVG for clean export. PNGs are previews. Shading is the range of independent training-seed means, not a confidence interval. If zero or infinite means occur, labeled boundary markers replace unplottable logarithmic points; raw scores are never floored or clipped. Red crosses flag comparisons involving unstable controllers; a smaller finite-rollout MSE is not a stability guarantee.

## Per-dimension results

| Dimension | History / stability | Complete execution grid | Time-resolved test error |
|---:|---|---|---|
| 2 | [profiles](figures/n2/history_and_stability.png) | [heatmaps](figures/n2/grid.png) | [rollouts](figures/n2/rollouts.png) |
| 4 | [profiles](figures/n4/history_and_stability.png) | [heatmaps](figures/n4/grid.png) | [rollouts](figures/n4/rollouts.png) |
| 8 | [profiles](figures/n8/history_and_stability.png) | [heatmaps](figures/n8/grid.png) | [rollouts](figures/n8/rollouts.png) |
| 16 | [profiles](figures/n16/history_and_stability.png) | [heatmaps](figures/n16/grid.png) | [rollouts](figures/n16/rollouts.png) |

## What was run

5 independent training seeds ([1118, 1119, 1120, 1121, 1122]); dimensions [2, 4, 8, 16]; 128 training trajectories of 100 steps per seed, training regime, and dimension. Noiseless training trajectories stay on the demonstrated line. Noisy training data use known Gaussian action perturbations and exact expert continuation labels. Both retain the actual applied actions as history inputs.

Histories cover every length from 1 through n+1, plus common absolute lengths [2, 4, 8, 16]. Execution horizons are [1, 2, 4, 8, 16], prediction horizons are 1 and 16. At fixed history the same fitted multi-action predictor is reused for every execution length. Single-action H=1 and multi-action H>1 executed one step at a time are paired training controls.

Each policy is tested on 40 shared visible amplitudes, each with an on-support, farthest-hidden, and distributed-hidden initial state. The 80 perturbed cases have fixed TOTAL hidden Euclidean norm in [.0005,.0015]. Scores average full-state squared distance over 200 post-action steps and perturbed cases, then training seeds. Total MSE is primary; per-coordinate MSE divides the same error by n and is not the primary scaling comparison.

Expert history initialization lasts 16 steps before a hidden jump is injected. There are no continuing disturbances, observation noise, or deployed expert queries. Training target times and test perturbation amplitudes are matched across dimensions. Farthest-hidden departures act on x1; distributed departures allocate the same total norm over all hidden coordinates.

## One action at the reconstruction threshold

| n | Data | History | Mean total MSE | Seed range | Stable seeds | Observer MSE | Ratio to observer |
|---:|---|---:|---:|---|---|---:|---:|
| 2 | Noiseless training data | 2 | 5.045e-06 | 5.045e-06–5.045e-06 | 5/5 | 1.746e-08 | 289 |
| 2 | Noisy training data | 2 | 1.746e-08 | 1.746e-08–1.746e-08 | 5/5 | 1.746e-08 | 1 |
| 4 | Noiseless training data | 4 | 7.723e-07 | 7.723e-07–7.723e-07 | 5/5 | 4.507e-08 | 17.1 |
| 4 | Noisy training data | 4 | 4.507e-08 | 4.507e-08–4.507e-08 | 5/5 | 4.507e-08 | 1 |
| 8 | Noiseless training data | 8 | 9.252e-07 | 9.252e-07–9.252e-07 | 5/5 | 1.485e-07 | 6.23 |
| 8 | Noisy training data | 8 | 1.485e-07 | 1.485e-07–1.485e-07 | 5/5 | 1.485e-07 | 1 |
| 16 | Noiseless training data | 16 | 2.462e-06 | 2.462e-06–2.462e-06 | 5/5 | 5.47e-07 | 4.5 |
| 16 | Noisy training data | 16 | 5.47e-07 | 5.469e-07–5.473e-07 | 5/5 | 5.47e-07 | 1 |

## One observation below the threshold

| n | Data | History | One-step MSE | Stable seeds | Longest-execution MSE | Stable seeds |
|---:|---|---:|---:|---|---:|---|
| 2 | Noiseless training data | 1 | 0.005435 | 0/5 | 3.45e-07 | 5/5 |
| 2 | Noisy training data | 1 | 1994 | 0/5 | 2.119e-06 | 5/5 |
| 4 | Noiseless training data | 3 | 1.243e-06 | 5/5 | 4.399e-07 | 5/5 |
| 4 | Noisy training data | 3 | 0.2743 | 0/5 | 9.637e-06 | 5/5 |
| 8 | Noiseless training data | 7 | 9.641e-07 | 5/5 | 8.641e-07 | 5/5 |
| 8 | Noisy training data | 7 | 26.04 | 0/5 | 0.0001994 | 5/5 |
| 16 | Noiseless training data | 15 | 2.465e-06 | 5/5 | 2.456e-06 | 5/5 |
| 16 | Noisy training data | 15 | 14.22 | 0/5 | 0.2939 | 0/5 |

## Main observations in this run

At history h=n, one-step controllers trained on noisy data are stable in 20/20 seed/dimension cases. At h-1, that count is 0/20. Their exact loss and seed ranges are listed above; the denser history plot shows the full transition, including non-monotone failures below h.

- n=2: history-h one-step MSE is 1.746e-08, 1 times the observer reference. Executing 16 actions gives 1.365e-07 (7.817 times the one-step seed-mean loss).
- n=4: history-h one-step MSE is 4.507e-08, 1 times the observer reference. Executing 16 actions gives 2.054e-07 (4.557 times the one-step seed-mean loss).
- n=8: history-h one-step MSE is 1.485e-07, 1 times the observer reference. Executing 16 actions gives 3.903e-07 (2.628 times the one-step seed-mean loss).
- n=16: history-h one-step MSE is 5.47e-07, 1 times the observer reference. Executing 16 actions gives 5.536e-07 (1.012 times the one-step seed-mean loss).

## Analysis and limits

The comparison at r=h separates representability from identification: enough history permits exact expert recovery from known equations, but noiseless training does not identify the missing corrections. Compare the two training-data regimes at identical history and execution, not different selected optima.

The observer provides a reference for the initial information delay. Its tracking loss can grow with n even though its eventual spectral rate remains .75. Ratios to this reference help distinguish that unavoidable observation delay from additional learned-controller error. The observer is not claimed optimal.

Longer execution may stabilize an otherwise bad feedback rule, or may postpone a useful correction. The paired-ratio plot fixes the fitted predictor and shows the mean and range of within-seed ratios; it is NOT the ratio of two separately averaged losses. Consult stability counts before interpreting a favorable finite-horizon ratio. All unstable seeds remain in the means.

More raw context is not guaranteed to improve loss or stability. Coupled-family reconstruction O_n has condition number one, but raw history features and ridge-selected feedback can still be ill-conditioned or underidentified. The original pair has a different reconstruction condition, recorded in pair_checks.csv. Training data counts are fixed across n. Changing n changes information delay, unexcited directions, and policy parameter count simultaneously; this is not dimension-only scaling. Noiseless training target amplitudes differ from test amplitudes because targets start after a common history window.

## Audit

- learned_deployments: 2400
- reference_deployments: 24
- raw_episode_records: 290880
- perturbed_cases_per_deployment: 80
- independent_training_seeds: 5
- prefix_checks: 400
- temporal_profile_checks: 522
- maximum_relative_aggregation_gap: 4.335990744220889e-16
- matched_initial_states: True
- complete_grid: True
- matched_horizon_checks: 150

Each metric row is a deployment, not an independent fit. The first-action control is deliberately redundant. Raw test episodes, seed-level losses, per-step diagnostic profiles, matrices, full settings, exact Schur checks, and source hashes are retained. CSV codes narrow/recovery remain as stable internal identifiers for noiseless/noisy training respectively.

## Files and reproduction

`metrics.csv`: learned seed-level deployments. `summary.csv`: all grid cells. `paired_chunking.csv`: within-seed effects. `test_rollouts.csv`: all individual test cases. `temporal_profiles.csv`: selected test-case mean time profiles. `sample_trajectories.csv`: one matched farthest-direction example per diagnostic deployment. `paired_test_horizons.csv` compares short and long means on that same initial condition. `baselines.csv`, `pair_checks.csv`, `observability_ranks.csv`, `test_states.csv`, `audit.json`, and `source_manifest.json` document references and checks. `source_snapshot/` preserves the exact original run code; `analysis_manifest.json` records the current figure code and input hashes.

```sh
# Audit and regenerate figures from these saved results:
python3 -m pitfalls --plot-only --exploratory
# Run into a separate directory:
python3 pitfalls/high_dimensional.py --output /tmp/coupled-new-run
```

Use `--overwrite` only to intentionally replace a saved run. The mathematical construction is explained in `../../docs/coupled-hard-example.md`. The previous three-seed snapshot is preserved separately; do not pool it with these fresh-seed results.

For n=2, no history and one observation below h are the same setting; the two perturbation families also occupy the same hidden axis. They are not independent spatial directions.

## Original Simchowitz n=2 comparison

The n=2 point in the dimension-scaling plots uses the coupled family, so all four points share one construction. The paper's original n=2 matrices are different and are kept as a separately labeled matched reference. They use exactly the same test states, five training seeds, history grid, trajectory counts, target times, and warm-up.

[Original versus coupled n=2 graph](figures/cross_dimension/original_vs_coupled_n2.png) · [Original n=2 alongside all dimensions](figures/cross_dimension/original_reference_scaling.png) · [Original n=2 raw results and plots](original_n2/README.md)

| Family | Training | History | Prediction | Execution | Mean MSE | Stable seeds |
|---|---|---:|---:|---:|---:|---|
| Coupled n=2 | Noiseless training data | 2 | 1 | 1 | 5.045e-06 | 5/5 |
| Coupled n=2 | Noiseless training data | 2 | 16 | 16 | 3.132e-07 | 5/5 |
| Coupled n=2 | Noisy training data | 2 | 1 | 1 | 1.746e-08 | 5/5 |
| Coupled n=2 | Noisy training data | 2 | 16 | 16 | 1.365e-07 | 5/5 |
| Original n=2 | Noiseless training data | 2 | 1 | 1 | 6.091e+09 | 0/5 |
| Original n=2 | Noiseless training data | 2 | 16 | 16 | 0.0005464 | 0/5 |
| Original n=2 | Noisy training data | 2 | 1 | 1 | 7.066e-09 | 5/5 |
| Original n=2 | Noisy training data | 2 | 16 | 16 | 2.61e-07 | 5/5 |
