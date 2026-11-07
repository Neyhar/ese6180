# Saved hard-example studies

For the coupled higher-dimensional extension, start with
[coupled_higher_dimensions/README.md](coupled_higher_dimensions/README.md).
For the original 2D study, start with [four_conditions/README.md](four_conditions/README.md). It contains
the requested history × chunking ablation, fresh seeds, both training regimes,
and raw records. Plots label instability separately from finite-rollout MSE.

| Directory | Question | Training seeds | Evaluation seed | Status |
|---|---|---|---|---|
| `coupled_higher_dimensions/` | Coupled loops; dense histories, cross-dimension analysis | 1118–1122 | 20261110 | Current; n=2/4/8/16 plus separate original n=2 reference |
| `higher_dimensions/` | Independent-copy prototype | 918–920 | 20261108 | Superseded control, h=2 |
| `four_conditions/` | All four history/chunking conditions | 818–820 | 20261107 | Primary current study |
| `chunking_ablation/` | Same predictor with different commitment lengths | 718–720 | 20261106 | Retained independent comparison |
| `partial_observation/` | Full versus partial sensing, narrow versus recovery | 618–620 | 20261105 | Earlier four-suite study |
| `context_chunking/` | Full-state narrow-data history/execution sweep | 618–622 | 20261105 | Earlier full-state study |

Each suite records its exact settings in `config.json`. Result counts include
reused fitted models evaluated at several execution lengths; they are not
independent training replications. New comparisons use mean seed losses;
older graph labels specify median-based displays. Do not pool protocols or
compare scores without checking initialization, time window, and loss definition.

The standalone `../analysis/` PDF is a frozen audit of earlier active and
archived results. It does not yet incorporate the fresh ablations. Older noisy
protocols live in `../../archive/pitfalls-before-focus/results/`.
