# Saved observability results

Generated from metrics.csv; this summary does not retrain controllers.

- 20 plant/sensor cases; 1095 seed/configuration rows at H=16.
- At history r=h, execution 1 has the lowest seed-mean tail error in 19 of 20 cases.
- 30 seed settings have spectral radius >=1; 21 reach the numerical failure cutoff.
- 9 unstable seed settings do not reach the cutoff within the finite rollout.
- 0 unstable seed settings occur at r>=h.

These are descriptive results for this plant family, not an observability or learning theorem.

| State dimension n | Observability h | Best execution at r=h | One-step tail MSE |
|---:|---:|---:|---:|
| 2 | 2 | 1 | 1.409e-06 |
| 3 | 2 | 1 | 8.315e-06 |
| 3 | 3 | 1 | 9.125e-06 |
| 4 | 2 | 1 | 3.71e-06 |
| 4 | 4 | 1 | 1.042e-05 |
| 5 | 2 | 1 | 3.35e-06 |
| 5 | 4 | 1 | 1.333e-05 |
| 5 | 5 | 1 | 1.269e-05 |
| 10 | 2 | 1 | 2.82e-06 |
| 10 | 4 | 1 | 7.639e-06 |
| 10 | 8 | 1 | 1.646e-05 |
| 10 | 10 | 1 | 2.64e-05 |
| 50 | 2 | 1 | 6.697e-07 |
| 50 | 4 | 1 | 2.16e-06 |
| 50 | 8 | 1 | 4.919e-06 |
| 50 | 50 | 1 | 5.647e-05 |
| 100 | 2 | 1 | 5.081e-07 |
| 100 | 4 | 1 | 1.388e-06 |
| 100 | 8 | 1 | 2.972e-06 |
| 100 | 100 | 2 | 4.282e-05 |

## Example to read first

See `figures/start_here.png`: n=10, h=4, H=16.

| History r | Execution ell | Mean tail MSE | Worst per-step rate | Failed episodes |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.009573 | 0.996658 | 0 |
| 1 | 2 | 0.008365 | 0.995000 | 0 |
| 1 | 4 | 0.006037 | 0.995000 | 0 |
| 1 | 8 | 0.005761 | 0.995000 | 0 |
| 1 | 16 | 0.005767 | 0.995000 | 0 |
| 3 | 1 | inf | 1.321512 | 24 |
| 3 | 2 | 0.0002799 | 0.992291 | 0 |
| 3 | 4 | inf | 1.238468 | 24 |
| 3 | 8 | inf | 1.146944 | 24 |
| 3 | 16 | 0.003609 | 0.992291 | 0 |
| 4 | 1 | 7.639e-06 | 0.992291 | 0 |
| 4 | 2 | 8.794e-06 | 0.992291 | 0 |
| 4 | 4 | 1.401e-05 | 0.992291 | 0 |
| 4 | 8 | 2.311e-05 | 0.992291 | 0 |
| 4 | 16 | 5.547e-05 | 0.992291 | 0 |
| 5 | 1 | 8.027e-06 | 0.992291 | 0 |
| 5 | 2 | 9.928e-06 | 0.992291 | 0 |
| 5 | 4 | 1.473e-05 | 0.992291 | 0 |
| 5 | 8 | 2.446e-05 | 0.992291 | 0 |
| 5 | 16 | 5.701e-05 | 0.992291 | 0 |

## How this differs from pitfalls

This study uses partial observations, varied sensor layouts, and continuing process noise. Its loss is normalized per state coordinate and evaluated over the final window. Pitfalls uses full observations, narrow training states and an initial perturbation without continuing noise. Its main loss averages full-state tracking error over the whole deployment. Absolute scores cannot be compared directly.
