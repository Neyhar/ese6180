# Four-way history and action-chunking experiment

Partial observation y=x2. No history means r=1: the learner sees the current observation, but no past observations or actions. History means r=2,4,8,16 and includes the intervening applied actions. No chunking uses H=1, execution=1. Chunking uses H=16, execution=2,4,8,16. A separate H=16, execution=1 control matches the single-action baseline within numerical tolerance in all 30 seed/history/training comparisons.

Both narrow and recovery training are run independently. Training seeds 818–820 and test seed 20261107 are fresh relative to previous comparisons. Training uses 32 trajectories of 80 steps per seed; target times match across history and prediction horizon. The same 20 perturbed held-out states are scored for 200 steps, after a common 15-step expert warm-up. There is no continuing noise or deployed expert query. Loss is full-state expert-tracking MSE. An additional 20 on-support cases and 1000-step canonical diagnostics are retained in raw outputs.

There are 60 trained models, 180 deployment records, and 150 records in the four requested conditions. The other 30 records are prediction-versus-execution controls. These are not 180 independently trained learners.

## All tested settings (mean across three seeds)

| Training | Condition | History | Execution | Mean MSE | Stable seeds | Worst rate |
|---|---|---:|---:|---:|---:|---:|
| narrow | 1_chunk_history | 2 | 2 | 5.65e+07 | 0/3 | 1.090646 |
| narrow | 1_chunk_history | 2 | 4 | 4.95e+04 | 0/3 | 1.070747 |
| narrow | 1_chunk_history | 2 | 8 | 6.27 | 0/3 | 1.045471 |
| narrow | 1_chunk_history | 2 | 16 | 0.00056 | 0/3 | 1.018462 |
| narrow | 1_chunk_history | 4 | 2 | 910 | 0/3 | 1.059395 |
| narrow | 1_chunk_history | 4 | 4 | 5.74 | 0/3 | 1.045010 |
| narrow | 1_chunk_history | 4 | 8 | 0.00827 | 0/3 | 1.026083 |
| narrow | 1_chunk_history | 4 | 16 | 1.41e-05 | 0/3 | 1.005482 |
| narrow | 1_chunk_history | 8 | 2 | 0.000172 | 0/3 | 1.014240 |
| narrow | 1_chunk_history | 8 | 4 | 1.9e-05 | 0/3 | 1.006507 |
| narrow | 1_chunk_history | 8 | 8 | 2.32e-06 | 3/3 | 0.995443 |
| narrow | 1_chunk_history | 8 | 16 | 9.38e-07 | 3/3 | 0.982963 |
| narrow | 1_chunk_history | 16 | 2 | 6.5e-07 | 3/3 | 0.969471 |
| narrow | 1_chunk_history | 16 | 4 | 6.29e-07 | 3/3 | 0.966652 |
| narrow | 1_chunk_history | 16 | 8 | 6.08e-07 | 3/3 | 0.962358 |
| narrow | 1_chunk_history | 16 | 16 | 5.92e-07 | 3/3 | 0.957052 |
| narrow | 2_single_history | 2 | 1 | 6.26e+09 | 0/3 | 1.104050 |
| narrow | 2_single_history | 4 | 1 | 2.34e+04 | 0/3 | 1.068572 |
| narrow | 2_single_history | 8 | 1 | 0.000758 | 0/3 | 1.018915 |
| narrow | 2_single_history | 16 | 1 | 6.65e-07 | 3/3 | 0.971100 |
| narrow | 3_chunk_no_history | 1 | 2 | 3.74e+10 | 0/3 | 1.109249 |
| narrow | 3_chunk_no_history | 1 | 4 | 9.88e+06 | 0/3 | 1.085885 |
| narrow | 3_chunk_no_history | 1 | 8 | 275 | 0/3 | 1.056425 |
| narrow | 3_chunk_no_history | 1 | 16 | 0.00519 | 0/3 | 1.025466 |
| narrow | 4_single_no_history | 1 | 1 | 9.06e+12 | 0/3 | 1.124982 |
| recovery | 1_chunk_history | 2 | 2 | 1.69e-08 | 3/3 | 0.750000 |
| recovery | 1_chunk_history | 2 | 4 | 4.21e-08 | 3/3 | 0.750000 |
| recovery | 1_chunk_history | 2 | 8 | 1.11e-07 | 3/3 | 0.750001 |
| recovery | 1_chunk_history | 2 | 16 | 2.71e-07 | 3/3 | 0.750003 |
| recovery | 1_chunk_history | 4 | 2 | 1.92e-08 | 3/3 | 0.750000 |
| recovery | 1_chunk_history | 4 | 4 | 4.21e-08 | 3/3 | 0.750000 |
| recovery | 1_chunk_history | 4 | 8 | 1.11e-07 | 3/3 | 0.750000 |
| recovery | 1_chunk_history | 4 | 16 | 2.71e-07 | 3/3 | 0.750001 |
| recovery | 1_chunk_history | 8 | 2 | 3.04e-08 | 3/3 | 0.750001 |
| recovery | 1_chunk_history | 8 | 4 | 4.99e-08 | 3/3 | 0.750001 |
| recovery | 1_chunk_history | 8 | 8 | 1.11e-07 | 3/3 | 0.750003 |
| recovery | 1_chunk_history | 8 | 16 | 2.71e-07 | 3/3 | 0.750016 |
| recovery | 1_chunk_history | 16 | 2 | 3.86e-08 | 3/3 | 0.750006 |
| recovery | 1_chunk_history | 16 | 4 | 5.91e-08 | 3/3 | 0.750009 |
| recovery | 1_chunk_history | 16 | 8 | 1.18e-07 | 3/3 | 0.750022 |
| recovery | 1_chunk_history | 16 | 16 | 2.71e-07 | 3/3 | 0.750125 |
| recovery | 2_single_history | 2 | 1 | 7.33e-09 | 3/3 | 0.750000 |
| recovery | 2_single_history | 4 | 1 | 1.24e-08 | 3/3 | 0.750000 |
| recovery | 2_single_history | 8 | 1 | 2.29e-08 | 3/3 | 0.750001 |
| recovery | 2_single_history | 16 | 1 | 3.04e-08 | 3/3 | 0.750005 |
| recovery | 3_chunk_no_history | 1 | 2 | 1.6e+13 | 0/3 | 1.113444 |
| recovery | 3_chunk_no_history | 1 | 4 | 1.18e+09 | 0/3 | 1.088141 |
| recovery | 3_chunk_no_history | 1 | 8 | 1.73e+04 | 0/3 | 1.057683 |
| recovery | 3_chunk_no_history | 1 | 16 | 0.253 | 0/3 | 1.026179 |
| recovery | 4_single_no_history | 1 | 1 | 3.52e+16 | 0/3 | 1.133006 |

## Interpretation

In these fresh runs, recovery training with history two and no chunking gives MSE 7.33e-9; executing two actions gives 1.69e-8 and longer chunks increase error further. All history-based recovery settings are stable. Without history, every tested setting remains unstable in both training regimes, including execution sixteen. Narrow training needs execution eight or sixteen at history eight, while history sixteen is stable with single-action execution (MSE 6.65e-7). Compare execution lengths at fixed history to isolate commitment. Compare histories at fixed H and execution to study context. The H=1 versus H=16 execution-one check verifies that the extra independent ridge prediction heads do not change the first-action baseline.

A configuration with any spectral rate at least one is unstable even if its 200-step MSE is small. History can reveal hidden state but does not supply corrective training coverage. Chunking may suppress a bad learned feedback loop, while frequent replanning can help a learner that has learned recovery. Do not choose the minimum loss across this grid and treat it as an independently confirmed optimum.

Three seeds are descriptive, not a confidence guarantee. This is one stable plant, small initial departures, and expert-initialized noiseless histories. History changes the number and scaling of features, so its effect includes ridge regularization and feedback selection, not only observability.

## Reproduce

```sh
python3 pitfalls/run_four_conditions.py
python3 pitfalls/run_four_conditions.py --summarize-only
```

`condition_metrics.csv` gives the four-condition seed-level records; `summary.csv` aggregates them. `narrow/` and `recovery/` retain all raw rollout data, test states, spectral rates, and exact configs.
