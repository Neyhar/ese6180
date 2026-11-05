# Observability experiment: how much history is enough?

This is the partial-observation companion to `../pitfalls/`.
The expert sees the full state, while the learner sees sensor readings and its
own past actions. We test whether history reaching the observability index h
makes one-step feedback reliable under continuing disturbances.

## Read the existing results first

Open [results/latest/RESULTS.md](results/latest/RESULTS.md) and
[results/latest/figures/start_here.png](results/latest/figures/start_here.png).
These summarize the saved 20-case sweep without retraining. The folders
`n10/h4`, for example, mean state dimension 10 and observability index 4;
**h4 does not mean prediction horizon 4**. The saved prediction horizon is H=16.
`case_summary.csv` has one row per plant/sensor case; `summary.json` has counts.

## Run a focused experiment

From the repository root:

```sh
python3 -m observability.run
```

The default is now the representative n=10,h=4 case, contexts 1,3,4,5, prediction
H=16 and execution 1,2,4,8,16. It uses the original 600-step protocol, 200-step
late window, three training seeds, and eight test episodes per seed.
Use a new `--output` because the existing full sweep is preserved:

```sh
python3 -m observability.run --output observability/results/focused
```

To reproduce the full saved grid:

```sh
python3 -m observability.run --dimensions 2 3 4 5 10 50 100 --observability-indices 2 4 8 n --output observability/results/full-rerun
```

`python3 run_experiment.py` remains a compatibility entry point. Existing
nonempty outputs require explicit `--overwrite`; replacement is restricted to
a generated directory inside `observability/results/`. To summarize an existing
run without training:

```sh
python3 -m observability.summarize --input observability/results/latest
```

## What is learned and tested?

The plant is a stable coupled cyclic system: A=0.995P, B=b, K=-0.05 bᵀP.
Sensor layouts change the exact observability index while A,B,K stay fixed
within each state dimension. This is not the paper's 2D challenging pair.

Each seed supplies 32 noiseless expert training trajectories, eight independent
expert validation trajectories and eight independent noisy deployment episodes.
All policies share training-window IDs, test initial states and process-noise
sequences. Training fits linear ridge policies on observation/action histories.
Test process noise has SD 0.02 per coordinate; observations are exact. Expert
warm-up supplies history and is excluded from scoring. It is an evaluation
convention, not a learned initialization method.

The main score is average squared learner/expert state distance **per coordinate
in the final 200 of 600 deployment steps**. The expert receives the same
process noise. A radius below 1 certifies linear stability; finite tail error
is not a proven stationary noise floor. The cutoff at state/action norm 1e6
records failures as infinite errors without dropping them from averages.

## Compared with pitfalls

| | Observability | Pitfalls |
|---|---|---|
| Plant | Stable cyclic systems, n=2–100 in saved sweep | 2D challenging pair, fixed A1 |
| Observations | Partial; varies h | Full state, h=1 |
| Training restriction | Noiseless expert histories | Expert histories confined to x1=0 |
| Deployment | Random initial state plus continuing noise | Small initial departure, no continuing noise |
| Main loss | Late-window MSE per coordinate | Whole-rollout full-state MSE |
| Main question | Does history supply missing information? | Does history change an underidentified controller's feedback? |

The absolute losses are not directly comparable. Both studies use ridge
behavioral cloning, committed action prefixes, paired expert references, and
exact spectral stability checks. Neither establishes universal monotonicity.
Prediction H and executed prefix ell remain distinct; separable regression
makes prediction-length curves overlap at a fixed ell when windows match.

## Code and outputs

- `run.py`: CLI, sweep plan and case orchestration.
- `experiment.py`: matched regression, batched evaluation, tracking metrics and boundary maps.
- `plotting.py`: detailed per-case plots and full-grid overview.
- `plotting_utils.py`: scientific figure styling/export and optional mathematical-test plots.
- `summarize.py`: readable result summary and a two-panel starting-point graph.
- `../model.py`: shared plant, histories, policy classes and exact reconstruction.

`results/latest/` retains all original CSVs, system matrices and error curves.
Top-level CSVs combine cases; per-case files are useful for drilling down, not
additional experiments. Original config files retain their historical output
paths, while data now reside under this folder. Old test figures and duplicate
notes were archived at `../archive/observability-before-focus/`.
