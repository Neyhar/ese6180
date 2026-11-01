# ESE 618: context, observability, and action chunking

Study when observation/action history permits reliable imitation of a full-state linear expert, and how executing multiple predicted actions between feedback updates changes error.

## Setup and run

Python 3.10+:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

The default run writes `results/latest/`. To replace that generated run explicitly:

```sh
python run_experiment.py --overwrite
```

An existing nonempty output is rejected unless overwrite is requested. Overwrite is restricted to a generated sweep subfolder inside this repository's results directory. Use a different output folder to keep a deliberate comparison:

```sh
python run_experiment.py --noise 0 --output results/noiseless
```

## Current experiment

| Setting | Default |
|---|---|
| Scored episode length T | 600 actions after warm-up |
| Prediction horizon H | 16 future actions |
| Execution lengths ell | 1, 2, 4, 8, 16 |
| Late evaluation window | Final 200 scored steps |
| State dimensions n | 2, 3, 4, 5, 10, 50, 100 |
| Requested observability indices h | 2, 4, 8, n, skipping infeasible values |
| Contexts r | Unique positive values from 1, h-1, h, h+1 |
| Training seeds | 618, 619, 620 |
| Trajectories per seed | 32 training, 8 validation, 8 evaluation |
| Evaluation process noise | SD 0.02 per state coordinate |

Warm-up is controlled by the expert and is additional to T. It is held fixed across observability cases at a given dimension. Scored metrics exclude it. Training is noiseless, and observation measurements are exact.

If one step were 0.05 seconds, T=600 would represent 30 seconds of scored control, H=16 would predict 0.8 seconds, and execution would span 0.05–0.8 seconds. This is only an interpretation: the synthetic matrices and disturbances have not been calibrated to a physical robot or sampling interval. The final 200-step window describes late behavior, not proven steady state.

For a small complete run:

```sh
python run_experiment.py --dimensions 2 --observability-indices 2 --output results/quick
```

For a deliberate long-duration stress test:

```sh
python run_experiment.py --horizons 1000 --executions 1 2 4 8 16 50 100 200 500 1000 --steps 5000 --tail-steps 1000 --output results/stress
```

## Model and fair comparisons

Plant: x_next = A x + B u + w. Observation: y = C x. Expert: u = K x.
The expert sees the entire state; learners receive histories of observations and past actions. Context r includes the current observation and r-1 past actions. Full-rank observability allows exact state reconstruction only in the noiseless model with known inputs.

Each case compares a separately fitted one-action policy against an H-action linear ridge-regression policy. Training windows, regularization, initial conditions, and evaluation disturbances are matched. At execution ell, the learner commits to ell different predicted actions, records observations throughout, and replans using its latest history. Unused predictions are discarded.

Linear regression outputs are separable: matching inputs makes the first-action weights identical for one-step and chunk training. Prediction horizon alone cannot improve their first-action estimator in this model. Changing execution length changes the feedback behavior. Expert histories also have restricted rank, so good held-out prediction does not identify off-expert behavior. Both facts are diagnostics, not bugs.

The sensor construction varies h while keeping A, B, and K fixed within a state dimension. The exact requested index is checked. A history of sufficient length removes noiseless state ambiguity but does not guarantee robustness under disturbances.

## Results and figures

Root CSVs combine cases. Each `n<N>/h<H>/` folder contains its own config, matrices, time series, and PNG/PDF figures.

- `metrics.csv`: full-episode, late-window, and cumulative tracking error per state coordinate, validation error, and failures, per seed/configuration.
- `episodes.csv`: episode-level metrics and paired differences from one-step control. Negative paired differences favor chunking.
- `fit_diagnostics.csv`: history feature rank and first-action weight agreement.
- `systems.csv`: observability conditioning and plant/expert spectral radii.
- `figures/observability_comparison`: comparison across h at context r=h.
- Per-case figures: late error versus execution length, error over time, and cumulative error. All requested execution lengths above one are shown in time plots alongside the one-step baseline.
- Test figures: `results/tests/figures/mathematical_checks` verifies noiseless reconstruction and exact chunks.

Seed shading is the range of seed means, not a confidence interval. Divergent episodes retain infinite error and failure counts; finite plots cannot show infinity. Inspect failures before comparing averages. Shared disturbances permit paired comparisons.

Historical outputs from before the 600-step configuration were compressed into `archive/` during cleanup. This archive and generated results are ignored by Git. Remove the archive manually when no longer needed. Future runs do not create automatic archives or dated folders; overwrite results/latest explicitly to keep the active output current.

## Files

- `model.py`: plants, observability, reconstruction, expert demonstrations, policy containers, and simple oracle checks.
- `run_experiment.py`: current CLI and observability sweep.
- `long_experiment.py`: shared matched-regression, batched-rollout, and metric helpers (used for short and long episodes).
- `long_plots.py`: current experiment plots.
- `plots.py`: shared figure styling/export and mathematical test plots.
- `tests/`: mathematical and experiment verification.
- `theory-notes/`: retained mathematical notes; not generated results.

The obsolete original runner and its unused plot function have been removed. Core and oracle helpers remain for tests and mathematical comparison.
