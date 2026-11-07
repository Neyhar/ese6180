# History, training coverage, and action chunking

The question is whether observation history lets a learned controller use
feedback every step instead of committing to an open-loop action chunk.
The answer depends on both state information and corrective training coverage.
The fixed two-dimensional plant is the challenging example from the Pitfalls
paper. Its expert demonstrations can hide the recovery rule even when their
actions are fitted perfectly. See [theory and assumptions](CHUNK_EXECUTION_THEORY.md).

## Start here: the four-condition experiment

```sh
python3 pitfalls/run_four_conditions.py
# Rebuild summaries and figures without retraining:
python3 pitfalls/run_four_conditions.py --summarize-only
```

| Condition | Learner inputs | Prediction / execution |
|---|---|---|
| Chunking with history | 2, 4, 8, or 16 observations and past applied actions | Predict 16; execute 2, 4, 8, or 16 |
| No chunking with history | Same history lengths | Predict and execute 1 |
| Chunking without history | Current observation only | Predict 16; execute 2, 4, 8, or 16 |
| No chunking without history | Current observation only | Predict and execute 1 |

All four conditions use partial sensing, y=x2. “No history” means r=1,
not no observation. Both narrow and recovery training are included. The same
fitted chunk predictor is reused across execution lengths; separate H=1 models
provide the genuine single-action baseline. H=16, execution=1 controls verify
that extra prediction heads do not change the first-action fit.

The saved fresh run uses training seeds 818–820, test seed 20261107, 60 fitted
models, and 180 deployment records (150 requested conditions, 30 controls).
See [protocol, all results, and interpretation](results/four_conditions/README.md),
[narrow-training panels](results/four_conditions/narrow_four_conditions.png),
and [recovery-training panels](results/four_conditions/recovery_four_conditions.png).

Recovery data plus history two favor feedback every step. Narrow history eight
needs chunking for stability; history sixteen is stable without it. Without
history, all tested execution lengths remain unstable. These are findings for
this plant and protocol, not universal claims about chunking or observability.

## Training and evaluation

Narrow training uses 32 noiseless expert trajectories on x1=0. Recovery training
applies known probing actions u=Kx+0.05*Gaussian noise, then labels visited
histories with nominal expert continuations K F^j x. The learner receives its
assigned observations and the actions actually applied; target labels never
replace past-action features. Both use shared linear ridge fitting.

Each seed supplies 80-step training trajectories. Training target times match
across history and prediction length. Tests use 20 held-out initial-state pairs:
x2 is uniform on [-1,1], and x1 is either zero or a signed departure of magnitude
0.0005–0.0015. After a common expert warm-up, the departure is injected and the
learner controls independently for 200 steps, without expert queries or
continuing noise. A separate expert trajectory supplies the scoring reference.

Mean test MSE averages squared full-state tracking distance over post-action
time steps, the 20 perturbed rollouts, and training seeds. Support-state losses
are separate. Exact spectral rates below one indicate asymptotic stability;
low finite-horizon error alone is insufficient. A 1000-step canonical rollout
is a separate diagnostic. Seed ranges are not confidence intervals. The
illustrative success threshold 1e-5 must be fixed before confirmatory work.

## One engine, several saved studies

See the [saved-results index](results/README.md) for snapshot seeds and scope.

| Entry point | Purpose | Saved results |
|---|---|---|
| `run_four_conditions.py` | Primary four-way ablation; fresh seeds 818–820 | `results/four_conditions/` |
| `compare_chunking.py` | Earlier matched commitment comparison; seeds 718–720 | `results/chunking_ablation/` |
| `chunk_sweep.py` | Configurable training/evaluation engine | Output chosen with `--output` |
| `compare_partial.py` | Earlier full/partial × narrow/recovery comparison | `results/partial_observation/` |

The earlier full-state sweep is retained in `results/context_chunking/`.
Earlier plots often use seed medians; their labels and raw CSVs specify the
aggregation. The new comparisons use arithmetic means and retain unstable
losses. Different snapshots are not independent replications and should not be
pooled without matching their protocols.

For a custom run:

```sh
python3 pitfalls/chunk_sweep.py --observation partial --training recovery \
  --contexts 1 2 4 8 16 --horizons 1 16 --executions 1 2 4 8 16 \
  --seeds 818 819 820 --output /tmp/pitfalls-custom
python3 -m unittest discover -s tests -v
```

Use a new output directory to preserve an existing snapshot. Re-running a
comparison intentionally rebuilds its own saved outputs.

## Source map

- `pitfalls.py`: paper matrices, reference gains, narrow demonstrations.
- `chunk_sweep.py`: common experiment orchestration and CLI.
- `recovery_data.py`: exploration histories and counterfactual expert labels.
- `evaluation.py`: paired test states, deployment, tracking loss, pass criterion.
- `stability.py`: exact augmented-history dynamics and model-informed references.
- `plotting.py`: sweep loss/stability figures; `partial_view.py`: actual learner inputs.
- `experiment_io.py`: shared result-schema and matched-comparison checks.
- `../model.py`: shared plant, history policy, ridge fitting, and rollouts.
- `../observability/experiment.py`: shared matched-window/ridge utilities used by recovery training.

Each run retains `config.json`, seed-level `metrics.csv`, individual
`test_rollouts.csv`, and exact `test_states.csv`. Sample trajectories illustrate
one deployment and are not test-set averages. Model-informed observers are
privileged references, not learners fitted from demonstrations.

## Reports and preserved history

[The current four-page L4DC report](../midterm-report/main.tex) combines theory
and simulations; the [short results note](../midterm-report/partial-observation-note.tex)
is separate. The [standalone analysis](analysis/README.md) is a frozen review
of eight earlier active/archived runs and does not include the new two ablations.
LaTeX is its active report source. Prior broad noisy experiments, scalar studies,
and the superseded ReportLab builder are preserved in
`../archive/pitfalls-before-focus/`, outside the active workflow.

## Higher-dimensional study: start with two plots

The focused experiment changes only x1 once at deployment. The learner sees
only x_n. We compare 2-, 4-, 8-, and 16-state members of the same coupled family.
There is no disturbance-type split in the main analysis.

1. [Does history help one-action control?](results/coupled_higher_dimensions/figures/overview_history.png)
2. [With enough history, does executing more actions help?](results/coupled_higher_dimensions/figures/overview_execution.png)

[Read the setup and results](results/coupled_higher_dimensions/README.md).

```sh
python3 pitfalls/high_dimensional.py --plot-only
```

This reuses only the saved x1-change cases: 40 starting conditions, five seeds,
and 200 test steps. Noiseless training stays on the demonstrated line; noisy
training adds action perturbations and expert corrective labels. All testing
is noiseless after the initial x1 change. Score means retain unstable fits.

The broad diagnostic figures and original-paper n=2 reference are preserved
under the exploratory results. To deliberately rebuild those as well:

```sh
python3 pitfalls/high_dimensional.py --plot-only --exploratory --original-reference
```

The mathematics remain in `coupled_pair.py`, numerical work in
`coupled_experiment.py`, and the focused presentation in `focused_analysis.py`.
`coupled_analysis.py` retains the broader diagnostic/audit utilities.
The [construction](COUPLED_HARD_EXAMPLE.md) explains the identical K1/K2
demonstrations, wrong-expert instability, and history requirement n.
