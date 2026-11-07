# Pitfalls: history versus action chunking

One question: **does remembering observations let the learner act one step at a
time instead of executing an action sequence before checking again?**

## Start here

1. [Does history improve one-action control?](results/coupled_higher_dimensions/figures/overview_history.png)
2. [With enough history, does executing more actions help?](results/coupled_higher_dimensions/figures/overview_execution.png)
3. [Experiment, data, and results](results/coupled_higher_dimensions/README.md)

The main analysis uses one coupled system at each dimension: 2, 4, 8, and 16.
The learner observes only the last state. We change x1 once at deployment and
measure its recovery. Plot lines are mean errors across five training seeds,
without shading. Red crosses indicate at least one unstable fit.

Noiseless training stays on the demonstrated line. Noisy training adds known
action perturbations and supplies exact expert answers at the resulting states.
Every test is noiseless after the initial x1 change.

## Commands

```sh
# Audit existing data and rebuild only the two main plots:
python3 -m pitfalls --plot-only

# Run the current experiment into a new directory:
python3 -m pitfalls --output /tmp/pitfalls-new-run

# Run the paper's original 2D pair with the same engine:
python3 -m pitfalls --family original --dimensions 2 --output /tmp/pitfalls-original

# Optional: rebuild the full exploratory diagnostics and original comparison:
python3 -m pitfalls --plot-only --exploratory --original-reference

python3 -m unittest discover -s tests -v
```

`python3 pitfalls/high_dimensional.py` is retained as an equivalent entry point.
A saved run is overwritten only with an explicit `--overwrite` flag.

## Code map

| Location | Responsibility |
|---|---|
| `coupled/system.py` | Coupled matrices, indistinguishable experts, stability and observability checks |
| `coupled/experiment.py` | Settings, training, deployment, and raw results; supports both system families |
| `coupled/plots.py` | The two main mean-error plots and their x1-only summary |
| `coupled/diagnostics.py` | Optional exploratory plots and comparisons |
| `original/system.py` | The original Simchowitz two-state construction |
| `original/data.py`, `original/evaluation.py` | Original-example data and mathematical verification helpers |
| `shared/audit.py`, `shared/results.py` | Complete-grid, seed, episode, and timing checks; CSV handling |
| `shared/stability.py`, `shared/plotting.py` | Exact feedback dynamics and common figure export |
| `../model.py`, `../observability/experiment.py` | Shared plant, policy, ridge fitting, and rollout primitives |

[Coupled construction and proof](docs/coupled-hard-example.md) ·
[Theory and assumptions](docs/theory.md)

## Results and history

`results/coupled_higher_dimensions/` is the only active result collection. It
retains raw episodes, settings, matrix checks, source snapshots, and optional
diagnostics. The primary scores use only x1-change cases; additional saved
perturbation types are exploratory. The paper's original n=2 system is a
separate reference under `original_n2/`, not a pooled dimension-scaling point.

Earlier runners, superseded result sets, and the frozen comprehensive analysis
are preserved in [the pre-cleanup archive](../archive/pitfalls-before-cleanup/README.md).
They are outside the active workflow. No saved measurements were discarded.

Report sources remain in [midterm-report/](../midterm-report/README.md).
