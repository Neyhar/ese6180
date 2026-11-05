# Does observation history reduce the need for action chunking?

The active experiment learns linear history controllers on the paper's narrow
hard-example demonstrations, then tests recovery from small unseen departures.
The hypothesis is that longer history improves control and can reduce the
execution length needed for stable, accurate behavior. Results may support or
reject this hypothesis; no fitting objective enforces the desired trend.

```sh
python3 pitfalls/chunk_sweep.py
python3 -m unittest discover -s tests -p 'test_pitfalls*.py' -v
```

Default sweep: history r=1,2,4,8,16; predicted chunk H=1,2,4,8,16; executed prefix
ell=1,2,4,8,16 with ell<=H; five independent training seeds. ell=1 means feedback
every step, without executing an open-loop multi-action chunk. Observations
are full state C=I; r includes the current observation and r-1 past actions.

```sh
python3 pitfalls/chunk_sweep.py --contexts 1 2 4 8 16 --steps 200 --long-steps 1000
```

Use `--output PATH` to retain a separate run. Other options include `--horizons`,
`--executions`, `--seeds`, `--test-pairs`, `--test-seed`, and `--loss-tolerance`.
The default MSE threshold 1e-5 is illustrative, not prescribed by the paper.
Fix a meaningful threshold before a confirmatory run. Testing a trend on this
set does not establish generalization; use fresh seeds for confirmation.

## Evaluation

Training uses 32 noiseless expert trajectories on x1=0. The held-out test set
contains 20 pairs of initial states: x2 uniform on [-1,1], paired with x1=0 and
x1 of random sign and magnitude 0.0005–0.0015. The same set is shared across all
controllers and generated independently of training. No continuing noise is
added. Expert history is supplied on x1=0, then the perturbation is applied at
deployment so warm-up cannot erase it. This is an explicit initialization
convention, not a learned initialization strategy.

Main test loss averages squared full-state tracking error against K1 over 200
post-action states and then the 20 perturbed states. Support-set loss and worst
perturbed loss are saved separately. A 1000-step canonical rollout is a separate
long-term diagnostic. Exact spectral stability is required when declaring a
setting successful: low finite-horizon error alone is insufficient.

Training target times are matched across both r and H. Static K1, K2, minimax,
minimum-norm, zero-control and model-informed analytic-memory baselines use the same test set.
The memory baseline shows representability, not learning from the demonstrations.

## Outputs: `pitfalls/results/context_chunking/`

- `context_hypothesis.png`: loss versus history, exact stability, and shortest passing execution. A selected length of 1 means no open-loop chunking is needed under the stated criterion; None means no tested setting passes.
- `combined_test_loss.png`: history curves on one execution/loss plot at the largest H.
- `test_loss.png`: separate panels by history, including all prediction horizons.
- `sample_rollouts.png`: canonical [0.001,1] trajectories and baseline comparisons.
- `metrics.csv`: per-trained-controller test means, worst loss, exact stability, rank and long-rollout diagnostic.
- `test_rollouts.csv`, `test_states.csv`: individual losses and exact held-out initial states.
- `execution_candidates.csv`, `context_summary.csv`: pass criteria and shortest feasible execution, requiring all training seeds to pass at the largest H.
- `baselines.csv`, `sample_trajectories.csv`, `config.json`: references and reproduction details.

Shading is the min–max range of seed means, not a confidence interval. All losses
are raw; plot floors are display-only. The enormous wrong-K2 loss and exact-zero
K1 loss remain in CSVs and sample trajectories, but are omitted from comparison
loss plots for readability. Each configuration trains on narrow data; history
can change its dynamics without resolving all information missing in that data.

## Active code

- `pitfalls.py`: paper matrices, static gains, narrow demonstrations, CSV helper; also forwards to the sweep when run directly.
- `chunk_sweep.py`: configurable experiment orchestration.
- `evaluation.py`: independent test states, paired rollouts, tracking loss and hypothesis criterion.
- `stability.py`: exact augmented-history maps, spectral stability and analytic memory witness.
- `plotting.py`: figures for the one research question.
- `../model.py`: shared LinearSystem, HistoryPolicy, ridge fitting and rollout implementation.
- `CHUNK_EXECUTION_THEORY.md`: assumptions and mathematical limits.

Previous broad noisy/partial-observation sweeps, scalar examples, optimization
studies, duplicate analysis scripts and old outputs were moved to
`../archive/pitfalls-before-focus/`. This is a reference snapshot with an archival
README; it is not part of the active workflow.
