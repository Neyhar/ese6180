# Partial-observation hard example

All four runs share plant A1, evaluation initial states, three training seeds, contexts 1,2,4,8,16, H=1,4,16 and execution lengths up to H.
There are 135 learned settings per run (540 total), each evaluated on 20 support and 20 perturbed 200-step rollouts, with no continuing noise.
A separate 1000-step canonical rollout is a diagnostic. No test states enter training.

## What does the learner see?

Full observation gives [x1,x2]; partial observation gives only y=x2. Both receive actual past actions.
For partial observation, O2=[[0,1],[-0.1875,0.75]] has full rank: the observability index is two.
An unannounced jump in x1 at deployment does not change any available y or past action immediately. Its effect appears in y one step later.
Thus a partial learner cannot match the full-state expert on its first action for both paired initial states. After the next transition, known-model reconstruction is exact in these noiseless runs.

## Coverage intervention

Narrow training stays on x1=0. Recovery training applies known probing actions u=Kx+0.05*Gaussian noise to the plant (no unknown process noise),
then labels each visited history with the expert counterfactual action chunk K F^j x. Input histories contain applied actions, not expert target labels.
The expert uses hidden state to generate labels; the learner receives only its assigned observations and applied-action history.

## One-step results (H=16)

| Sensing | Training | History | Median test MSE | Worst per-step radius | Feature rank/dimension |
|---|---|---:|---:|---:|---:|
| full | narrow | 1 | 9.108e+12 | 1.124978 | 1/2 |
| full | narrow | 2 | 6.177e+09 | 1.104049 | 1/6 |
| full | narrow | 4 | 2.291e+04 | 1.068571 | 1/14 |
| full | narrow | 8 | 0.0007396 | 1.018915 | 1/30 |
| full | narrow | 16 | 6.487e-07 | 0.971100 | 1/62 |
| full | recovery | 1 | 1.218e-18 | 0.750000 | 2/2 |
| full | recovery | 2 | 6.63e-10 | 0.750000 | 4/6 |
| full | recovery | 4 | 1.075e-09 | 0.750000 | 8/14 |
| full | recovery | 8 | 1.106e-09 | 0.750000 | 16/30 |
| full | recovery | 16 | 1.106e-09 | 0.750000 | 32/62 |
| partial | narrow | 1 | 9.108e+12 | 1.124978 | 1/1 |
| partial | narrow | 2 | 6.177e+09 | 1.104049 | 1/4 |
| partial | narrow | 4 | 2.291e+04 | 1.068571 | 1/10 |
| partial | narrow | 8 | 0.0007396 | 1.018915 | 1/22 |
| partial | narrow | 16 | 6.487e-07 | 0.971100 | 1/46 |
| partial | recovery | 1 | 2.38e+14 | 1.146360 | 1/1 |
| partial | recovery | 2 | 7.154e-09 | 0.750000 | 4/4 |
| partial | recovery | 4 | 1.214e-08 | 0.750000 | 8/10 |
| partial | recovery | 8 | 2.235e-08 | 0.750001 | 16/22 |
| partial | recovery | 16 | 2.967e-08 | 0.750002 | 32/46 |

## Reading the files

- `observation_coverage_comparison.png`: separates observation loss from training coverage at one-step execution.
- Each suite has `context_hypothesis.png` for the history/execution grid and `metrics.csv` for every configuration.
- Partial suites also have `learner_view.png`, `learner_view.csv`, and `learner_inputs.csv` showing exactly which features are passed to the policy.
- The reconstructed-state curves are privileged known-model diagnostics; they are not inputs to the learned ridge policies.
- Full-state static and analytic-memory baselines are privileged references under partial observation. The separately labeled known-model observer uses only partial observations and actual past actions.

The full and partial narrow fits agree numerically across all paired configurations: removing an always-zero training feature does not fix or worsen their learned extension.
These results concern noiseless measurements and a single initial departure. They do not imply robustness to continuing process or measurement noise, or generalization to unknown plants.
