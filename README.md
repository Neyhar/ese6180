# ESE 618: history, observability, and action chunking

Research question: how does context relative to the observability index affect closed-loop imitation error, and when does sufficient context favor shorter execution horizons?

This is an executable starting framework, not a claim that longer history always favors shorter chunks.

## Run

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py --output results/noiseless
python run_experiment.py --noise 0.02 --output results/disturbed
```

Only NumPy is required. Each run exports a configuration JSON and a CSV of metrics. No neural-network packages or robot simulator are needed.

## Mathematical scope

Plant: x_(t+1) = A x_t + B u_t + w_t; observation y_t = C x_t. Expert: u*_t = K x_t. Training demonstrations are noiseless. Evaluation optionally includes process disturbances; observations remain noiseless. A and K in the example produce stable open-loop and expert closed-loop dynamics.

The expert is Markovian in the hidden state. Partial observability alone causes the learner's need for memory. Arbitrary expert memory, latent intent, nonlinear dynamics, sensor noise, and physical actuator delays are outside this initial model.

Context r counts observations including the current observation, and optionally r-1 past executed actions. Prediction H counts predicted actions. Execution ell counts actions used before replanning. They are independent parameters, with ell <= H.

For histories ordered oldest first:

- Y = O_r x_old + T_r U
- x_t = A^(r-1) x_old + R_r U
- M_r = [A^(r-1) O_r^dagger, R_r - A^(r-1) O_r^dagger T_r]
- x_t = M_r [Y; U], when O_r has full column rank and the history is noiseless.
- With F=A+BK, the exact H-action map stacks K F^j M_r for j=0,...,H-1.

`model.py` constructs these matrices explicitly. Reconstruction is tested on arbitrary inputs, not just expert trajectories. Unknown process disturbances invalidate exact finite-history reconstruction; the disturbed experiment intentionally measures this limitation.

## What is implemented

- `LinearSystem`: dynamics, observation, observability index, history matrices, reconstruction.
- `demonstrations`: full-state expert demonstrations; hidden states are used for evaluation, never as learned-policy features.
- `dataset` / `fit_policy`: ridge regression from histories to expert action chunks.
- `exact_policy`: model-aware oracle baseline using known A, B, C, K.
- `rollout`: chunk execution, observing at every step and replanning from the actual learner history.
- `run_experiment.py`: context {1,2,3,4}, execution {1,2,4,8}, fixed prediction horizon 8. Baselines include learned outputs-only, learned outputs-and-actions, and exact outputs-and-actions policies where observable.

The example hides the second state coordinate and has observability index 2. Edit `example_system()` to change the plant and expert; changing dimensions also requires updating the initial-state and disturbance shapes in the experiment runner.

## Evaluation conventions and interpretation

All methods receive the same three-step expert warm-up, initial conditions, and process disturbances. Warm-up actions are privileged initialization, excluded from scored action steps; deploying without an expert requires a separate initialization rule. The warm-up supplies enough history to every policy. Replanning observes the current state output before choosing the action. Measurements during a chunk are retained for the next replan.

Training and validation are separated by trajectory. Metrics include held-out chunk-action MSE, rollout expert-action MSE, state-trajectory error against a full-state expert receiving the same disturbances, and an illustrative quadratic state/action cost (Q=I, R=I; K is not claimed to be its optimizer). Standard deviations summarize evaluation episodes, not uncertainty across training seeds.

In the exact deterministic baseline, every execution length should match the expert to numerical precision. This is an essential control experiment. Do not infer an optimal execution length from tiny floating-point differences.

Expert trajectories can lie on a low-dimensional subspace of history features. The rank diagnostic exposes this: perfect training fit does not identify behavior on arbitrary learner histories. Ridge regression selects one extension, not necessarily the robust one. Past expert actions can reveal hidden state, which motivates the outputs-only ablation. More noiseless demonstrations of the same linear expert need not resolve this rank deficiency.

## Next research steps

1. Prove the exact history representation and the weaker action-reconstruction condition ker(O_r) subset ker(K A^(r-1)), with arbitrary known past inputs.
2. Sweep data count, ridge strength, and independent training seeds; inspect on-policy versus held-out error.
3. Add deliberately perturbed data collection with explicit expert labels if studying coverage beyond expert histories. Keep this intervention separate from ordinary behavioral cloning.
4. Add delayed-observation policies u_t = W_d y_(t-d), distinguishing delayed conditioning from actuator latency.
5. Compare systems with different observability indices and conditioning; define functional action recovery before claiming a universal context threshold.
6. Only then add sensor noise, model mismatch, or a robotics task.

## Starting references

- Zhang et al., Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control: https://arxiv.org/abs/2507.09061
- Zeng et al., Revisiting Open-Loop Execution in Robotics: https://arxiv.org/abs/2608.15938
- Lazzati et al., Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?: https://arxiv.org/abs/2608.02547

References are carried forward from the accompanying project discussion. The framework isolates mechanisms rather than reproducing those papers' experiments.
