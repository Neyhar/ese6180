"""Run: python run_experiment.py --output results --noise 0.0"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from model import (example_system, demonstrations, dataset, fit_policy, exact_policy, rollout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results')
    parser.add_argument('--noise', type=float, default=0.0, help='Evaluation process-noise standard deviation')
    parser.add_argument('--seed', type=int, default=618)
    parser.add_argument('--train-trajectories', type=int, default=40)
    args = parser.parse_args()
    if args.noise < 0 or args.train_trajectories < 1:
        parser.error('Noise must be nonnegative and training count positive')
    rng = np.random.default_rng(args.seed)
    system, K = example_system()
    contexts, executions, horizon, steps = [1, 2, 3, 4], [1, 2, 4, 8], 8, 60
    warmup = max(contexts)-1
    train = demonstrations(system, K, rng.normal(size=(args.train_trajectories, 2)), steps)
    validation = demonstrations(system, K, rng.normal(size=(20, 2)), steps)
    initial = rng.normal(size=(20, 2))
    noise = rng.normal(scale=args.noise, size=(20, steps, 2))
    references = []
    for x0, w in zip(initial, noise):
        xs = [x0]
        for t in range(steps):
            xs.append(system.step(xs[-1], K @ xs[-1], w[t]))
        references.append(np.asarray(xs))
    rows = []
    traces = {}
    for context in contexts:
        policies = []
        for inputs in (True, False):
            policy, diagnostics = fit_policy(train, context, horizon, include_actions=inputs)
            policies.append(('learned_yu' if inputs else 'learned_y', policy, diagnostics))
        if context >= system.observability_index():
            policies.append(('exact_yu', exact_policy(system, K, context, horizon), {}))
        for name, policy, diagnostics in policies:
            X, Y = dataset(validation, context, horizon, policy.include_actions)
            heldout = float(np.mean((X @ policy.weights.T-Y)**2))
            for execution in executions:
                errors, costs, action_errors = [], [], []
                for episode, (x0, w, reference) in enumerate(zip(initial, noise, references)):
                    tr = rollout(system, K, policy, execution, x0, steps, warmup, w)
                    if episode == 0:
                        traces[(name, context, execution)] = tr
                    errors.append(np.mean(np.sum((tr['x'][warmup+1:] - reference[warmup+1:])**2, axis=1)))
                    costs.append(np.mean(np.sum(tr['x'][warmup:-1]**2, axis=1) + np.sum(tr['u'][warmup:]**2, axis=1)))
                    action_errors.append(np.mean((tr['u'][warmup:] - tr['x'][warmup:-1] @ K.T)**2))
                rows.append(dict(policy=name, context=context, execution=execution, prediction=horizon,
                                 validation_mse=heldout, trajectory_mse=float(np.mean(errors)),
                                 trajectory_mse_std=float(np.std(errors)), cost=float(np.mean(costs)),
                                 rollout_action_mse=float(np.mean(action_errors)),
                                 feature_rank=diagnostics.get('feature_rank', ''), feature_dim=diagnostics.get('feature_dim', '')))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = vars(args) | dict(contexts=contexts, executions=executions, horizon=horizon, steps=steps, warmup=warmup,
                              evaluation_trajectories=20, ridge=1e-6, observability_index=system.observability_index(),
                              A=system.A.tolist(), B=system.B.tolist(), C=system.C.tolist(), K=K.tolist())
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    from plots import experiment_plots
    experiment_plots(rows, config, traces, references[0], out/'figures')
    print(f'Figures saved to {out}/figures (PNG and PDF)')
    print(f'Wrote {len(rows)} configurations to {out}/metrics.csv; h={system.observability_index()}')
    print('Expert-only histories may be rank deficient; inspect feature_rank and rollout errors together.')


if __name__ == '__main__':
    main()
