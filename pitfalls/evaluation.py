"""Paired held-out recovery rollouts for the noiseless hard example."""
import numpy as np
from model import rollout


def make_test_states(count, seed):
    rng = np.random.default_rng(seed)
    states = []
    for test_id in range(count):
        x2 = float(rng.uniform(-1, 1))
        delta = float(rng.choice([-1, 1]) * rng.uniform(.0005, .0015))
        for suite, x1 in [('support', 0.), ('perturbed', delta)]:
            states.append(dict(test_id=test_id, suite=suite, x1=x1, x2=x2))
    return states


def deploy(system, expert_gain, policy, execution, initial, steps, warmup):
    """Supply on-support expert history, then perturb only at deployment."""
    decay = float((system.A + expert_gain)[1, 1])
    disturbances = np.zeros((warmup + steps, 2))
    x0 = np.array(initial, dtype=float)
    if warmup:
        disturbances[warmup - 1, 0] = initial[0]
        x0 = np.array([0., initial[1] * decay**(-warmup)])
    xs = rollout(system, expert_gain, policy, execution, x0,
                 warmup + steps, warmup, disturbances)['x'][warmup:]
    if not np.allclose(xs[0], initial, atol=1e-12):
        raise RuntimeError('Warm-up changed the deployment initial state')
    return xs


def tracking_loss(xs, reference):
    """Average squared full-state tracking distance; exclude initial state."""
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.mean(np.sum((xs[1:] - reference[1:])**2, axis=1)))
    return value if np.isfinite(value) else float('inf')


def evaluate_test_set(system, gain, policy, execution, states, references,
                      steps, warmup, identity):
    losses = {'support': [], 'perturbed': []}
    records = []
    for state, ref in zip(states, references):
        xs = deploy(system, gain, policy, execution,
                    (state['x1'], state['x2']), steps, warmup)
        loss = tracking_loss(xs, ref)
        losses[state['suite']].append(loss)
        records.append(dict(**identity, **state, test_loss=loss))
    return dict(test_loss=float(np.mean(losses['perturbed'])),
                support_test_loss=float(np.mean(losses['support'])),
                worst_test_loss=float(max(losses['perturbed']))), records


def summarize_hypothesis(rows, contexts, executions, horizon, seeds, tolerance):
    """Require every training seed to pass loss and exact stability at fixed H."""
    candidates, summary = [], []
    for r in contexts:
        passing = []
        for ell in executions:
            if ell > horizon:
                continue
            group = [v for v in rows if v['context'] == r
                     and v['horizon'] == horizon and v['execution'] == ell]
            if len(group) != len(seeds) or {v['seed'] for v in group} != set(seeds):
                raise ValueError('Missing or duplicate paired training seeds')
            worst = max(v['test_loss'] for v in group)
            radius = max(v['per_step_radius'] for v in group)
            passes = np.isfinite(worst) and worst <= tolerance and radius < 1
            candidates.append(dict(context=r, execution=ell, horizon=horizon,
                                   worst_seed_test_loss=worst, worst_per_step_radius=radius,
                                   passes=bool(passes)))
            if passes:
                passing.append(ell)
        summary.append(dict(context=r, horizon=horizon,
                            shortest_execution=min(passing) if passing else None,
                            no_chunking_passes=1 in passing,
                            loss_tolerance=tolerance))
    return candidates, summary
