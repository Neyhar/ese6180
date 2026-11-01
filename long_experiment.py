"""Matched-window training and batched evaluation for long-horizon studies."""
import numpy as np
from model import HistoryPolicy, history_feature


def window_ids(trajectories, max_context, max_horizon, windows_per_trajectory):
    """A single set of (trajectory,time) indices shared by ALL contexts and H."""
    ids = []
    for i, tr in enumerate(trajectories):
        first, last = max_context-1, len(tr['u'])-max_horizon
        if last < first:
            raise ValueError('Training trajectories are too short')
        times = np.unique(np.linspace(first, last, min(windows_per_trajectory, last-first+1), dtype=int))
        ids.extend((i, int(t)) for t in times)
    return ids


def matched_arrays(trajectories, ids, context, max_horizon, include_actions=True):
    X = np.array([history_feature(trajectories[i]['y'], trajectories[i]['u'], t,
                                 context, include_actions) for i,t in ids])
    Y = np.array([trajectories[i]['u'][t:t+max_horizon].reshape(-1) for i,t in ids])
    return X, Y


def fit_matched(X, Y, context, action_dim, ridge=1e-6, include_actions=True):
    """One-step target fit separately from chunk targets, with shared SVD of X.

    Output blocks are separable. Solving H_max once also solves every shorter
    H on these same windows: the first H*m output columns are unchanged.
    """
    if ridge < 0 or Y.shape[1] % action_dim:
        raise ValueError('Invalid regularization or target shape')
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    tol = np.finfo(float).eps * max(X.shape) * (s[0] if len(s) else 0)
    gain = s / (s*s + ridge) if ridge else np.divide(1, s, out=np.zeros_like(s), where=s>tol)
    def solve(target):
        return (Vt.T @ (gain[:,None] * (U.T @ target))).T
    one = HistoryPolicy(solve(Y[:,:action_dim]), context, 1, action_dim, include_actions)
    chunk = HistoryPolicy(solve(Y), context, Y.shape[1]//action_dim, action_dim, include_actions)
    delta = float(np.max(np.abs(one.weights-chunk.weights[:action_dim])))
    np.testing.assert_allclose(one.weights, chunk.weights[:action_dim], atol=1e-10, rtol=1e-9)
    return one, chunk, dict(feature_rank=int(np.sum(s>tol)), feature_dim=X.shape[1],
                            samples=X.shape[0], first_action_weight_gap=delta)


def batch_rollout(system, K, policy, execution, initial, disturbances, warmup,
                  divergence_limit=1e6):
    """Vectorize episodes; predict only the executed prefix, mathematically identical.

    Failed episodes retain infinity after divergence; never dropped from averages.
    All compared methods observe each step and receive identical disturbances.
    """
    if not 1 <= execution <= policy.horizon or warmup < policy.context-1:
        raise ValueError('Invalid execution horizon or warm-up')
    episodes, steps, n = disturbances.shape
    m, p = system.B.shape[1], system.C.shape[0]
    xs = np.empty((episodes, steps+1, n))
    ys = np.empty((episodes, steps+1, p))
    us = np.empty((episodes, steps, m))
    xs[:,0] = initial; ys[:,0] = initial @ system.C.T
    alive = np.ones(episodes, dtype=bool)
    fail = np.full(episodes, -1, dtype=int)
    for t in range(steps):
        if t < warmup:
            action = xs[:,t] @ K.T
        else:
            offset = (t-warmup) % execution
            if offset == 0:
                r = policy.context
                z = ys[:,t-r+1:t+1].reshape(episodes, -1)
                if policy.include_actions:
                    z = np.concatenate((z, us[:,t-r+1:t].reshape(episodes, -1)), axis=1)
                chunk = (z @ policy.weights[:execution*m].T).reshape(episodes, execution, m)
            action = chunk[:,offset]
        # Dead episodes are kept as zeros internally, then marked infinity in results.
        action = np.where(alive[:,None], action, 0)
        with np.errstate(over='ignore', invalid='ignore'):
            nxt = xs[:,t] @ system.A.T + action @ system.B.T + disturbances[:,t]
        valid = np.all(np.isfinite(nxt), axis=1) & np.all(np.isfinite(action), axis=1)
        valid &= np.linalg.norm(nxt, axis=1) <= divergence_limit
        valid &= np.linalg.norm(action, axis=1) <= divergence_limit
        fail[alive & ~valid] = t
        alive &= valid
        xs[:,t+1] = np.where(alive[:,None], nxt, 0)
        us[:,t] = action
        ys[:,t+1] = xs[:,t+1] @ system.C.T
    for e,t in enumerate(fail):
        if t>=0:
            xs[e,t+1:] = np.inf; us[e,t:] = np.inf
    return xs, us, fail


def expert_reference(system, K, initial, disturbances):
    xs = np.empty((len(initial), disturbances.shape[1]+1, initial.shape[1]))
    xs[:,0] = initial
    F = system.A+system.B@K
    for t in range(disturbances.shape[1]):
        xs[:,t+1] = xs[:,t] @ F.T + disturbances[:,t]
    return xs


def trajectory_errors(xs, reference, warmup):
    # Per-coordinate normalization prevents dimension alone multiplying the score.
    return np.mean((xs[:,warmup+1:]-reference[:,warmup+1:])**2, axis=2)


def summarize_errors(errors, tail_steps):
    return dict(trajectory_mse=float(np.mean(errors)),
                tail_mse=float(np.mean(errors[:,-tail_steps:])),
                cumulative_error=float(np.mean(np.sum(errors, axis=1))))


def paired_tail_difference(errors, baseline, tail_steps):
    """Infinity minus infinity is deliberately undefined; never discard failures."""
    with np.errstate(invalid='ignore'):
        return float(np.mean(errors[..., -tail_steps:] - baseline[..., -tail_steps:]))
