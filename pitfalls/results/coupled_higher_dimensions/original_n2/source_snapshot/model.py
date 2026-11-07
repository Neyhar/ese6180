"""Partially observed linear imitation learning; column-vector math, row datasets."""
from dataclasses import dataclass
import numpy as np


@dataclass
class LinearSystem:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

    def __post_init__(self):
        self.A, self.B, self.C = [np.asarray(a, dtype=float) for a in (self.A, self.B, self.C)]
        n = self.A.shape[0]
        if self.A.shape != (n, n) or self.B.ndim != 2 or self.C.ndim != 2:
            raise ValueError('Expected matrix A (n,n), B (n,m), C (p,n)')
        if self.B.shape[0] != n or self.C.shape[1] != n:
            raise ValueError('Incompatible system dimensions')

    def step(self, x, u, disturbance=None):
        return self.A @ x + self.B @ u + (0 if disturbance is None else disturbance)

    def observe(self, x):
        return self.C @ x

    def observability(self, context):
        if context < 1:
            raise ValueError('Context must be positive')
        return np.vstack([self.C @ np.linalg.matrix_power(self.A, j) for j in range(context)])

    def observability_index(self):
        n = self.A.shape[0]
        for r in range(1, n + 1):
            if np.linalg.matrix_rank(self.observability(r)) == n:
                return r
        raise ValueError('System is not observable')

    def history_matrices(self, context):
        O = self.observability(context)
        p, n = self.C.shape
        m = self.B.shape[1]
        T = np.zeros((context * p, (context - 1) * m))
        for i in range(context):
            for j in range(i):
                T[i*p:(i+1)*p, j*m:(j+1)*m] = self.C @ np.linalg.matrix_power(self.A, i-j-1) @ self.B
        R = np.hstack([np.linalg.matrix_power(self.A, context-2-j) @ self.B for j in range(context-1)]) if context > 1 else np.zeros((n, 0))
        return O, T, R

    def reconstruction(self, context):
        """x_t = M [y_(t-r+1:t), u_(t-r+1:t-1)] for noiseless histories."""
        O, T, R = self.history_matrices(context)
        if np.linalg.matrix_rank(O) != self.A.shape[0]:
            raise ValueError('Context does not identify the full state')
        D = np.linalg.matrix_power(self.A, context-1) @ np.linalg.pinv(O)
        return np.hstack((D, R - D @ T))


def history_feature(observations, actions, t, context, include_actions=True):
    start = t - context + 1
    if start < 0:
        raise ValueError('Insufficient history; supply an explicit warm-up')
    y = np.asarray(observations[start:t+1]).reshape(-1)
    if include_actions:
        return np.concatenate((y, np.asarray(actions[start:t]).reshape(-1)))
    return y


@dataclass
class HistoryPolicy:
    weights: np.ndarray  # (H*m, r*p + (r-1)*m), without intercept
    context: int
    horizon: int
    action_dim: int
    include_actions: bool = True

    def predict(self, feature):
        return (self.weights @ feature).reshape(self.horizon, self.action_dim)


def exact_policy(system, K, context, horizon):
    F = system.A + system.B @ K
    targets = np.vstack([K @ np.linalg.matrix_power(F, j) for j in range(horizon)])
    return HistoryPolicy(targets @ system.reconstruction(context), context, horizon, system.B.shape[1])


def demonstrations(system, K, initial_states, steps):
    """Independent, noiseless expert trajectories. Split by trajectory, not window."""
    trajectories = []
    for x0 in initial_states:
        xs, ys, us = [np.array(x0)], [], []
        for _ in range(steps):
            ys.append(system.observe(xs[-1]))
            us.append(K @ xs[-1])
            xs.append(system.step(xs[-1], us[-1]))
        ys.append(system.observe(xs[-1]))
        trajectories.append(dict(x=np.asarray(xs), y=np.asarray(ys), u=np.asarray(us)))
    return trajectories


def dataset(trajectories, context, horizon, include_actions=True):
    features, targets = [], []
    for tr in trajectories:
        for t in range(context-1, len(tr['u'])-horizon+1):
            features.append(history_feature(tr['y'], tr['u'], t, context, include_actions))
            targets.append(tr['u'][t:t+horizon].reshape(-1))
    if not features:
        raise ValueError('Trajectories too short for context and horizon')
    return np.asarray(features), np.asarray(targets)


def fit_policy(trajectories, context, horizon, ridge=1e-6, include_actions=True):
    if ridge < 0:
        raise ValueError('Ridge must be nonnegative')
    X, Y = dataset(trajectories, context, horizon, include_actions)
    # Augmented least squares avoids explicitly forming the normal equations.
    d = X.shape[1]
    coefficients = np.linalg.lstsq(np.vstack((X, np.sqrt(ridge)*np.eye(d))),
                                 np.vstack((Y, np.zeros((d, Y.shape[1])))), rcond=None)[0]
    policy = HistoryPolicy(coefficients.T, context, horizon, trajectories[0]['u'].shape[1], include_actions)
    return policy, {'feature_rank': int(np.linalg.matrix_rank(X)), 'feature_dim': d,
                    'training_mse': float(np.mean((X @ coefficients - Y)**2))}


def rollout(system, K, policy, execution, x0, steps, warmup, disturbances=None):
    """Observe every step; replan every execution steps after common expert warm-up.

    Expert warm-up is an evaluation convention, not a deployable initialization rule.
    Process disturbances are applied after the chosen action. Measurements are exact.
    """
    if not 1 <= execution <= policy.horizon:
        raise ValueError('Execution must be between 1 and prediction horizon')
    if not policy.context-1 <= warmup < steps:
        raise ValueError('Warm-up must supply context and leave evaluation steps')
    w = np.zeros((steps, system.A.shape[0])) if disturbances is None else np.asarray(disturbances)
    xs, ys, us, chunk = [np.array(x0)], [system.observe(x0)], [], None
    for t in range(steps):
        if t < warmup:
            u = K @ xs[-1]
        else:
            offset = (t-warmup) % execution
            if offset == 0:
                feature = history_feature(ys, us, t, policy.context, policy.include_actions)
                chunk = policy.predict(feature)
            u = chunk[offset]
        us.append(u)
        xs.append(system.step(xs[-1], u, w[t]))
        ys.append(system.observe(xs[-1]))
    return dict(x=np.asarray(xs), y=np.asarray(ys), u=np.asarray(us))


def example_system():
    """Position/velocity-like stable plant, hidden second coordinate; h=2."""
    system = LinearSystem([[0.9, 0.2], [0, 0.8]], [[0], [0.2]], [[1, 0]])
    K = np.array([[-0.5, -0.6]])
    return system, K


def coupled_system(dimension, seed=618, decay=0.995, sensors=None, observability_index=None):
    """Stable cyclic coupling, scalar control, partial sensors; not independent copies.

    A=aP, B=b (unit norm), K=-g b^T P, so F=(aI-g bb^T)P
    and ||F||_2 <= a < 1 for 0<g<a. Default sensors see every other
    coordinate, giving h=2 for every n>=2. One sensor gives h=n.
    """
    if dimension < 2 or not 0.1 < decay < 1:
        raise ValueError('Require dimension >= 2 and 0.1 < decay < 1')
    P = np.roll(np.eye(dimension), -1, axis=1)
    rng = np.random.default_rng(seed + dimension)
    b = rng.normal(size=(dimension, 1))
    b /= np.linalg.norm(b)
    if observability_index is not None:
        if sensors is not None or not 1 <= observability_index <= dimension:
            raise ValueError('Specify either sensors or an index in [1,n]')
        # The largest cyclic gap between sensors is exactly h, hence index h.
        indices = np.arange(0, dimension, observability_index)
    elif sensors is None:
        indices = np.arange(0, dimension, 2)
    else:
        if not 1 <= sensors < dimension:
            raise ValueError('Sensor count must be between 1 and n-1')
        indices = np.arange(sensors) * dimension // sensors
    return LinearSystem(decay * P, b, np.eye(dimension)[indices]), -0.05 * b.T @ P
