"""Exact history/chunk stability and a model-informed memory witness."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from model import HistoryPolicy

def memory_policy(mu):
    """Analytic full-state r=2 witness; model-informed, not learned from narrow data."""
    a1, a2 = 1 + mu, -(1 - mu / 4)
    W = np.zeros((2, 6))  # x_(t-1), x_t, u_(t-1)
    W[0, 0] = -((a1 - a2) / 4) ** 2
    W[0, 2] = -(a1 + a2) / 2
    W[0, 3] = -1.5 * mu
    return HistoryPolicy(W, 2, 1, 2)

def boundary_maps(system, policy, execution):
    """Exact maps at each phase of a committed chunk, including disturbance covariance.

    Boundary state z=[x_(t-r+1),...,x_t,u_(t-r+1),...,u_(t-1)].
    Return R_j, Q_j with z_(t+j)=R_j z_t + noise, unit iid plant noise.
    Pending actions are computed once from the boundary feature, never recomputed.
    """
    r, n, m, p = policy.context, system.A.shape[0], system.B.shape[1], system.C.shape[0]
    if not 1 <= execution <= policy.horizon:
        raise ValueError('Invalid execution horizon')
    d = n*r + m*(r-1)
    D = np.zeros((p*r + m*(r-1), d))
    D[:p*r, :n*r] = np.kron(np.eye(r), system.C)
    if r > 1:
        D[p*r:, n*r:] = np.eye(m*(r-1))
    S, U, E = np.zeros((d,d)), np.zeros((d,m)), np.zeros((d,n))
    if r > 1:
        S[:n*(r-1), n:n*r] = np.eye(n*(r-1))
        if r > 2:
            S[n*r:n*r+m*(r-2), n*r+m:] = np.eye(m*(r-2))
        U[-m:] = np.eye(m)
    current = slice(n*(r-1), n*r)
    S[current, current] = system.A
    U[current] = system.B
    E[current] = np.eye(n)
    R, Q = np.eye(d), np.zeros((d,d))
    phases = [(R.copy(), Q.copy())]
    for j in range(execution):
        G = policy.weights[j*m:(j+1)*m] @ D
        R = S @ R + U @ G
        Q = S @ Q @ S.T + E @ E.T
        phases.append((R.copy(), Q.copy()))
    return phases

def stability(system, policy, execution, noise):
    phases = boundary_maps(system, policy, execution)
    M, Q = phases[-1]
    radius = float(np.max(np.abs(np.linalg.eigvals(M))))
    result = dict(boundary_radius=radius, per_step_radius=radius**(1/execution),
                  stable=bool(radius < 1), stationary_state_rms=None)
    if radius < 1 and noise == 0:
        result['stationary_state_rms'] = 0.0
    elif radius < 1:
        P = solve_discrete_lyapunov(M, noise**2 * Q)
        n, r = system.A.shape[0], policy.context
        current = slice(n*(r-1), n*r)
        variances = [np.trace((R @ P @ R.T + noise**2 * V)[current,current])
                     for R,V in phases[:-1]]
        result['stationary_state_rms'] = float(np.sqrt(max(0, np.mean(variances))))
    return result
