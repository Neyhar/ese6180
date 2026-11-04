"""Can a trained history policy find the context-2 stabilizer of the challenging pair?

The policy is linear in the repo's history feature [x_{t-r+1..t}, u_{t-r+1..t-1}] (full state,
C = I, first action only). We score each fit by its worst-case closed-loop spectral radius over
the two candidate plants A_1, A_2 (Def. 4.1, p. 15). The fit stabilizes both plants iff
rho_worst < 1. We also report rollouts from an e_1 error of 1e-3.

Training regimes:
  narrow      expert demos from plant 1 on span(e_2), noiseless (the paper's data; the sweep's failure)
  dart-1      plant 1 only, process noise sd sigma injected at every demo step, expert labels
              (DART-style, cf. Remark I.1, p. 93). The data then identifies K_1.
  dart-both   the same, but half the demos come from plant 2 labelled by K_2. No linear policy
              fits both exactly, so least squares has to compromise.
  robust      narrow data, plus a penalty on rho_worst during fitting: the objective rather than
              the data selects the stabilizer.

Run from repo root:  python3 pitfalls/train.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from model import dataset  # noqa: E402
from pitfalls import challenging_pair  # noqa: E402

MU = 0.125
A1, A2, K1, K2 = challenging_pair(MU)
PLANTS = [(A1, K1), (A2, K2)]


def demos(regime, n=64, steps=30, sigma=0.05, seed=618):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        A, K = PLANTS[1] if (regime == 'dart-both' and k % 2) else PLANTS[0]
        noise = 0.0 if regime in ('narrow', 'robust') else sigma
        x = np.array([0.0, rng.uniform(-1, 1)])
        xs, us = [x], []
        for _ in range(steps):
            u = K @ x                                   # expert label on the visited state
            x = A @ x + u + noise * rng.normal(size=2)
            us.append(u); xs.append(x)
        out.append(dict(x=np.array(xs), y=np.array(xs), u=np.array(us)))
    return out


def closed_loop(W0, r, A):
    """Lifted closed loop for u_t = W0 [x_{t-r+1..t}, u_{t-r+1..t-1}], z = [x_t..x_{t-r+1}, u_{t-1}..u_{t-r+1}]."""
    nz = 2 * r + 2 * (r - 1)
    F = np.zeros((nz, nz))                              # feature = F z
    for k in range(r):
        F[2 * k:2 * k + 2, 2 * (r - 1 - k):2 * (r - k)] = np.eye(2)
    for k in range(r - 1):
        F[2 * r + 2 * k:2 * r + 2 * k + 2, 2 * r + 2 * (r - 2 - k):2 * r + 2 * (r - 1 - k)] = np.eye(2)
    G = W0 @ F
    M = np.zeros((nz, nz))
    M[:2, :2] = A
    M[:2] += G
    for k in range(1, r):
        M[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = np.eye(2)
    if r > 1:
        M[2 * r:2 * r + 2] = G
        for k in range(1, r - 1):
            M[2 * r + 2 * k:2 * r + 2 * k + 2, 2 * r + 2 * (k - 1):2 * r + 2 * k] = np.eye(2)
    return M


def rho_each(W0, r):
    return [float(max(abs(np.linalg.eigvals(closed_loop(W0, r, A))))) for A, _ in PLANTS]


def fit(regime, r, ridge=1e-6, lam=10.0, margin=0.8):
    X, Y = dataset(demos(regime), r, 1)
    d = X.shape[1]
    W = np.linalg.lstsq(np.vstack((X, np.sqrt(ridge) * np.eye(d))),
                        np.vstack((Y, np.zeros((d, 2)))), rcond=None)[0].T
    mse = lambda W: float(np.mean((X @ W.T - Y) ** 2))
    if regime == 'robust':
        def loss(w):
            Wm = w.reshape(2, d)
            return 1e4 * mse(Wm) + lam * max(0.0, max(rho_each(Wm, r)) - margin) ** 2
        best = None
        for s in range(8):                              # multistart: rho is nonsmooth
            w0 = W.ravel() + 0.3 * np.random.default_rng(s).normal(size=W.size)
            res = minimize(loss, w0, method='Nelder-Mead',
                           options=dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-14))
            best = res if best is None or res.fun < best.fun else best
        W = best.x.reshape(2, d)
    return W, mse(W)


def rollout_err(W0, r, A, K, T=80, eps=1e-3):
    # expert warm-up supplies the history, then the error eps is placed on e_1
    x = np.array([0.0, 0.5]); xs, us = [x], []
    for _ in range(r - 1):
        u = K @ x; us.append(u); x = A @ x + u; xs.append(x)
    xs[-1] = x = x + np.array([eps, 0.0])
    peak = eps
    for _ in range(T):
        feat = np.concatenate([np.concatenate(xs[-r:]), np.concatenate(us[-(r - 1):]) if r > 1 else []])
        u = W0 @ feat; us.append(u); x = A @ x + u; xs.append(x)
        peak = max(peak, min(1.0, abs(x[0])))
        if abs(x[0]) > 1e6:
            break
    return peak, float(min(1.0, abs(x[0])))


def main():
    rows = []
    for regime in ['narrow', 'dart-1', 'dart-both', 'robust']:
        for r in [1, 2, 3, 4]:
            if regime == 'robust' and r > 3:
                continue
            W, mse = fit(regime, r)
            rho1, rho2 = rho_each(W, r)
            (p1, f1), (p2, f2) = [rollout_err(W, r, A, K) for A, K in PLANTS]
            rows.append(dict(regime=regime, r=r, train_mse=mse, rho_plant1=rho1, rho_plant2=rho2,
                             both_stable=bool(max(rho1, rho2) < 1), peak_e1=max(p1, p2), final_e1=max(f1, f2)))
            print(f"{regime:9s} r={r}  mse={mse:.1e}  rho(A1)={rho1:.3f} rho(A2)={rho2:.3f}  "
                  f"worst peak|x1|={max(p1, p2):.3g} final={max(f1, f2):.3g}")
    out = HERE / 'results' / 'train'
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(mu=MU, rows=rows), open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
