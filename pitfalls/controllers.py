"""The specific controllers in Simchowitz, Pfrommer & Jadbabaie (arXiv:2503.09722 v4),
simulated on the paper's own systems. PDF page numbers are those printed in the paper.

Part A, scalar toy, Sec. 5.2, Eq. (5.1), p. 18:  x_{t+1} = xi*rho*x_t + u_t,  x_1 = eps,
  rho > 1 known, sign xi in {-1, +1} unknown.
    smooth linear  u = k x                              (any k: one sign diverges, p. 18)
    action-switch  u = -rho x (t odd), +rho x (t even)  (Sec. 5.2.1, p. 18)
    adaptive       u_t = -((x_2 - u_1)/x_1) x_t         (Sec. 5.2.1, p. 18: history-dependent)
    BGR            u = +/- rho x w.p. 1/2               (Sec. 5.2.2, pp. 18-19)
    concentric     u = +rho x if j(x) even else -rho x, |x| in I_j = ((2rho)^{-2j}, (2rho)^{-2(j-1)}]
                                                        (Sec. 5.2.3, p. 19)

Part B, the 2D challenging pair, Def. 4.1, p. 15 (mu = 1/8 as in Appendix J, p. 98).
  Appendix H (Prop. H.1, pp. 91-93) applies the same three tricks to this pair. The learner
  keeps the expert's e_2 column, K_1(I - e1 e1^T), and adds a base policy that cancels the
  e_1 column of one candidate plant, -A_j e1 e1^T x. Note that -A_j e1 = K_j e1, so the base
  policy is simply "play expert j's e_1 response". The pp. 92-93 text writes (I - e1 e1^T)
  there; we read that as a typo for e1 e1^T, because only that version acts on e_1 as the
  proof requires. Also simulated: the smooth Markov learners (Prop. 4.1(b), p. 15), zero
  control (Remark 4.1, p. 15), open-loop action chunks (Sec. 5.1, pp. 17-18; Prop. H.1(c)),
  and a *linear* controller with one step of memory given by an analytic construction.

Run from repo root:  python3 pitfalls/controllers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitfalls import challenging_pair  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results' / 'controllers'


# ---------------------------------------------------------------- Part A: scalar (Sec. 5.2)
def concentric_sign(x, rho):
    if x == 0:
        return 0.0
    j = int(np.floor(-np.log(abs(x)) / (2 * np.log(2 * rho)))) + 1   # |x| in I_j
    return 1.0 if j % 2 == 0 else -1.0


def scalar_run(policy, xi, rho, eps, T, rng):
    xs, us = [eps], []
    for t in range(1, T + 1):
        x = xs[-1]
        if policy == 'smooth k=0':
            u = 0.0
        elif policy == 'smooth k=-rho':          # the expert for xi=+1
            u = -rho * x
        elif policy == 'action-switch':
            u = (-rho if t % 2 == 1 else rho) * x
        elif policy == 'adaptive':
            u = -((xs[1] - us[0]) / xs[0]) * x if t >= 2 and xs[0] != 0 else 0.0
        elif policy == 'BGR':
            u = rng.choice([-1.0, 1.0]) * rho * x
        elif policy == 'concentric':
            u = concentric_sign(x, rho) * rho * x
        us.append(u)
        xs.append(xi * rho * x + u)
        if abs(xs[-1]) > 1e12:                   # clipped later anyway
            xs.append(np.inf); break
    xs = np.array(xs + [xs[-1]] * (T + 1 - len(xs)))[:T + 1]
    return np.nan_to_num(np.abs(xs), posinf=1e12)


def part_a(rho=1.5, eps=1e-3, T=30, mc=4000):
    rng = np.random.default_rng(618)
    policies = ['smooth k=0', 'smooth k=-rho', 'action-switch', 'adaptive', 'BGR', 'concentric']
    curves, rows = {}, []
    for p in policies:
        worst = None
        for xi in (-1, 1):
            n = mc if p == 'BGR' else 1
            runs = np.array([scalar_run(p, xi, rho, eps, T, rng) for _ in range(n)])
            clipped = np.minimum(1.0, runs).mean(0)            # E[min{1, |x_t|}], p. 19
            worst = clipped if worst is None else np.maximum(worst, clipped)
            if p == 'BGR':
                rows.append(dict(policy=p, xi=xi, P_nonzero_t10=float((runs[:, 10] > 0).mean()),
                                 max_clipped=float(clipped.max())))
        curves[p] = worst
        rows.append(dict(policy=p, xi='worst', max_clipped=float(worst.max()),
                         final=float(worst[-1])))
    # BGR scaling with eps: the paper says the max over t is at most eps^p, p in (0,1) (p. 19).
    eps_grid = np.logspace(-8, -2, 7)
    bgr_scaling = []
    for e in eps_grid:
        runs = np.array([scalar_run('BGR', 1, rho, e, 60, rng) for _ in range(mc)])
        bgr_scaling.append(float(np.minimum(1, runs).mean(0).max()))
    slope = np.polyfit(np.log(eps_grid), np.log(bgr_scaling), 1)[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for p, c in curves.items():
        ax[0].semilogy(np.maximum(c, 1e-16), label=p, marker='o' if p in ('adaptive', 'concentric') else None, ms=3)
    ax[0].set(xlabel='t', ylabel=r'worst over $\xi$ of E min(1,|x_t|)',
              title=f'Scalar toy Eq. (5.1), rho={rho}, eps={eps}')
    ax[0].legend(fontsize=7)
    ax[1].loglog(eps_grid, bgr_scaling, 'o-', label=f'BGR, fitted slope {slope:.2f}')
    ax[1].loglog(eps_grid, eps_grid, 'k:', label='slope 1 (linear in eps)')
    ax[1].set(xlabel='initial error eps', ylabel='max_t E min(1,|x_t|)', title='BGR: sublinear in eps (p. 19)')
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'scalar_controllers.png', dpi=150); plt.close(fig)
    return rows, dict(eps=eps_grid.tolist(), bgr_max=bgr_scaling, slope=float(slope))


# ---------------------------------------------------------------- Part B: 2D challenging pair
def memory_controller(mu):
    """Construct a LINEAR controller with one step of memory that matches the expert on V.
    It adds an e_1-channel action v_t = k0 x1_t + k1 x1_{t-1} + g v_{t-1}, which is 0 on expert data.
    Both plants are lower triangular once u keeps K_1 e_2, so e_1 obeys x1+ = a_i x1 + v with
    a_1 = 1+mu, a_2 = -(1-mu/4). This is simultaneous stabilisation of two scalar plants."""
    a1, a2 = 1 + mu, -(1 - mu / 4)
    k0 = -(a1 + a2) / 2
    k1 = -((a1 - a2) / 4) ** 2
    # e1 recurrence has double roots +/- (a1-a2)/4; include the x2 mode.
    full_rho = max(abs((a1 - a2) / 4), abs(1 - 2 * mu))
    return np.array([k0, k1, 0.0]), float(full_rho)


def run_2d(name, i, mu, eps, T, rng, chunk=None, mem=None):
    A1, A2, K1, K2 = challenging_pair(mu)
    A, Ks = (A1, A2)[i - 1], (K1, K2)
    e1 = np.array([1.0, 0.0])
    P_V = np.diag([0.0, 1.0])
    keep = K1 @ P_V                                      # expert's e_2 column, Eq. (H.1)
    base = lambda j, x: Ks[j - 1] @ e1 * x[0]            # -A_j e1 e1^T x = K_j e1 x_1
    x = np.array([eps, 0.5])                             # learner error eps along e_1, as in (H.5)
    xs, v_prev, x1_prev, plan = [x], 0.0, 0.0, []
    hist = []
    for t in range(1, T + 1):
        if name == 'expert (oracle)':
            u = Ks[i - 1] @ x
        elif name == 'zero control':
            u = np.zeros(2)
        elif name == 'smooth min-norm':
            u = keep @ x
        elif name == 'smooth minimax':
            u = keep @ x + np.array([-5 * mu / 8, 0]) * x[0]
        elif name == 'action-switch':
            u = keep @ x + base(1 if t % 2 else 2, x)
        elif name == 'BGR':
            u = keep @ x + base(rng.integers(1, 3), x)
        elif name == 'concentric':
            # a mismatched e_1 column multiplies x_1 by m = |a_1 - a_2| = 2 + 3mu/4, the analogue
            # of 2rho in the scalar toy, so the rings use ratio m^2 (concentric_sign takes rho = m/2)
            j = 1 if concentric_sign(x[0], (2 + 0.75 * mu) / 2) > 0 else 2
            u = keep @ x + base(j, x)
        elif name == 'adaptive (history)':
            # identify i from one transition, x_t - u_{t-1} = A_i x_{t-1}, then act as expert i
            if hist:
                xp, up = hist[-1]
                r = [np.linalg.norm(x - up - Aj @ xp) for Aj in (A1, A2)]
                j = int(np.argmin(r)) + 1 if abs(xp[0]) > 1e-15 else 1
            else:
                j = 1
            u = keep @ x + base(j, x)
        elif name == 'linear memory':
            k0, k1, g = mem
            v = k0 * x[0] + k1 * x1_prev + g * v_prev
            u = keep @ x + np.array([v, 0.0])
            v_prev, x1_prev = v, x[0]
        elif name.startswith('chunk'):
            # open-loop chunk from a BC model with the min-norm e_1 column: it predicts the expert
            # closed loop diag(0, 1-2mu) on V and applies keep @ F^j x_t
            if not plan:
                F = np.diag([0.0, 1 - 2 * mu])
                plan = [keep @ np.linalg.matrix_power(F, k) @ x for k in range(chunk)]
            u = plan.pop(0)
        hist.append((x, u))
        x = A @ x + u
        xs.append(x)
    return np.array(xs)


def part_b(mu=0.125, eps=1e-3, T=60):
    rng = np.random.default_rng(618)
    mem, mem_rho = memory_controller(mu)
    names = ['expert (oracle)', 'smooth min-norm', 'smooth minimax', 'zero control',
             'action-switch', 'BGR', 'concentric', 'adaptive (history)', 'linear memory',
             'chunk 4', 'chunk 16']
    rows, curves = [], {}
    for n in names:
        per_i = []
        for i in (1, 2):
            reps = 2000 if n == 'BGR' else 1
            e1 = np.array([np.abs(run_2d(n, i, mu, eps, T, rng, chunk=int(n.split()[1]) if n.startswith('chunk') else None,
                                         mem=mem)[:, 0]) for _ in range(reps)])
            per_i.append(np.minimum(1, e1).mean(0))
        worst = np.maximum(*per_i)
        curves[n] = worst
        rows.append(dict(controller=n, worst_max_e1=float(worst.max()), worst_final_e1=float(worst[-1]),
                         plant1_max=float(per_i[0].max()), plant2_max=float(per_i[1].max())))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {'expert (oracle)': 'k-', 'zero control': 'k--'}
    for n, c in curves.items():
        ax.semilogy(np.maximum(c, 1e-17), styles.get(n, '-'), label=n, lw=1.4)
    ax.set(xlabel='t', ylabel=r'worst over $i$ of E min(1, |$x_{t,1}$|)',
           title=f'Challenging pair (Def. 4.1), mu={mu}, eps={eps} on e1', ylim=(1e-17, 3))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / 'pair_controllers.png', dpi=150); plt.close(fig)
    return rows, dict(k0=mem[0], k1=mem[1], g=mem[2], worst_rho=mem_rho)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a_rows, bgr = part_a()
    b_rows, mem = part_b()
    json.dump(dict(scalar=a_rows, bgr_scaling=bgr, pair=b_rows, linear_memory=mem),
              open(OUT / 'summary.json', 'w'), indent=1)
    print('Part A, scalar toy (rho=1.5, eps=1e-3, T=30): worst-case max_t E min(1,|x_t|)')
    for r in a_rows:
        print('  ', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  BGR max vs eps: slope {bgr['slope']:.3f} (paper: eps^p, p<1)")
    print('Part B, challenging pair (mu=1/8, eps=1e-3 on e1, T=60): worst-case over i')
    for r in b_rows:
        print(f"   {r['controller']:20s} max={r['worst_max_e1']:.3g} final={r['worst_final_e1']:.3g} "
              f"(plant1 {r['plant1_max']:.3g}, plant2 {r['plant2_max']:.3g})")
    print('  linear memory controller', {k: round(v, 4) for k, v in mem.items()})


if __name__ == '__main__':
    main()
