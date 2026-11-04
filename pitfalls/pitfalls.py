"""Hard instance from Simchowitz, Pfrommer & Jadbabaie, "The Pitfalls of Imitation
Learning when Actions are Continuous" (arXiv:2503.09722), Sec. 4, Def. 4.1.

Linear core of the "challenging pair" (x in R^2, u in R^2, B = I):
    A_1 = [[1+mu, c], [-c, 1-2mu]],     K_1 = [[-(1+mu), -c], [c, 0]]
    A_2 = [[-(1-mu/4), c], [0, 1-2mu]], K_2 = [[1-mu/4, -c], [0, 0]],   c = 3mu/2
Both A_i are stable and both experts give A_i + K_i = diag(0, 1-2mu). Expert data
started on span(e_2) never leaves span(e_2), and K_1 e_2 = K_2 e_2, so the data
cannot reveal K e_1. Any K_hat matching the data has A_i + K_hat lower triangular
with (1,1) entry 1+mu+a (i=1) or -(1-mu/4)+a (i=2): at least one is >= 1+mu/4.

We take the true plant to be i=1 and compare:
  expert   K_1 (the "good" controller)
  wrong    K_2 (expert for the other plant; fits the data perfectly)
  minimax  K_hat with a = -5mu/8, b = 0: best worst case, still |eig| = 1+3mu/8 on one plant
  min-norm K_hat with a = b = 0: what ridge/least squares returns on span(e_2) data
Then we pose the same problem in the repo's history/chunking framework (model.py).

Run from repo root:  python pitfalls/pitfalls.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import LinearSystem, demonstrations, fit_policy, rollout  # noqa: E402


def challenging_pair(mu=0.2):
    if not 0 < mu <= 0.5:
        raise ValueError('mu must be in (0, 1/2]')
    c = 1.5 * mu
    A1 = np.array([[1 + mu, c], [-c, 1 - 2 * mu]])
    A2 = np.array([[-(1 - mu / 4), c], [0, 1 - 2 * mu]])
    K1 = np.array([[-(1 + mu), -c], [c, 0]])
    K2 = np.array([[1 - mu / 4, -c], [0, 0]])
    return A1, A2, K1, K2


def gains(mu):
    A1, A2, K1, K2 = challenging_pair(mu)
    c = 1.5 * mu
    data_matching = lambda a, b: np.array([[a, -c], [b, 0.0]])
    return {'expert K1': K1, 'wrong K2': K2,
            'minimax': data_matching(-5 * mu / 8, 0.0),
            'min-norm fit': data_matching(0.0, 0.0)}


def rho(M):
    return float(max(abs(np.linalg.eigvals(M))))


def part1_static(mu, steps, noise, seed, out):
    """Markovian state-feedback controllers on the true plant A_1."""
    A1, A2, K1, K2 = challenging_pair(mu)
    rng = np.random.default_rng(seed)
    w = noise * rng.normal(size=(steps, 2))
    rows, curves = [], {}
    for name, K in gains(mu).items():
        # Every candidate is consistent with expert data: K e_2 = K_1 e_2.
        assert np.allclose(K[:, 1], K1[:, 1])
        x, norms = np.array([0.0, 1.0]), []          # start in the data support
        for t in range(steps):
            x = A1 @ x + K @ x + w[t]
            norms.append(np.linalg.norm(x))
        curves[name] = np.array(norms)
        rows.append(dict(controller=name, rho_plant1=rho(A1 + K), rho_plant2=rho(A2 + K),
                         final_norm=norms[-1]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, n in curves.items():
        ax.semilogy(np.maximum(n, 1e-12), label=name)
    ax.semilogy((1 + mu) ** np.arange(steps) * noise, 'k:', lw=1, label=r'$(1+\mu)^t\sigma$')
    ax.set(xlabel='t', ylabel=r'$\|x_t\|$', title=f'Plant 1, mu={mu}, noise sd={noise}')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'static_controllers.png', dpi=150); plt.close(fig)
    return rows


def train_set(system, K, n_traj, steps, excite, rng):
    """Expert demos. excite=0: initial states uniform on [-e2, e2] (paper's support).
    excite>0: add N(0, excite^2) on e_1 so the learner sees how K_1 reacts to e_1."""
    x0 = np.zeros((n_traj, 2))
    x0[:, 1] = rng.uniform(-1, 1, n_traj)
    x0[:, 0] = excite * rng.normal(size=n_traj)
    return demonstrations(system, K, x0, steps)


def part2_framework(mu, C, contexts, executions, horizon, steps, warmup, noise,
                    excite, seeds, out, tag):
    A1, _, K1, _ = challenging_pair(mu)
    system = LinearSystem(A1, np.eye(2), C)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train = train_set(system, K1, 32, 40, excite, rng)
        w = noise * rng.normal(size=(steps, 2))
        for r in contexts:
            policy, diag = fit_policy(train, r, horizon)
            for ell in executions:
                tr = rollout(system, K1, policy, ell, np.array([0.0, 1.0]), steps, warmup, w)
                err = np.linalg.norm(tr['x'][warmup + 1:], axis=1)
                rows.append(dict(setting=tag, seed=seed, context=r, execution=ell,
                                 feature_rank=diag['feature_rank'], feature_dim=diag['feature_dim'],
                                 late_rms=float(np.sqrt(np.mean(np.minimum(err[-50:], 1e6) ** 2))),
                                 max_norm=float(min(err.max(), 1e6))))
    return rows


def heatmap(rows, contexts, executions, path, title):
    M = np.zeros((len(contexts), len(executions)))
    for i, r in enumerate(contexts):
        for j, e in enumerate(executions):
            v = [x['late_rms'] for x in rows if x['context'] == r and x['execution'] == e]
            M[i, j] = np.median(v)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(np.log10(M), cmap='viridis', aspect='auto')
    for i in range(len(contexts)):
        for j in range(len(executions)):
            ax.text(j, i, f'{M[i, j]:.2g}', ha='center', va='center', fontsize=7, color='w')
    ax.set_xticks(range(len(executions)), executions); ax.set_yticks(range(len(contexts)), contexts)
    ax.set(xlabel='execution length ell', ylabel='context r', title=title)
    fig.colorbar(im, label='log10 late RMS ||x||')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return M


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mu', type=float, default=0.2)
    p.add_argument('--noise', type=float, default=0.01)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--output', default=str(Path(__file__).parent / 'results'))
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    static = part1_static(a.mu, a.steps, a.noise, 618, out)
    write_csv(static, out / 'static_controllers.csv')

    contexts, executions = [1, 2, 3, 4, 6, 8], [1, 2, 4, 8, 16]
    warmup, seeds = max(contexts) - 1, [618, 619, 620]
    observations = {'full': np.eye(2), 'partial': np.array([[0.0, 1.0]])}  # partial: see e_2 only
    summary, all_rows = {}, []
    for obs, C in observations.items():
        for data, excite in [('narrow', 0.0), ('excited', 0.3)]:
            tag = f'{obs}_{data}'
            rows = part2_framework(a.mu, C, contexts, executions, a.horizon, a.steps + warmup,
                                   warmup, a.noise, excite, seeds, out, tag)
            M = heatmap(rows, contexts, executions, out / f'grid_{tag}.png',
                        f'{obs} obs, {data} data (mu={a.mu})')
            summary[tag] = M.round(4).tolist()
            all_rows += rows
    write_csv(all_rows, out / 'framework_sweep.csv')
    json.dump(dict(mu=a.mu, noise=a.noise, steps=a.steps, horizon=a.horizon, contexts=contexts,
                   executions=executions, static=static, median_late_rms=summary),
              open(out / 'summary.json', 'w'), indent=1)

    print('Static controllers on plant 1 (all match expert data exactly):')
    for r in static:
        print(f"  {r['controller']:13s} rho(A1+K)={r['rho_plant1']:.3f} rho(A2+K)={r['rho_plant2']:.3f} "
              f"final ||x||={r['final_norm']:.3g}")
    for tag, M in summary.items():
        print(f'\n{tag}: median late RMS (rows r={contexts}, cols ell={executions})')
        for r, row in zip(contexts, M):
            print(f'  r={r}: ' + '  '.join(f'{v:9.3g}' for v in row))


if __name__ == '__main__':
    main()
