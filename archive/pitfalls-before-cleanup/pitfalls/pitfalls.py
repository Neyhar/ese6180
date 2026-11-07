"""Definition 4.1 challenging pair and narrow expert demonstrations."""
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import demonstrations

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

def train_set(system, K, n_traj, steps, excite, rng):
    """Expert demos. excite=0: initial states uniform on [-e2, e2] (paper's support).
    excite>0: add N(0, excite^2) on e_1 so the learner sees how K_1 reacts to e_1."""
    x0 = np.zeros((n_traj, 2))
    x0[:, 1] = rng.uniform(-1, 1, n_traj)
    x0[:, 0] = excite * rng.normal(size=n_traj)
    return demonstrations(system, K, x0, steps)

def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

if __name__ == '__main__':
    from chunk_sweep import main
    main()
