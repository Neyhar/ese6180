"""Definition 4.1 original two-dimensional challenging pair."""
import numpy as np


def challenging_pair(mu=0.2):
    if not 0 < mu <= 0.5:
        raise ValueError('mu must be in (0, 1/2]')
    c = 1.5 * mu
    A1 = np.array([[1 + mu, c], [-c, 1 - 2 * mu]])
    A2 = np.array([[-(1 - mu / 4), c], [0, 1 - 2 * mu]])
    K1 = np.array([[-(1 + mu), -c], [c, 0]])
    K2 = np.array([[1 - mu / 4, -c], [0, 0]])
    return A1, A2, K1, K2

