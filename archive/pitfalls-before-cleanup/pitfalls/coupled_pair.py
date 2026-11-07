"""The coupled delay-loop hard pair, independent of data collection and plotting."""
from fractions import Fraction
import numpy as np
from model import LinearSystem


def higher_dimensional_pair(dimension):
    """Return A1, A2, K1, K2, C for one n-state feedback loop.

    A1: x1+=1.02*x1+.04*x_n; x2+=-x1; x_j+=x_(j-1), j>=3.
    A2 replaces A1's first column by -.995*e1, breaking the loop.
    Both experts make the closed loop F=diag(0,...,0,.75).
    """
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("Dimension must be an integer >= 2")
    A1 = np.zeros((dimension, dimension))
    A1[0, 0] = 1.02
    A1[0, -1] = .04
    for j in range(1, dimension):
        A1[j, j-1] = -1 if j == 1 else 1
    A2 = A1.copy()
    A2[:, 0] = 0
    A2[0, 0] = -.995
    F = np.zeros_like(A1)
    F[-1, -1] = .75
    return A1, A2, F-A1, F-A2, np.eye(dimension)[-1:]


def exact_schur_certificate(dimension):
    """Certify z^n-(51/50)z^(n-1)+1/25 by exact-rational Schur recursion.

    For real p, reduce (p-k*reverse(p))/z with k=p(0)/leading(p).
    Strict |k|<1 at every reduced degree certifies all roots inside the disk.
    Returns the smallest reflection-coefficient margin; raises on failure.
    """
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("Dimension must be an integer >= 2")
    coefficients = ([Fraction(1), -Fraction(51, 50)] +
                    [Fraction(0)]*(dimension-2) + [Fraction(1, 25)])
    margins = []
    while len(coefficients) > 1:
        reflection = coefficients[-1] / coefficients[0]
        if abs(reflection) >= 1:
            raise ValueError(f"A1 is not Schur stable at dimension {dimension}")
        margins.append(float(1-abs(reflection)))
        reduced = [a-reflection*b for a, b in zip(coefficients, reversed(coefficients))][:-1]
        coefficients = [value/reduced[0] for value in reduced]
    return min(margins)


def spectral_radius(matrix):
    return float(max(abs(np.linalg.eigvals(matrix))))


def verify_pair(dimension):
    """Verify ambiguity on the entire support, both cross-failures, and h=n."""
    A1, A2, K1, K2, C = higher_dimensional_pair(dimension)
    support = C.T
    F1, F2 = A1+K1, A2+K2
    residuals = [np.linalg.norm((K1-K2)@support),
                 np.linalg.norm((F1-F2)@support),
                 np.linalg.norm(F1@support-.75*support)]
    if max(residuals) > 1e-12:
        raise ValueError("Demonstrations distinguish the pair")
    system = LinearSystem(A1, np.eye(dimension), C)
    ranks = [int(np.linalg.matrix_rank(system.observability(r))) for r in range(1, dimension+1)]
    if ranks != list(range(1, dimension+1)):
        raise ValueError("Incorrect observability index")
    # Invertibility makes full observability necessary for exact current-action recovery.
    if np.linalg.matrix_rank(K1) != dimension or np.linalg.matrix_rank(A1) != dimension:
        raise ValueError("Action recovery may have a smaller information threshold")
    radii = [spectral_radius(matrix) for matrix in (A1, A2, F1, F2, A1+K2, A2+K1)]
    if max(radii[:4]) >= 1 or min(radii[4:]) <= 1:
        raise ValueError("Incorrect stability properties")
    expected = np.zeros(dimension+1)
    expected[0], expected[1], expected[-1] = 1, -1.02, .04
    np.testing.assert_allclose(np.poly(A1), expected, atol=1e-10)
    return dict(dimension=dimension, demonstration_subspace_dimension=1, sensors=1,
                observability_index=dimension, action_recovery_index=dimension,
                action_agreement_residual=residuals[0], trajectory_agreement_residual=residuals[1],
                support_invariance_residual=residuals[2], A1_radius=radii[0], A2_radius=radii[1],
                expert_K1_radius=radii[2], expert_K2_radius=radii[3], wrong_K2_radius=radii[4],
                wrong_K1_on_A2_radius=radii[5],
                exact_schur_minimum_margin=exact_schur_certificate(dimension),
                observability_condition=float(np.linalg.cond(system.observability(dimension))))


def experiment_pair(dimension, family="coupled"):
    """Keep the paper's actual 2D example distinct from our coupled extension."""
    if family == "coupled":
        return higher_dimensional_pair(dimension)
    if family != "original" or dimension != 2:
        raise ValueError("The original reference is defined only at n=2")
    from pitfalls import challenging_pair
    A1,A2,K1,K2 = challenging_pair(.125)
    return A1,A2,K1,K2,np.array([[0.,1.]])


def verify_experiment_pair(dimension, family="coupled"):
    if family == "coupled":
        return verify_pair(dimension)
    A1,A2,K1,K2,C = experiment_pair(dimension,family)
    F1,F2 = A1+K1,A2+K2
    np.testing.assert_allclose(F1,np.diag([0.,.75]),atol=1e-14)
    np.testing.assert_allclose(F1,F2,atol=1e-14)
    np.testing.assert_allclose((K1-K2)@C.T,0,atol=1e-14)
    radii=[spectral_radius(x) for x in (A1,A2,F1,F2,A1+K2,A2+K1)]
    if max(radii[:4])>=1 or min(radii[4:])<=1:
        raise ValueError("Original reference failed stability checks")
    system=LinearSystem(A1,np.eye(2),C)
    if system.observability_index()!=2:
        raise ValueError("Original reference has incorrect observability index")
    # Its characteristic polynomial is (z-15/16)^2; exact Schur margins.
    return dict(dimension=2,demonstration_subspace_dimension=1,sensors=1,
                observability_index=2,action_recovery_index=2,
                action_agreement_residual=float(np.linalg.norm((K1-K2)@C.T)),
                trajectory_agreement_residual=float(np.linalg.norm((F1-F2)@C.T)),
                support_invariance_residual=float(np.linalg.norm(F1@C.T-.75*C.T)),
                A1_radius=radii[0],A2_radius=radii[1],expert_K1_radius=radii[2],
                expert_K2_radius=radii[3],wrong_K2_radius=radii[4],wrong_K1_on_A2_radius=radii[5],
                exact_schur_minimum_margin=float(Fraction(1,481)),
                observability_condition=float(np.linalg.cond(system.observability(2))))
