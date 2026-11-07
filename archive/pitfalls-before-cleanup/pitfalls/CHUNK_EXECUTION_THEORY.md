# Theory and interpretation

## Paper construction

`challenging_pair` uses Definition 4.1 of Simchowitz, Pfrommer and Jadbabaie,
[The Pitfalls of Imitation Learning when Actions are Continuous](https://arxiv.org/html/2503.09722v4#S4).
For x(t+1)=A_i x(t)+u(t), the matched experts have A_i+K_i=diag(0,1-2mu).
Expert trajectories started on x1=0 stay there, and both candidate instances
yield the same actions and trajectories on that subspace. The data therefore
cannot identify recovery behavior in the x1 direction. Proposition 4.1 gives
an instability obstruction for a data-matching static linear controller on at
least one candidate plant. It is not a statement that every controller fails
on plant A1. Our sweep fixes A1 and explicitly injects a small initial departure;
it isolates the linear mechanism, not the paper's full lower-bound construction
or Appendix J neural-network experiments.

## Learned policy

Features concatenate r observations and r-1 actual past actions. Ridge regression
minimizes ||XB-Y_H||_F^2 + 1e-6 ||B||_F^2 to predict H expert actions. This is a
linear behavioral-cloning baseline, not a stability-aware learning objective.
The penalty chooses small coefficients among underdetermined fits; it does not
supply information about unvisited histories or guarantee stability.

Output columns are fitted independently. Matched input windows therefore imply
that larger H leaves the earlier action predictions unchanged, up to numerical
precision. At fixed executed prefix ell, curves for all H>=ell should overlap.
Do not interpret this as a general result about chunk-trained neural policies.

## Context versus execution

Executing ell>1 commits to an action prefix and postpones feedback; ell=1
replans every step. Different ell values change the closed-loop matrix even
when the same prediction model is used. Longer context expands the policy
class, but the narrow demonstrations do not identify every feature direction.
A model-informed memory controller can stabilize both plants, demonstrating
that a suitable controller exists without implying ridge regression finds it.

The hypothesis is tested at the largest H: for each r, find the shortest tested
ell for which every training seed has mean perturbed-test loss below the fixed
threshold and exact spectral radius below one. A decreasing ell supports the
hypothesis within the grid; ell=1 means no multi-action execution is required.
A constant one has no room to improve. None means no tested setting passes,
not impossibility. This criterion uses means over test states, not a guarantee
for every state or a population-level confidence statement.

## Stability and long-term behavior

The augmented state contains the current/past plant states and past actions.
The exact chunk-boundary map M_ell is computed with actions frozen at their
boundary predictions. rho(M_ell)<1 establishes asymptotic decay in these linear
noiseless dynamics. Use rho(M_ell)^(1/ell) to compare execution lengths.

Finite-horizon tracking MSE can be small despite instability, especially with
small initial departures. The long canonical rollout is a diagnostic; exact
spectral stability is the long-term check. Small ridge weights or low training
MSE cannot replace it. Stable does not necessarily mean accurate imitation.

The plant itself is stable without control, so longer memory may eventually
approximate weak feedback rather than expert recovery. Interpret improvements
against the expert and analytic memory baseline, and do not equate reduced
finite loss with successful identification of the expert. Full observation
already reveals the current state: memory's effect here is on controller
selection and dynamics, not recovery of an unobserved state.

## Partial observation: distinguish information from coverage

The optional sensor C=[0,1] observes y=x2. Its observability matrix is
O2=[[0,1],[-c,q]], whose determinant is c>0. With noiseless known-action
histories, reconstruct

    x1_(t-1) = (q*y_(t-1) + u2_(t-1) - y_t)/c,
    x1_t = (1+mu)*x1_(t-1) + c*y_(t-1) + u1_(t-1).

The unannounced x1 jump injected at deployment violates the noiseless history
relation for that transition. It is invisible in y at the first decision;
its effect appears in the next observation, when the reconstruction becomes
valid again. Exactness is mathematical; absolute floating-point reconstruction
errors can grow on divergent trajectories. A partial controller cannot choose
the full-state expert's first corrective action for both indistinguishable
initial histories. The known-model observer is therefore a useful nonzero-loss
baseline, not an oracle with instantaneous hidden-state access.

Narrow expert histories have x1=0, so removing the identically zero x1 feature
leaves the ridge fit's effective action rule unchanged. Full and partial narrow
runs consequently agree, despite different observation capabilities. This is
an identification limitation, not a failure of observability.

The recovery intervention uses known applied probes to cover reconstructible
histories; expert targets are KF^j x at each visited boundary. It creates no
unknown process disturbances, and does not substitute expert labels for past
applied actions. At context two the partial feature vector has rank four in
these runs, enabling the learned rule to approximate the model observer.
These noiseless results do not extend directly to continuing process noise,
measurement noise, or unknown-model reconstruction.


## Historical independent-copy extension (superseded)

`high_dimensional.py` extends the exact K1/K2 ambiguity by direct sums.
For even dimension n=2b, define

\[
 A_i^{(n)}=I_b\otimes A_i,\qquad K_i^{(n)}=I_b\otimes K_i,
 \qquad B^{(n)}=I_n,\qquad C^{(n)}=I_b\otimes[0\ \ 1].
\]

Let S map b visible coordinates into the odd-indexed state coordinates (Python
indexing), with every hidden coordinate zero. The original pair obeys

\[
 (K_1^{(n)}-K_2^{(n)})S=0,\qquad
 (A_i^{(n)}+K_i^{(n)})S=qS,\qquad q=1-2\mu.
\]

Consequently, starting from the same state in im(S), both expert/plant pairs
produce exactly the same states, observations, and actions at every time.
This is equality on the entire demonstration subspace, not just a finite sample.
Both correct closed loops have eigenvalues 0 and q, repeated b times.

In contrast, applying K2 to A1 gives independent blocks

\[
 A_1+K_2=\begin{bmatrix}2+3\mu/4&0\\-3\mu/2&1-2\mu\end{bmatrix}.
\]

Any nonzero hidden departure therefore has a component growing by
2+3mu/4 per step: 2.09375 at mu=1/8. The pair remains indistinguishable from
narrow demonstrations while the wrong expert is unstable on A1. This preserves
the original construction; it is not a new coupled-system lower bound.

Each block's two-row observability matrix is full rank, so the observability
index is still two for every n. Sensor and action counts grow with dimension.
The experiment separates dimension from necessary history, rather than making
state reconstruction require longer history. The learner can nevertheless
cross-couple blocks because its ridge weight matrix is unrestricted.

For cross-dimensional comparisons, hidden test departures have fixed total
Euclidean magnitude, and total tracking MSE is the primary score. Per-coordinate
MSE is retained but divides the same total error by n. Training trajectories
increase with the number of blocks; these runs do not measure fixed-data sample
complexity. Three seeds and noiseless deployment do not establish generalization.


## Current coupled higher-dimensional construction

See [COUPLED_HARD_EXAMPLE.md](COUPLED_HARD_EXAMPLE.md) for the implemented
single-loop pairs. Their scalar-observation indices grow as 4, 8, and 16; both
wrong experts destabilize the other plant. The earlier direct sums above remain
only as a preserved control and are no longer the active implementation.
