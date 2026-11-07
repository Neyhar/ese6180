# A coupled n-dimensional hard pair

This is a new construction inspired by the Pitfalls example. It is not a
repetition of the original two-state blocks. Implemented dimensions: 2, 4, 8, 16.

## One feedback loop through all n coordinates

Use B=I and observe only y=x_n. Plant A1 is

\[
\begin{aligned}
x_{1,t+1}&=1.02x_{1,t}+0.04x_{n,t}+u_{1,t},\\
x_{2,t+1}&=-x_{1,t}+u_{2,t},\\
x_{j,t+1}&=x_{j-1,t}+u_{j,t},\quad j=3,\ldots,n.
\end{aligned}
\]

Thus x1 affects x2, which affects x3, continuing to the observed xn. The observed
state feeds back to x1, completing a single n-state loop. At n=4,

\[
A_1=\begin{bmatrix}
1.02&0&0&0.04\\
-1&0&0&0\\
0&1&0&0\\
0&0&1&0
\end{bmatrix},\qquad
A_2=\begin{bmatrix}
-0.995&0&0&0.04\\
0&0&0&0\\
0&1&0&0\\
0&0&1&0
\end{bmatrix}.
\]

For every dimension, A2 replaces A1's first column by -0.995 e1. This breaks the
loop and reverses the first-coordinate dynamics. A1's characteristic polynomial
is z^n-1.02 z^(n-1)+0.04. Exact-rational Schur recursion certifies stability at
n=2,4,8,16; numerical spectral radii are approximately 0.979148, 0.977124, 0.970767, and 0.964764.
A2 has eigenvalues -0.995 and zero and is also stable. There is no claim of A1
stability at every possible n with these fixed coefficients.

## Two experts that demonstrations cannot distinguish

Set F=diag(0,...,0,0.75) and K_i=F-A_i. Then A_i+K_i=F. Both experts eliminate
every hidden coordinate in one step and shrink xn by 0.75 per step.

Demonstrations start on S=span(e_n). Both experts choose exactly

\[
K_i(ze_n)=-0.04z e_1+0.75z e_n.
\]

Their state trajectories are exactly 0.75^t z e_n and their action trajectories
are identical for every starting amplitude z. Unlimited demonstrations on this
line cannot distinguish the two plants or the appropriate off-line expert.

However,

\[
A_1+K_2=F+(2.015e_1-e_2)e_1^\top,\qquad
A_2+K_1=F-(2.015e_1-e_2)e_1^\top.
\]

These have eigenvalues +2.015 and -2.015 respectively. A small departure in x1
grows under the wrong expert, while the correct expert removes it. Thus both
cross-applied experts are unstable, despite exact demonstration agreement.

## Why increasing dimension changes the information problem

The scalar observation is xn. For k=0,...,n-2, C A1^k selects x_(n-k). The next
row selects -x1. Consequently O_n is a signed permutation matrix: rank(O_r)=r
for r<=n, and condition(O_n)=1. The observability index is n, not two.

A departure in x1 first affects the output after n-1 transitions under identical
applied inputs. K1 and A1 are invertible, so the exact expert-action recovery
condition ker(O_r) subset ker(K1 A1^(r-1)) also first holds at r=n. This is a
real information delay, with no ill-conditioned reconstruction matrix to explain
away the result. Unknown initial jumps still invalidate an old history buffer
until enough post-jump observations have been collected.

A memoryless linear policy that exactly fits the demonstrations must use
L=-0.04e1+0.75en and u=L*y. On A1 this cancels the feedback from xn to x1,
leaving eigenvalue 1.02. Thus even this simple fitted extension is unstable.
The correct full-state expert remains stable. Zero control is also stable,
so insufficient history is not an impossibility theorem for stabilization.

## What the simulations test

`high_dimensional.py` runs narrow and recovery datasets with histories from 1
through n+1, plus shared absolute histories 4, 8, and 16. Prediction horizons
are 1 and 16; execution horizons are 1, 2, 4, 8, and 16. All dimensions use 128
training trajectories per seed, 100 steps per trajectory, five training seeds,
and the same training target times and expert warm-up. Forty test amplitudes
are matched across dimensions; support, farthest-hidden, and distributed-hidden
cases give 80 perturbed rollouts per deployment. Tracking is scored over 200
steps, with selected 1000-step diagnostics retained separately.

The main analysis uses only x1 changes and two mean-error plots. Additional
saved disturbance types are optional exploratory diagnostics.

The numerical work reuses the shared matched ridge fitter, batched rollout, and
exact augmented-history stability routines. The construction, experiment, and
analysis live in separate modules. The [generated results guide](../results/coupled_higher_dimensions/README.md)
links to per-dimension grids, temporal errors, and cross-dimension comparisons.
Run `python3 -m pitfalls --plot-only` to audit and regenerate
these figures without retraining. Original source snapshots and hashes are saved.

The construction proves representability and expert ambiguity. It does not
prove that ridge regression identifies the expert, that history monotonically
improves loss, or that larger execution always helps. Saved seed-level losses,
spectral radii, raw test episodes, and observation ranks support those empirical
comparisons. More state variables now mean both a longer information delay and
more unexcited training directions; the experiment is not dimension-only scaling.


## Original n=2 reference and plot terminology

The coupled-family n=2 matrices above are not the original Simchowitz matrices.
The runner also evaluates the original pair at mu=1/8, using the same seeds,
training data counts, target times, warm-up, and held-out states as the coupled
n=2 run. Its separate outputs are under `results/coupled_higher_dimensions/original_n2/`.
Dimension-scaling curves connect only the coupled family; the original pair is
shown as a separately labeled reference.

Noiseless training data correspond to expert-only collection. Noisy training
data add Gaussian action perturbations with standard deviation .05 and retain
exact expert labels. These are changes to data collection, not test-time noise.
Internal CSV identifiers narrow/recovery remain for backward compatibility.
