# Theory write-up: history, action chunks, and stability

A plain-language summary of `theory-notes/main.tex`, for a reader who has not seen the project.

## The question

A learner imitates an expert controller. The expert sees the full state of a linear system; the learner sees only partial sensor readings. The learner gets two design choices:

- **History length r**: how many past observations (and the actions between them) it uses.
- **Execution length ℓ**: how many actions from a predicted chunk of H actions it applies before replanning.

How do these two choices affect whether the learned controller is stable and how closely it tracks the expert?

## Setup

```
x_{t+1} = A x_t + B u_t + w_t      (state, action, disturbance)
y_t     = C x_t                    (partial observation)
u*_t    = K* x_t                   (expert)
```

The learner's input is `z_t = (y_{t-r+1}, …, y_t, u_{t-r+1}, …, u_{t-1})`, and it outputs a chunk of H actions with one matrix: `(û_t, …, û_{t+H-1}) = W z_t`. W is fit by ridge regression on expert demonstrations.

## The argument in seven steps

1. **History can replace the state.** Without noise, if `r ≥ h` (the observability index), the state is an exact linear function of the history: `x_t = M_r z_t`. So "reconstruct, then apply the expert" is representable as one matrix. Recovering the *action* can need less than full state recovery: what matters is that every state difference the history cannot see is also irrelevant to the expert's action.

2. **Perfect prediction makes execution length irrelevant.** With the ideal weights `W_j = K* F^j M_r` (where `F = A + BK*`) and no noise, every ℓ reproduces the expert exactly. Any difference between execution lengths must come from disturbances or fitting error.

3. **Demonstrations do not pin down the policy.** Expert histories lie in an n-dimensional subspace of the much larger history space. Many W fit them equally well and differ elsewhere, on histories created by disturbances or by the learner's own actions. Low validation error does not imply closed-loop stability.

4. **Too-short histories can cheat through past actions.** With r < h, observations alone cannot recover the state, yet the fit can still be perfect: on expert data, past actions `K* x` encode the full state. The regression learns to read the state from its action inputs. At deployment those inputs are the learner's own actions, the shortcut breaks, and the loop can go unstable.

5. **Committing has a cost.** During a chunk, a disturbance grows under the uncorrected dynamics A rather than the expert's F. The expected action error grows with each step into the chunk. This favours short ℓ *when the policy is otherwise accurate*.

6. **One matrix decides stability.** Add the history buffer to the state. Looking only at replanning times, the closed loop is a fixed linear map:

   ```
   s_{k+1} = 𝓜_{r,ℓ} s_k + noise,   𝓜_{r,ℓ} = D^ℓ + Σ_j D^{ℓ-1-j} E W_j P
   ```

   The learned controller is stable exactly when the spectral radius `ρ(𝓜) < 1`. `ρ^{1/ℓ}` is the per-step decay rate, comparable across ℓ. Changing r changes both W and the buffer; changing ℓ changes how the fitted rows combine. Nothing forces a monotone trend in either.

7. **Stable is not the same as accurate.** Under ongoing noise, the error settles to a covariance given by a Lyapunov equation. A stable loop with ρ close to 1 retains much more noise. The appendix gives the exact expected tracking error, including the correlation from learner and expert sharing the same disturbances.

## Extension to nonlinear systems

The same structure carries over locally. Reconstruction becomes an error budget (estimation error × how fast the expert's action changes with state, plus fitting error). The chunk boundary map becomes a nonlinear map whose Jacobian at an equilibrium has exactly the linear formula for 𝓜, so `ρ(Jacobian) < 1` gives local exponential stability. Bounded disturbances stay bounded when they are small relative to the contraction margin. These results are local and do not directly cover Gaussian noise.

## What the experiments show

Full sweep (`python run_experiment.py --overwrite`, November 3 run): 20 (n, h) cases, 3 seeds, 1095 chunk configurations.

- **Stability test vs simulation.** ρ(𝓜) ≥ 1 in 30 configurations. 21 of them crossed the 1e6 failure cutoff within 600 steps. The other 9 did not, but their error grows exponentially (for example n=50, h=8, r=7, ℓ=8: error 2e-7 at step 0, 2e5 at step 500, 1.6e7 at step 599); they are slow divergences the cutoff had not reached yet. No configuration with ρ(𝓜) < 1 diverged. The spectral test therefore detects every failure the rollout detects, plus slow ones it misses.
- **Where instability occurs.** All 30 unstable configurations have r < h. With r ≥ h (600 configurations), the per-step rate ρ(𝓜)^{1/ℓ} equals the expert's ρ(F) to within 1e-10 for every ℓ: once the history is long enough, the learned loop has the same spectrum as the expert loop.
- **Execution length when r = h.** One-step execution has the lowest mean tail error in 19 of 20 cases (the exception is n = h = 100, ℓ=2, by 0.12%), consistent with the drift result: with a correct loop, fresh feedback only helps.
- **Short history.** For n=10, h=4, r=3, ρ(𝓜) = 1.32, 0.985, 2.35, 2.99, 0.884 for ℓ = 1, 2, 4, 8, 16. Exactly ℓ = 1, 4, 8 diverge. The pattern is not monotone in ℓ.

## What is established and what is not

- **Derived:** reconstruction, ideal-chunk equivalence, non-identifiability, the drift formula, the boundary-matrix stability test, the covariance equations, and the local nonlinear results. These are standard control facts specialized to this setup.
- **Checked numerically:** the boundary-matrix test against simulated divergence.
- **Not yet checked:** the covariance prediction of tracking error, and anything nonlinear.
- **Scope:** one plant family, three seeds, finite runs. These are not universal laws.
