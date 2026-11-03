# Code and its relation to the theory

This maps each piece of the codebase to the result in `theory-notes/main.tex` it implements or tests, and lists where the two differ. Short answer: **the code matches the theory**. Every derived formula used by the experiments has a direct implementation, several are enforced by assertions or unit tests, and the remaining differences are evaluation conventions or analyses not yet implemented, not contradictions.

## Map from theory to code

| Theory (notes section) | Code | How the match is checked |
|---|---|---|
| Plant `x+ = Ax + Bu + w`, `y = Cx`, expert `u* = K*x` (§1) | `LinearSystem.step`, `observe` in `model.py` | Disturbance added after the action, exactly as in the notes |
| History input `z_t = col(y_{t-r+1..t}, u_{t-r+1..t-1})`, `d_r = rp + (r-1)m` (§2) | `history_feature`; vectorized copy in `batch_rollout` | All observations first, then actions; `HistoryPolicy.weights` has shape `(Hm, d_r)` |
| Observability matrix `O_r`, index `h` (§3) | `observability`, `observability_index` | `run_case` asserts the measured `h` equals the requested one |
| Reconstruction `x_t = M_r z_t` with `T_r`, `R_r` (Thm 3.1) | `history_matrices`, `reconstruction` (`pinv` as the left inverse `L_r`) | `tests/test_model.py` |
| Ideal chunk `W_j* = K* F^j M_r` (Prop 4.1) | `exact_policy` | `tests/test_model.py`: exact policy tracks the expert for every ℓ when noiseless |
| Expert histories have rank ≤ n (Prop 5.1) | `fit_diagnostics.csv`: `feature_rank` vs `feature_dim` | Reported for every case |
| Ridge objective with unnormalized λ‖W‖²_F (§5) | `fit_policy` (augmented least squares), `fit_matched` (shared SVD) | Test re-solves the ridge problem independently |
| First chunk block equals the one-step fit (Lemma 5.2) | `fit_matched` | `assert_allclose` on every fit; `run_case` also asserts the ℓ=1 rollouts coincide |
| Matched windows across r and H (§5.1) | `window_ids` called with the largest context for every r | Same `(trajectory, time)` ids for all contexts |
| Commit ℓ actions, observe every step, replan at boundaries (§4) | `rollout`, `batch_rollout` | `batch_rollout` computes only the first ℓ rows, which is identical because unused rows are discarded |
| Separate expert `x*` driven by the same disturbances (§6, Appendix) | `expert_reference` | Same disturbance array passed to learner and expert |
| Per-coordinate tracking error `‖x - x*‖²/n` (Appendix metric) | `trajectory_errors`, `summarize_errors` | Mean over coordinates; tail = final `--tail-steps` |
| Augmented state `s_t`, matrices `D, E, J, P, X` (§7) | `augmented_matrices` in `long_experiment.py` | Test: `P s` equals the real history input, and the buffer update matches a real plant rollout |
| Boundary matrix `𝓜_{r,ℓ}` and test `ρ(𝓜) < 1` (Prop 7.1, Thm 7.2) | `boundary_matrix`, `boundary_radius` | Test against a real rollout; `metrics.csv` columns `boundary_radius`, `per_step_rate` |
| Sweep plant family (used in §9 but only described in code) | `coupled_system`: `A = aΠ`, `K = -g bᵀΠ`, sensors every h-th coordinate | `‖F‖₂ ≤ a < 1` tested; index tested |

## Conventions in the code that the theory treats lightly

- **Expert warm-up.** Every evaluation episode starts with `r_max - 1` expert-controlled steps to fill the history buffer. Scoring and replanning begin afterward. The stability theory is unaffected (it is about the closed loop after warm-up), and §9 now states this.
- **Failure cutoff.** An episode is marked failed when the state or action norm exceeds `1e6`, and its error becomes infinity. This is a numerical cutoff, not an eigenvalue test. In practice the two agree: see the results below.
- **Noise model.** Training is noiseless; evaluation uses i.i.d. Gaussian process noise (SD 0.02 per coordinate) and exact measurements. This matches the linear covariance assumptions. It does *not* satisfy the bounded-disturbance assumption of the nonlinear section (§10.3), which the notes already say.
- **Tail window.** The final 200 steps are a finite-time average. With ℓ=16 they cover 12.5 chunks, so phases are slightly unevenly weighted relative to the appendix's phase average.
- **Hand example vs sweep plant.** `example_system` is the 2-D system of §1. The sweep uses `coupled_system`, a different (cyclic) family.

## Theory not yet implemented in code

- **Covariance prediction** `S = M S Mᵀ + N(I⊗Q)Nᵀ` and the tracking prediction `𝒥(r,ℓ)` of the Appendix. These would predict the tail error itself, not just stability.
- **Phase-drift formula** of §6 (expected action discrepancy grows with phase). Not measured directly.
- **Nonlinear sections** (§10–11). No nonlinear plant exists in the code yet.
- **Conditioning** of the reconstruction is reported (`observability_condition` in `systems.csv`) but not linked to errors.

## Results from the current run

Full sweep (`python run_experiment.py --overwrite`, November 3 run): 20 (n, h) cases, 3 seeds, 1095 chunk configurations.

- **Stability test vs simulation.** ρ(𝓜) ≥ 1 in 30 configurations. 21 of them crossed the 1e6 failure cutoff within 600 steps. The other 9 did not, but their error grows exponentially (for example n=50, h=8, r=7, ℓ=8: error 2e-7 at step 0, 2e5 at step 500, 1.6e7 at step 599); they are slow divergences the cutoff had not reached yet. No configuration with ρ(𝓜) < 1 diverged. The spectral test therefore detects every failure the rollout detects, plus slow ones it misses.
- **Where instability occurs.** All 30 unstable configurations have r < h. With r ≥ h (600 configurations), the per-step rate ρ(𝓜)^{1/ℓ} equals the expert's ρ(F) to within 1e-10 for every ℓ: once the history is long enough, the learned loop has the same spectrum as the expert loop.
- **Execution length when r = h.** One-step execution has the lowest mean tail error in 19 of 20 cases (the exception is n = h = 100, ℓ=2, by 0.12%), consistent with the drift result: with a correct loop, fresh feedback only helps.
- **Short history.** For n=10, h=4, r=3, ρ(𝓜) = 1.32, 0.985, 2.35, 2.99, 0.884 for ℓ = 1, 2, 4, 8, 16. Exactly ℓ = 1, 4, 8 diverge. The pattern is not monotone in ℓ.
