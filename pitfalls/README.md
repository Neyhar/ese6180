# Pitfalls hard instance (arXiv:2503.09722, Sec. 4)

`python3 pitfalls/pitfalls.py` writes `pitfalls/results/`.

**System.** The linear core of the paper's "challenging pair" (Def. 4.1), mu = 0.2, B = I. The true plant is A_1. Expert data starts on span(e_2) and never leaves it, so the data cannot show how the expert reacts to e_1.

**Part 1: static controllers.** All four gains fit the expert data exactly (same e_2 column).

| controller | rho(A1+K) | rho(A2+K) | outcome on plant 1 |
|---|---|---|---|
| expert K1 (good) | 0.60 | 2.15 | stable |
| wrong K2 | 2.15 | 0.60 | blows up |
| minimax K_hat | 1.075 | 1.075 | blows up (best possible worst case) |
| min-norm least squares | 1.20 | 0.95 | blows up |

**Part 2: the repo's history/chunk framework** (`model.fit_policy` / `model.rollout`). Sweep context r over {1,2,3,4,6,8}, execution ell over {1,2,4,8,16}, prediction horizon 16, process noise sd 0.01. Two observation settings (full state, or e_2 only) and two data settings (narrow = paper support; excited = add e_1 spread to initial states). Heatmaps are in `results/grid_*.png` and raw numbers are in `framework_sweep.csv`.

Takeaways (median late RMS of ||x||; for reference, the expert achieves about 0.015):
- With narrow data, a Markov policy that replans every step (r=1, ell=1) diverges, as the paper predicts.
- **Chunking helps.** On narrow data the learned chunk applies zero action along e_1. During open-loop execution, A_1's own stability (|eig| = 0.9) shrinks e_1. At ell=16, r=1 the RMS is 1.3 instead of divergence.
- **Context "helps", but only by giving up control.** At r >= 6, every ell stays bounded at about 0.1, roughly 7x the expert's error. See the next section for why.
- Excited data recovers the expert (0.015) only at r=1. The expert sets x_1 to zero after one step, so the e_1 spread exists only at t=0. Every training window with r >= 2 starts at t >= 1 and has x_1 = 0. For the learner, that data is the same as narrow data.

## Why long context looks stable (`python3 pitfalls/analysis.py`)

That script computes the exact closed-loop spectral radius of the learned ell=1 history policy in a lifted state, a stack of the last r states and actions.

| r | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| rho(closed loop) | 1.200 | 1.139 | 1.090 | 1.053 | 0.9996 | 0.967 |
| policy weight norm | 0.30 | 0.15 | 0.09 | 0.05 | 0.02 | 0.01 |

The mechanism:
1. Narrow expert windows all lie on a single line (feature rank 1). Each trajectory is just a scaled copy of one geometric sequence along e_2.
2. Ridge/least squares picks the minimum-norm solution. It spreads the weight over all r lags and puts most of it on the oldest ones, because older states are larger by a factor of 1/0.6 per step. The total gain shrinks roughly like 1/r.
3. A_1 alone is stable (rho = 0.9) only because of its off-diagonal coupling c. The expert's e_2 column (u_1 = -c x_2) cancels that coupling and exposes the 1+mu mode. A learner that copies the e_2 column at full strength is unstable. A learner that copies it weakly and late barely cancels the coupling.
4. So at r >= 6 the learned policy is close to doing nothing. Its late RMS of about 0.1 matches the open-loop noise floor of A_1 (0.103). Long chunks (ell=16) help for the same reason: the open-loop stretches let A_1's own stability act.

**Conclusion:** in this instance, longer context and longer chunks don't recover the expert. They fall back to the stable open-loop plant. This is consistent with the paper's remark that chunking helps because of open-loop stability. Any controller that copies the expert's e_2 behavior while ignoring e_1 is worse than doing nothing.

## The paper's own controllers (`python3 pitfalls/controllers.py`)

Page numbers are the ones printed in arXiv:2503.09722 v4. The paper's lower bound (Thm 1; Def. 3.3, p. 11) covers only **simple** policies: smooth, Markovian, and either deterministic or stochastic in a state-independent way. Sec. 5.2 (pp. 18–19) gives one controller that breaks each condition. Appendix H (Prop. H.1, pp. 91–93) shows that all three beat the Thm 1 construction.

### A. Scalar toy, Eq. (5.1), p. 18: x_{t+1} = ξρx_t + u_t, sign ξ unknown (ρ = 1.5, ε = 1e-3)
Figure: `results/controllers/scalar_controllers.png`

| controller | rule (page) | breaks | worst-case max error |
|---|---|---|---|
| any smooth linear u=kx | (ρ+k)^t or (ρ−k)^t, p. 18 | none | 1 (diverges) |
| action-switching | u = −ρx on odd t, +ρx on even t, p. 18 | Markovian (uses t) | 0.003, then x=0 by t=3 |
| adaptive | u_t = −((x_2−u_1)/x_1)·x_t, p. 18 | Markovian (uses history) | 0.0015, then x=0 |
| Benevolent Gambler's Ruin | u = ±ρx with prob. ½ each, pp. 18–19 | simple stochasticity | 0.012 (mean); P[x≠0 at t=10] ≈ 0.1% |
| concentric stabilization | sign of ±ρx alternates on rings \|x\| ∈ ((2ρ)^{−2j}, (2ρ)^{−2(j−1)}], p. 19 | smoothness | 0.003, then x=0 by t=3 |

For BGR, the peak error scales like ε^0.645 in simulation. Our derivation gives ε^{log 2 / log 2ρ} = ε^0.63. That is sublinear in ε, but it never grows exponentially in H. (The PDF text for this exponent on p. 19 is garbled, so we re-derived it.)

### B. The challenging pair, Def. 4.1 p. 15, with μ = 1/8 as in Appendix J p. 98
Start with ε = 1e-3 of learner error on e_1 and x_2 = 0.5. Report the worst case over the true plant i. Figure: `results/controllers/pair_controllers.png`

Appendix H builds every learner as follows. Keep the expert's e_2 column, K_1(I−e_1e_1ᵀ)x (Eq. H.1, p. 92). Then add a base policy that cancels the e_1 column of one candidate plant. Because −A_j e_1 = K_j e_1, that base policy just means "play expert j's e_1 response". (Pages 92–93 print (I−e_1e_1ᵀ) in the base policy; we treat that as a typo for e_1e_1ᵀ.)

| controller | source | worst max \|x_1\| | ends at |
|---|---|---|---|
| expert K_i (knows i) | Def. 4.1 | 0.001 | 0 |
| smooth Markov, min-norm (what BC learns) | Prop. 4.1(b), p. 15 | 1 (diverges) | 1 |
| smooth Markov, best possible | Prop. 4.1(b) bound | 0.016, still growing as (1+3μ/8)^t | growing |
| zero control | Remark 4.1, p. 15 | 0.57 | 0.13, decaying |
| action-switching (alternate K_1e_1, K_2e_1) | Prop. H.1(c), p. 93 | 0.0021 | 0 by t=3 |
| BGR (random K_je_1) | Prop. H.1(b), p. 93 | 0.0019 | 0 eventually |
| concentric (rings in \|x_1\|) | Prop. H.1(a), p. 93 | 0.0044 | 0 by t=3 |
| adaptive (identify i from one step) | Sec. 5.2.1 idea, p. 18 | 0.0021 | 0 |
| **linear, 1 step of memory** (ours) | — | 0.00105 | 1e-18 |
| BC chunks, ell=4 / ell=16, open loop | Sec. 5.1, pp. 17–18 | 0.14 / 0.0053 | still growing, slowly |

What this shows:
1. **Every non-simple controller fixes the problem**, as Prop. H.1 says. Each one tries both candidate e_1 responses. The right one zeros x_1, and once x_1 is zero it stays zero, because span(e_2) is invariant for both closed loops.
2. **Chunking helps but is not a fix** in our linear-BC version. Each open-loop stretch lets A_1's own stability act, so the growth rate drops, but x_1 is never actively zeroed. The paper also reports chunking only as an empirical improvement (Fig. 2, p. 17).
3. **For our project, a smooth linear policy with context r=2 exists that stabilizes both plants** and matches the expert exactly on V: u_1 = −c x_2 − 0.078 x_{1,t} − 0.274 x_{1,t−1}. Worst-case closed-loop spectral radius: 0.52. This does not contradict the paper. Thm 1 needs Markovian policies (p. 12), and with memory this becomes simultaneous stabilization of two scalar plants (cf. the gap-metric remark, p. 15). The policy is in our framework's hypothesis class, so **context removes the class-level obstruction**. But expert data on V cannot select it: least squares sets the e_1 weights to zero, which is the failure we saw in the sweep. Context helps only if the data or the training objective pushes the fit toward this solution.

## Can training find the context-2 stabilizer? (`python3 pitfalls/train.py`)

The policy is linear in the repo's history feature, with full state and μ = 1/8. Each fit is scored by its exact closed-loop spectral radius on **both** candidate plants. Rollouts start from an e_1 error of 1e-3, after an expert warm-up.

| regime | r | train MSE | rho(A_1) | rho(A_2) | worst peak \|x_1\| |
|---|---|---|---|---|---|
| narrow (paper's data) | 1–4 | ~1e-20 | 1.13 → 1.07 | 0.97 | 1 → 0.22 |
| DART, plant 1 only | 1–4 | ~1e-16 | 0.75 | **2.09** | 1 |
| DART, both plants | 1–4 | 1.4e-3 | 1.03–1.05 | 1.05–1.07 | 0.10–0.25 |
| robust objective (narrow data + penalty on worst rho) | 1 | 3.9e-7 | 0.94 | 0.94 | 0.013 |
| robust objective | 2 | **1e-25** | **0.78** | **0.79** | 0.0011 → 1e-11 |
| robust objective | 3 | 1e-25 | 0.75 | 0.78 | 0.0013 → 1e-12 |

Reading:
- **Narrow data:** least squares fits perfectly but is unstable on plant 1 at every context length. More context only slows the divergence (see the "Why long context looks stable" section above).
- **DART noise from one plant:** the data now pins down the true expert K_1. The fit is great on plant 1 and blows up on plant 2 (2.09 ≈ 2 + 3μ/4, Lemma 2.2). Better data recovers *the* expert, but that doesn't make it robust.
- **DART noise from both plants:** this is the realistic multi-environment BC setup. In principle, history reveals which plant is running. But the identifying map, the ratio (x_t − u_{t−1})/x_{t−1} from p. 18, is nonlinear. A linear fit averages the two experts and is mildly unstable on both at every r. Context doesn't help a linear learner here.
- **Robust objective:** keep the same narrow data and add `lam * max(0, rho_worst - 0.8)^2` to the squared loss. At r ≥ 2 the optimizer finds a policy that **matches the expert data exactly** (MSE 1e-25) and is **stable on both plants** (rho ≈ 0.78). That is the stabilizer from the paper-controllers section. At r = 1 the best it can do is trade exact fit (MSE 4e-7) for stability, as Prop. 4.1(b) says it must: an exact Markov fit has rho ≥ 1 + μ/4.

**Bottom line for the project:** context makes a robust, data-consistent linear policy *exist*, but neither more data nor plain BC finds it. Something has to prefer it: a stability or robustness term in the objective, or a nonlinear learner that can do the identification step. Caveat: the robust objective uses knowledge of the two candidate plants, which a real learner wouldn't have.
