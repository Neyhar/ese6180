# Challenging pair: paper example and history/chunk extension

Run from the repository root:

```sh
python3 pitfalls/pitfalls.py
python3 pitfalls/analysis.py
python3 -m unittest discover -s tests -p 'test_pitfalls.py' -v
```

The main command writes `pitfalls/results/aligned/`. It implements the **2D linear challenging pair in Definition 4.1 (printed page 15)** of [Simchowitz, Pfrommer and Jadbabaie](https://arxiv.org/html/2503.09722v4#S4). This is not a reproduction of the paper's Appendix J neural-network experiments, which use the higher-dimensional nonlinear construction.

## 1. Reproduce the ambiguity on one fixed plant

The default is mu=1/8. Both plants are stable; both matched expert closed loops equal diag(0, 1-2mu). On span(e2), both experts and their trajectories are indistinguishable.

`hard_example.png` compares K1, K2, the exact-fit minimax gain, and minimum-norm BC on **plant A1**. One panel starts exactly on the demonstration support, where all agree. The other introduces x1=0.001 at initialization, with no continuing noise. The wrong controller and minimum-norm fit amplify this departure. No noise is required after the initial perturbation. Raw, unclipped values are in `hard_example.csv`.

## 2. Apply the input-output history framework

For each context r, fit a linear policy on [y_(t-r+1),...,y_t,u_(t-r+1),...,u_(t-1)] using the actual past actions. Hold the prediction horizon H=16 fixed and vary execution ell in {1,2,4,8,16}. The same fitted policy is used for all ell at a given r.

- Full observation: C=I. Memory changes the controller dynamics; the current state is already observed.
- Partial observation: C=[0,1]. Plant A1 is observable with two noiseless observations and known inputs. Process noise prevents exact finite-history reconstruction.
- Narrow data: noiseless expert demonstrations on span(e2), reproducing the information restriction.
- Initial excitation: vary the initial x1. This adds information (rank typically 2 rather than 1), including in history windows, but does not cover arbitrary off-expert histories.
- Persistent excitation: add process disturbances of sd=0.05 throughout demonstrations, labeling visited states with K1. This changes coverage and is an extension of the narrow-data example. Future chunk targets include subsequent noisy expert responses; increasing execution length therefore also delays disturbance feedback.

Five independently trained datasets use seeds 618--622. Each gets one independent evaluation noise sequence (sd=0.01), paired across every observation/data/context/execution setting. All policies receive the same seven-step noiseless expert warm-up. This is an evaluation convention, not a deployable initialization method. Deployment lasts 200 steps after warm-up.

`framework_sweep.csv` reports every seed, feature rank/dimension, training MSE, full-state tracking RMS against a separately simulated expert receiving the **same disturbances**, action RMS against K1 at the learner's state, late state RMS, and peak |x1|. Values are not clipped. The heatmaps show median tracking RMS; an asterisk marks any seed with an unstable closed loop. Color limits are display limits only. Five seeds are descriptive evidence, not a significance claim.

## 3. Distinguish finite rollout performance from stability

`aligned.boundary_maps` constructs the exact boundary-to-boundary matrix for each (r, ell). Actions within each chunk remain frozen at their boundary predictions. The CSV reports both its spectral radius and the per-step equivalent radius^(1/ell). Compare the latter across execution lengths.

For stable policies, a discrete Lyapunov equation gives stationary state RMS under iid noise, averaged across all chunk phases. It is left undefined for unstable policies. This is state regulation, not stationary expert tracking. Finite late-window RMS is not called a noise floor: a radius near one can take thousands of steps to settle. For example, the former mu=0.2, narrow r=6, ell=1 result has radius about 0.9996 and stationary state RMS about 0.93, far above its 200-step estimate. r=8 is closer to the zero-controller baseline. Small policy weight norm alone does not establish equivalent control behavior.

`baselines.csv` evaluates the full-state expert, zero control, and an analytic memory witness with the same warm-up, initial state, and evaluation disturbances. The analytic witness uses model knowledge and is not learned from demonstrations.

## Analytic one-step-memory witness

Let a1=1+mu, a2=-(1-mu/4), c=3mu/2. Set

    k0 = -(a1+a2)/2
    k1 = -((a1-a2)/4)^2
    u1_t = -c*x2_t + k0*x1_t + k1*x1_(t-1)
    u2_t = 0

This matches both experts on the demonstrated subspace. The x1 characteristic polynomial is lambda^2-(ai+k0)*lambda-k1; its roots are +(a1-a2)/4 on plant 1 and -(a1-a2)/4 on plant 2, each repeated. Include the remaining x2 eigenvalue 1-2mu when reporting full-system stability. At mu=1/8, the x1 radius is 0.5234375 and the **full radius is 0.75** on both plants.

This witnesses representability, not identification from narrow data. Ordinary least squares and a model-informed stability objective need not choose the same off-subspace behavior. It does not prove a monotone context/execution tradeoff.

## Additional exploratory scripts

`controllers.py` retains the scalar and 2D non-simple-controller comparisons. Its metrics are explicitly **clipped** expectations E[min(1,|x1|)], not full-state tracking. The memory witness is now analytic and reports the full radius. Its Monte Carlo BGR scaling fit is exploratory: rare long-survival paths are poorly sampled at small epsilon; it is not an exponent estimate with established precision.

`train.py` compares narrow BC, noisy single-plant/pooled demonstrations, and a model-informed spectral penalty. Its robust objective knows both candidate plants, and its rollout summaries clip |x1| at one. These results establish numerical feasibility and selection behavior, not a guarantee for unknown plants or all linear-learning procedures. The main aligned experiment remains the fixed-plant context/chunk study requested for this project.
