# Implementation and verification map

| Mathematical object | Implementation | Verification |
|---|---|---|
| Linear plant, expert, observation history | `model.py` | `tests/test_model.py` |
| Noiseless reconstruction and ideal action chunks | `LinearSystem.reconstruction`, `exact_policy` | Arbitrary-input reconstruction and exact tracking tests |
| Matched training windows and separable ridge outputs | `observability/experiment.py` | Independent ridge solve and horizon-prefix tests |
| Paired noisy batched deployment | `observability/experiment.py` | Comparison against scalar rollout, including partial final chunks |
| Sensor changes at fixed plant/expert | `model.coupled_system`, `observability/run.py` | Exact requested-index and fixed-matrix tests |
| Exact observability-study boundary map | `observability/experiment.py` | Map compared with directly simulated committed actions |
| Hard-example construction and recovery tests | `pitfalls/original/system.py`, `pitfalls/original/evaluation.py` | Indistinguishability, paired-state and initialization tests |
| Hard-example boundary maps and analytic memory | `pitfalls/shared/stability.py` | Direct maps, full radius and covariance checks |
| Coupled hard pairs and focused comparisons | `pitfalls/coupled/` and `pitfalls/shared/audit.py` | Exact Schur certificate, growing observability, paired raw-loss audit |
| General history/chunk propositions | `midterm-report/main.tex` | Mathematical arguments; not universal learned-policy guarantees |

## Important conventions

Both studies provide expert warm-up to initialize history; this is an evaluation
choice rather than a learned deployment method. Expert references receive the
same disturbances or initial departures as learners. The observability study
records a numerical failure after state/action norm exceeds 1e6; this cutoff
and spectral instability answer different questions. Its score is per-coordinate
late error. Pitfalls' main score is whole-rollout full-state error.

## Not yet implemented or validated generally

- Joint learner/expert stationary covariance prediction of tracking error in the observability study.
- Direct measurement of within-chunk disturbance-response drift.
- A learned observer with a deployment-error guarantee or a conditioning/error analysis.
- Nonlinear plants and the local nonlinear claims in the extended notes.
- A statistical theorem guaranteeing history reduces the required execution length.

Use generated `observability/results/latest/RESULTS.md` as the numerical source;
this file deliberately does not repeat numbers that can drift out of sync.
