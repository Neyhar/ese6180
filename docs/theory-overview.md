# One question, two mechanisms

Can a controller use more observation/action history to reduce its need for
executing several actions before replanning?

1. **Information:** On a partially observed linear system, enough history can
   recover action-relevant state. The observability index h is sufficient for
   noiseless full-state reconstruction when dynamics and applied inputs are
   known. `observability/` studies this mechanism with continuing test noise.
2. **Feedback selection:** Restricted expert histories leave many policies
   observationally equivalent. More history changes which feedback extension
   ridge regression selects. `pitfalls/` studies this on the 2D challenging pair,
   where the full state is already observed and h=1.
3. **Commitment:** Executing an action chunk changes the closed-loop dynamics.
   It can suppress destabilizing learned feedback, but postpones disturbance
   correction. Longer execution need not improve every learned controller.

The active four-page synthesis is `midterm-report/main.tex`. The longer notes
in `theory-notes/main.tex` derive reconstruction, exact noiseless chunks,
non-identifiability, boundary stability, covariance formulas and local nonlinear
extensions. The latter analyses are not all implemented or experimentally checked.

Read `observability/results/latest/RESULTS.md` for numerical claims generated
from the saved sweep, and `pitfalls/results/context_chunking/context_summary.csv`
for the current hard-example criterion. These protocols use different losses
and disturbances. A per-chunk spectral radius and its per-step equivalent
radius^(1/ell) are different quantities; only the latter is directly comparable
across execution lengths.

No theorem here says that an arbitrary fitted learner improves with history.
Observability concerns available information; deployment guarantees additionally
require coverage, conditioning and control of learner error or feedback gains.
