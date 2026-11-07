# ESE 618: can history reduce the need for action chunking?

There are **two complementary experiments**, sharing `model.py`:

| Study | What changes? | What it tells us | Start here |
|---|---|---|---|
| [Observability](observability/README.md) | Partial sensors, state dimension, history, execution | History can supply missing state information; once sufficient, frequent feedback usually helps under noise | [Saved results](observability/results/latest/RESULTS.md) |
| [Pitfalls](pitfalls/README.md) | History, execution, and training coverage on a 2D hard example | Context and corrective coverage can enable one-step feedback | [Focused results](pitfalls/results/coupled_higher_dimensions/README.md) |

These are distinct mechanisms. Their raw losses use different normalizations,
evaluation windows and disturbance protocols, so do not compare their magnitudes.
The active [four-page report](midterm-report/main.tex) focuses on the hard example and its connection to the theory.

## Start with existing results

- **Observability:** [one representative graph](observability/results/latest/figures/start_here.png) and [result summary](observability/results/latest/RESULTS.md).
- **Pitfalls:** [two main comparisons](pitfalls/results/coupled_higher_dimensions/README.md) and [experiment/code guide](pitfalls/README.md).
- **Report:** [main.pdf](midterm-report/main.pdf), built from `midterm-report/main.tex` with the L4DC template.

The [coupled higher-dimensional hard pairs](pitfalls/results/coupled_higher_dimensions/README.md)
now extend the Pitfalls experiment to n=2,4,8,16 with one sensor and growing
observation delay. Their plot index includes cross-dimension and stability analysis.

## Run

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m observability.run --output observability/results/focused
python3 -m pitfalls --plot-only
```

The observability command now defaults to one representative n=10,h=4 case;
the original 20-case output is preserved in `observability/results/latest/`.
`python3 run_experiment.py` forwards to the same runner for compatibility.
See each study's README for parameters and the full-grid command. Tests do not
write result figures unless `ESE618_TEST_PLOTS` is explicitly set.

## Repository map

- `model.py`: shared linear systems, expert demonstrations, history policies, ridge fitting and reconstruction.
- `observability/`: partial-observation experiment and its results.
- `pitfalls/`: full/partial-observation hard-example experiments and results.
- `midterm-report/`: active report and retained synthesis drafts.
- `theory-notes/main.tex`: longer mathematical derivations, including work not yet simulated.
- `docs/theory-overview.md`: concise theory map; numerical claims live in generated result summaries.
- `docs/code-and-theory.md`: implementation map and unimplemented analyses.
- `tests/`: mathematical checks and experiment verification.
- `archive/`: preserved older experiments/notes/results; ignored by Git, not an active workflow.

A good training/validation fit does not guarantee deployed stability. Always
read spectral rates and failure counts alongside finite tracking losses.
