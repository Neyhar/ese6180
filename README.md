# ESE 618: can history reduce the need for action chunking?

There are **two complementary experiments**, sharing `model.py`:

| Study | What changes? | What it tells us | Start here |
|---|---|---|---|
| [Observability](observability/README.md) | Partial sensors, state dimension, history, execution | History can supply missing state information; once sufficient, frequent feedback usually helps under noise | [Saved results](observability/results/latest/RESULTS.md) |
| [Pitfalls](pitfalls/README.md) | History and execution on a 2D narrow-data hard example | History changes the ridge-selected feedback even with full observation | [Current graph](pitfalls/results/context_chunking/context_hypothesis.png) |

These are distinct mechanisms. Their raw losses use different normalizations,
evaluation windows and disturbance protocols, so do not compare their magnitudes.
The active [four-page report](midterm-report/main.tex) connects both to the theory.

## Start with existing results

- **Observability:** [one representative graph](observability/results/latest/figures/start_here.png) and [result summary](observability/results/latest/RESULTS.md).
- **Pitfalls:** [hypothesis graph](pitfalls/results/context_chunking/context_hypothesis.png), [partial-observation extension](pitfalls/results/partial_observation/README.md), and [experiment/theory guide](pitfalls/README.md).
- **Report:** [main.pdf](midterm-report/main.pdf), built from `midterm-report/main.tex` with the L4DC template.

## Run

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m observability.run --output observability/results/focused
python3 pitfalls/chunk_sweep.py
```

The observability command now defaults to one representative n=10,h=4 case;
the original 20-case output is preserved in `observability/results/latest/`.
`python3 run_experiment.py` forwards to the same runner for compatibility.
See each study's README for parameters and the full-grid command. Tests do not
write result figures unless `ESE618_TEST_PLOTS` is explicitly set.

## Repository map

- `model.py`: shared linear systems, expert demonstrations, history policies, ridge fitting and reconstruction.
- `observability/`: partial-observation experiment and its results.
- `pitfalls/`: full-observation hard-example experiment and its results.
- `midterm-report/`: active report and retained synthesis drafts.
- `theory-notes/main.tex`: longer mathematical derivations, including work not yet simulated.
- `docs/theory-overview.md`: concise theory map; numerical claims live in generated result summaries.
- `docs/code-and-theory.md`: implementation map and unimplemented analyses.
- `tests/`: mathematical checks and experiment verification.
- `archive/`: preserved older experiments/notes/results; ignored by Git, not an active workflow.

A good training/validation fit does not guarantee deployed stability. Always
read spectral rates and failure counts alongside finite tracking losses.
