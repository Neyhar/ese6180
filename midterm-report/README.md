# ESE 6180 midterm report

`main.tex` uses the unmodified `l4dc2026.cls` from the supplied L4DC2026 template archive. The main report targets four pages; references follow separately, as permitted by the assignment.

## Reproduce

Compile `main.tex` using the Codex LaTeX editor or a TeX installation with the JMLR/PMLR class, BibTeX and the standard packages. For example: `latexmk -pdf main.tex` or `tectonic main.tex`.

`make_figure.py` regenerates the report figure from the frozen `data/metrics.csv` using NumPy and Matplotlib. The saved `data/sweep_config.json` records the experiment settings. Data were copied from `../results/latest` on November 2, 2026; no experiments were rerun for this report.

The report describes the implementation in `../model.py`, `../long_experiment.py`, and `../run_experiment.py`, and condenses the mathematical framework in `../theory-notes/main.tex`. Spectral tests, asymptotic covariance validation, and nonlinear experiments are explicitly future work.

## Before submission

Review the scientific claims and author details. Only Nihar Ballamudi is listed, matching the theory notes; if this is a two-person project, add the second author and the required actual contribution/responsibility statement. This draft and portions of the existing code/theory were developed with Codex assistance; describe that assistance according to the course's policy. No individual contribution claims were invented.
