# Analysis of saved pitfalls experiments

`pitfalls-analysis.pdf` is a standalone L4DC-format review of all eight retained
pitfalls result runs: the current full-state sweep, the four partial-observation
comparison suites, and three archived protocols. It is separate from the short
L4DC note and the main report.

The report explains the experiment, learner inputs, coverage interventions,
mean/median loss, history and execution effects, exact stability, long-run
checks, baselines, archived differences, and limitations. It recomputes means
from raw CSVs; the existing experiment plots usually display medians.

## What was checked

- 1,215 controller-level means reconciled with 50,040 individual rollout rows.
- 2,400 partial-observation feature vectors checked against saved input blocks.
- Prediction-prefix invariance and paired full/partial narrow-fit agreement.
- Identical active test-state sets.
- All report pages rendered and visually checked.

The 2,565 learned records across snapshots include repeated prefixes and reused
training seeds. They are not independent experimental replications. No new
controllers were trained for this analysis.

## Reproduce

From this folder, with NumPy/Matplotlib and Tectonic available:

```sh
python3 analyze_outputs.py /path/to/ESE618
tectonic --keep-logs pitfalls-analysis.tex
```

`pitfalls-analysis.tex` is the editable report source; `l4dc2026.cls` is the
supplied L4DC template. Tectonic uses the system Times New Roman and Courier New
fonts; pdfLaTeX uses the standard Times setup and requires JMLR dependencies.

`analyze_outputs.py` reads the saved experiment folders and creates the figures,
`analysis_stats.json` and `source_manifest.json`. The manifest records source-file
sizes and SHA-256 hashes. The numerical narrative is maintained in the LaTeX
source; regenerating figures does not automatically update its tables or prose.
The superseded ReportLab builder is preserved under
`archive/pitfalls-before-focus/reportlab-analysis/`. LaTeX is the only active
report source. Neither build modifies experiment data.

This is a frozen review of the earlier eight runs; it does not include the
new `chunking_ablation` or `four_conditions` experiments. Their own READMEs
and raw CSVs document those fresh results.
