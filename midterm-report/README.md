# ESE 6180 report

`main.tex` is the active L4DC report, four pages total including references.
It uses the supplied, unmodified `l4dc2026.cls` and integrates observability,
Kyle Horton's chunk-stability proposition, the conditional history/execution
certificate, and current pitfalls results.

## Build

```sh
tectonic main.tex
```

Tectonic resolves the JMLR and font dependencies. Alternatively use
`latexmk -pdf main.tex` with the JMLR/PMLR packages installed. Fonts, margins,
and title layout follow the L4DC template; content was shortened to meet the
four-page limit.

The figure is `synthesis/figures/context_hypothesis.png`; frozen data and figure
regeneration instructions are in [synthesis/README.md](synthesis/README.md).
`main.pdf` is the current compiled report. The synthesis folder retains earlier
article-layout and extended working drafts; use `main.tex` for the active report.

The preamble selects `times` for pdfLaTeX and `fontspec` with Times New Roman
for XeLaTeX/LuaLaTeX/Tectonic, so bold and italic faces are loaded correctly.

## Partial-observation results note

[partial-observation-note.tex](partial-observation-note.tex) is the separate
two-page note. It reports the original n=2 reference and coupled systems at
n=2,4,8,16, using noisy/noiseless training terminology. Build with
`tectonic partial-observation-note.tex`; the compiled PDF is alongside it.

The papers summarize the broader saved experiment. The current focused,
x1-only mean-error comparisons are indexed in
[the Pitfalls results](../pitfalls/results/coupled_higher_dimensions/README.md).
Historical 2D result sets moved to
`../archive/pitfalls-before-cleanup/pitfalls/results/`.
