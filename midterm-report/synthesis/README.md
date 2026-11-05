# Integrated theory and results working draft

This four-page revision (including references) combines the supplied Kyle Horton proposition draft,
the existing `midterm-report/main.tex`, and the focused pitfalls recovery sweep.
It is a concise research synthesis. The previous eight-page treatment is retained
as `theory-results-synthesis-extended.tex` and `.pdf`; the original midterm report
is also retained.

## Main distinctions

Proposition numbers follow `main.tex` and `theory-results-synthesis.tex`; the extended draft numbers them 3, 4, 5.

1. Exact action recovery from history is a representation result; learning it
   requires data coverage and estimation guarantees.
2. Proposition 2 standardizes Kyle's notation and completes the chunk-stability
   argument using the completed proof already in the midterm report. It does
   not imply that additional history shortens the certified chunk: its context
   constant grows and it assumes execution >= context - 1.
3. Proposition 3 supplies a conditional general-learner contraction certificate.
   Its deployment-relative error assumption has not been verified for the raw
   history ridge learner.
4. Proposition 4 derives the specific rank-one ridge controller and gives a
   small-gain certificate for one-step stability at history 16. It is not a
   guarantee of monotonically decreasing test loss.
5. The earlier noisy/partial-observation results and the current noiseless,
   full-observation recovery sweep are different protocols and are labeled
   separately. The old simulations were not rerun for this synthesis.

`data/` freezes the current CSV/JSON results used for the new figure and table.
`make_figure.py` regenerates the publication-sized figure from that snapshot.
The exact ridge weight formula was checked against numerical fits at histories
1,2,4,8,16, agreeing within floating-point precision; the small-gain coefficients
were evaluated independently. These checks support the derivations but do not
replace mathematical review of the working propositions.

## Build

```sh
python3 make_figure.py
pdflatex -interaction=nonstopmode -halt-on-error theory-results-synthesis.tex
pdflatex -interaction=nonstopmode -halt-on-error theory-results-synthesis.tex
```

The local TeX installation lacks the JMLR class required by `l4dc2026.cls`, so
this separate working draft uses a standard article layout. The existing
midterm report and its L4DC formatting remain available in the parent folder.
