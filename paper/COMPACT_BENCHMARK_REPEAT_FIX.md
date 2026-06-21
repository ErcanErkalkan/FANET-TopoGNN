# Compact benchmark repetition fix

This update removes the repeated phrase `compact benchmark` from the manuscript body and included LaTeX table files.

## Main changes

- Replaced repeated `compact benchmark` wording with context-specific alternatives such as `cross-checked evaluation`, `reported setup`, `reported dataset`, `controller study`, `current evaluation`, `host-side run`, and `calibrated evaluation`.
- Preserved the methodological meaning: the manuscript still states that the evaluation uses three seeds, 64,800 labelled snapshots, fixed/adaptive graph policies, and nominal/degraded radio settings.
- Avoided weakening the paper with unnecessary repeated self-limitation language.
- Retained necessary references to the actual configuration path `configs/publication_compact.json` and output folder names, because those are reproducibility identifiers.

## Count check

- Exact `compact benchmark` occurrences in `main.tex`: 0
- Exact `compact-benchmark` occurrences in `main.tex`: 0
- Exact `compact benchmark` occurrences in included table files: 0

## Build check

- `pdflatex` completed successfully.
- The updated PDF was rendered to 63 page images for visual verification.
