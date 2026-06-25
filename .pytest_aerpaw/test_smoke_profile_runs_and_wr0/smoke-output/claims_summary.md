# Claims summary for smoke_30s

## Dataset basis
- Seeds: 1
- Total labelled snapshots represented in tables: 36
- Graph policies: fixed
- Radio scenarios: default
- Forecast horizon steps: 2
- Recorded wall-clock runtime: 16.3 s.
- CUDA available during recorded run: false.

## Data-supported claims
- Lowest MAE: Current-state persistence baseline with MAE=0.0000.
- Highest R2: Current-state persistence baseline with R2=1.0000.
- Best fragmentation-risk F1: Current-state persistence baseline, Density + min-distance heuristics with F1=1.0000.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.