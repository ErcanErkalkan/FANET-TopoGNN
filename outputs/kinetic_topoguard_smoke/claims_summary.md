# Claims summary for kinetic_topoguard_smoke

## Dataset basis
- Seeds: 1
- Total labelled snapshots represented in tables: 4,320
- Graph policies: adaptive, fixed
- Radio scenarios: degraded, nominal
- Forecast horizon steps: 6

## Data-supported claims
- Lowest MAE: Kinetic-TopoGuard with MAE=0.3858.
- Highest R2: Kinetic-TopoGuard with R2=0.4403.
- Best fragmentation-risk F1: Union-Find detection oracle with F1=0.7407.
- Longest median early-warning lead: Kinetic-TopoGuard, Shallow ML with 400.00 ms.
- Best/tied controller connectivity ratio: Kinetic-TopoGuard with 0.9819.

## Kinetic-TopoGuard status
- MAE=0.3858, R2=0.4403.
- Risk-F1=0.6003.
- Median early-warning lead=400.00 ms.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.