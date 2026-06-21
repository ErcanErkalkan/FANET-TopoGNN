# Claims summary for paper_like_submission

## Dataset basis
- Seeds: 20
- Total labelled snapshots represented in tables: 432,000
- Graph policies: adaptive, fixed
- Radio scenarios: degraded, nominal
- Forecast horizon steps: 6
- Recorded wall-clock runtime: 11700.4 s.
- CUDA available during recorded run: false.

## Data-supported claims
- Lowest MAE: Kinetic-TopoGuard with MAE=0.3077.
- Highest R2: Kinetic-TopoGuard with R2=0.5391.
- Best fragmentation-risk F1: Union-Find detection oracle with F1=0.7545.
- Longest median topology-change lead: Shallow ML with 495.00 ms.
- Best/tied controller connectivity ratio: Kinetic-TopoGuard with 0.8412.

## Kinetic-TopoGuard status
- MAE=0.3077, R2=0.5391.
- Risk-F1=0.5871.
- Median topology-change lead=230.00 ms.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.