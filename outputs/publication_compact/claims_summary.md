# Claims summary for publication_compact

## Dataset basis
- Seeds: 3
- Total labelled snapshots represented in tables: 64,800
- Graph policies: adaptive, fixed
- Radio scenarios: degraded, nominal
- Forecast horizon steps: 6
- Recorded wall-clock runtime: 4099.0 s.
- CUDA available during recorded run: false.

## Data-supported claims
- Lowest MAE: Union-Find detection oracle with MAE=0.3045.
- Highest R2: Shallow ML with R2=0.5594.
- Best fragmentation-risk F1: Union-Find detection oracle with F1=0.7311.
- Longest median topology-change lead: FANET-TopoGNN, PI+MLP with 600.00 ms.
- Best/tied controller connectivity ratio: GraphSAGE, Kinetic-TopoGuard with 0.8613.

## Kinetic-TopoGuard status
- MAE=0.3090, R2=0.4979.
- Risk-F1=0.5547.
- Median topology-change lead=233.33 ms.
- MAE delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): 0.1535 absolute.
- R2 delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): 0.0932 absolute.

## FANET-TopoGNN vs naive concat
- MAE delta vs concat (positive favours FANET-TopoGNN): -0.0177 absolute.
- R2 delta vs concat (positive favours FANET-TopoGNN): -0.0222 absolute.
- Risk-F1 delta vs concat (positive favours FANET-TopoGNN): -0.0730 absolute.
- Interpretation: the concat ablation outperforms the adaptive gated variant on at least one reported metric in this run.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.