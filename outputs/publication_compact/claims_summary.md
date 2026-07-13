# Claims summary for publication_compact

## Dataset basis
- Seeds: 3
- Total labelled snapshots represented in tables: 64,800
- Graph policies: adaptive, fixed
- Radio scenarios: degraded, nominal
- Forecast horizon steps: 6
- Recorded wall-clock runtime: 4017.1 s.
- CUDA available during recorded run: false.

## Data-supported claims
- Lowest MAE: Current-state persistence baseline, Kinetic-TopoGuard with MAE=0.1195.
- Highest R2: GraphSAGE with R2=0.4575.
- Best fragmentation-risk F1: Current-state persistence baseline with F1=0.7619.
- Longest median topology-change lead: FANET-TopoGNN, FANET-TopoGNN (concat), GAT, GCN, Shallow ML, GraphSAGE, PI+MLP, STGCN (w=5) with 600.00 ms.
- Best/tied controller connectivity ratio: Kinetic-TopoGuard with 0.9058.

## Kinetic-TopoGuard status
- MAE=0.1195, R2=0.4249.
- Risk-F1=0.7363.
- Median topology-change lead=200.00 ms.
- MAE delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): 0.1551 absolute.
- R2 delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): 0.1725 absolute.

## FANET-TopoGNN vs naive concat
- MAE delta vs concat (positive favours FANET-TopoGNN): -0.0009 absolute.
- R2 delta vs concat (positive favours FANET-TopoGNN): -0.0239 absolute.
- Risk-F1 delta vs concat (positive favours FANET-TopoGNN): -0.0062 absolute.
- Interpretation: the concat ablation outperforms the adaptive gated variant on at least one reported metric in this run.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.