# Claims summary for publication_neural_extension

## Dataset basis
- Seeds: 2
- Total labelled snapshots represented in tables: 43,200
- Graph policies: adaptive, fixed
- Radio scenarios: degraded, nominal
- Forecast horizon steps: 6
- Recorded wall-clock runtime: 2787.7 s.
- CUDA available during recorded run: false.

## Data-supported claims
- Lowest MAE: Kinetic-TopoGuard with MAE=0.3094.
- Highest R2: FANET-TopoGNN with R2=0.4933.
- Best fragmentation-risk F1: Kinetic-TopoGuard with F1=0.5672.
- Longest median topology-change lead: FANET-TopoGNN, TGN (w=5), PI+MLP with 600.00 ms.
- Best/tied controller connectivity ratio: FANET-TopoGNN, FANET-TopoGNN (concat), Kinetic-TopoGuard, GraphSAGE with 0.8653.

## Kinetic-TopoGuard status
- MAE=0.3094, R2=0.4473.
- Risk-F1=0.5672.
- Median topology-change lead=100.00 ms.
- MAE delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): 0.0751 absolute.
- R2 delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): -0.0460 absolute.

## FANET-TopoGNN vs naive concat
- MAE delta vs concat (positive favours FANET-TopoGNN): 0.0034 absolute.
- R2 delta vs concat (positive favours FANET-TopoGNN): 0.0208 absolute.
- Risk-F1 delta vs concat (positive favours FANET-TopoGNN): -0.0389 absolute.
- Interpretation: the concat ablation outperforms the adaptive gated variant on at least one reported metric in this run.

## Statistical testing
- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.

## Sensitivity evidence
- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.
- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.