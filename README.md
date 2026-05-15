# Kinetic-TopoGuard / FANET-TopoGNN Q1 Reproducibility Suite

This project rebuilds the experimental pipeline described in the manuscript
`Kinetic-TopoGuard: Risk-Aware Topological Connectivity Forecasting in Dynamic Flying Ad Hoc Networks`.

It creates a reproducible Python research workflow that can:

- simulate dynamic FANET trajectories under Random Waypoint, Gauss-Markov, and mission mobility,
- build snapshot datasets with graph statistics, fixed/adaptive connectivity labels, physical-layer link filtering, radio-scenario sweeps, and `H0` persistence-image features,
- optionally export snapshots as PyTorch Geometric `Data` objects through `fanet_q1.pyg_utils`,
- train Kinetic-TopoGuard plus static, topological, neural GNN, and temporal baselines,
- include the exact union-find current-topology detection oracle as a non-forecasting baseline,
- compute Q1-style reporting metrics, confidence intervals, and paired statistical tests,
- export publication-ready tables in CSV and LaTeX formats,
- generate figures for accuracy, lead time, latency, residuals, ablations, and network-level impact.
- aggregate results across multiple random seeds with confidence intervals and FDR-corrected significance tests.
- emit manuscript-ready CSV, LaTeX, Markdown, and artifact-manifest outputs.

The current manuscript-facing method is `Kinetic-TopoGuard`: a kinematic-topological forecaster that combines current connected-component state, velocity-projected link-margin features, `H0` persistence-image features, residual regression, and calibrated risk scoring. The older gated `FANET-TopoGNN` and concat variants are retained as neural topological comparators.

## Quick Start

```bash
python main.py --config configs/quickstart.json
```

Or on Windows PowerShell:

```powershell
scripts\setup_scientific_venv.ps1
scripts\run_quickstart.ps1
```

## Full Deep-Learning Environment

The current machine in this workspace uses Python 3.14. The scientific stack works there, but the full PyTorch path typically requires Python 3.12 or 3.11.

For the full deep-learning environment, use:

```bash
conda env create -f environment.yml
conda activate fanet-topognn-q1
python main.py --config configs/q1_publication_compact.json
```

If the Q1 publication run is interrupted, resume it with:

```bash
python main.py --config configs/q1_publication_compact.json --resume
```

On Windows PowerShell, the helper script runs the same current Q1 profile by default:

```powershell
scripts\run_paper_like.ps1
scripts\run_paper_like.ps1 -Resume
```

The older `paper_like*` profiles are retained only as legacy exploratory configurations. The manuscript-facing profile is:

```bash
python main.py --config configs/q1_publication_compact.json
python main.py --config configs/q1_publication_compact.json --resume
```

To run a legacy profile explicitly, pass it through `-ConfigPath`:

```powershell
scripts\run_paper_like.ps1 -ConfigPath configs\paper_like.json
```

For a submission-oriented large run with a reduced but stronger baseline set, use:

```bash
python main.py --config configs/paper_like_submission.json
python main.py --config configs/paper_like_submission.json --resume
```

This profile keeps the larger `paper_like` simulation scale but evaluates only the strongest static, topological, and temporal comparators plus the direct concat ablation.

To copy generated figure assets into the manuscript package, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_manuscript_assets.ps1 -Profile q1_publication_compact -ManuscriptDir ..\FANET_TopoGNN
```

The same script copies all generated PNG figures plus `manuscript_tables.tex`, `claims_summary.md`, and `manuscript_summary.json`.

## Outputs

Each run creates tables, figures, and a `summary.json` file in the configured output folder.

## Notes

- The original repository contents were not available in this workspace beyond the manuscript files, so this suite reconstructs the full experimental pipeline from the paper specification.
- `H0` persistent homology is implemented directly from the minimum spanning tree of the point cloud, which is exact for connected-component persistence.
- The project now treats the task as future-horizon connectivity forecasting rather than same-timestep regression, which is required for meaningful early-warning analysis.
- The simulator now implements the manuscript-level radio model: log-distance path loss, log-normal shadowing, Rayleigh/Nakagami fading, asymmetric transmit/receive chains, bidirectional link admission, optional SINR gating, and fixed/adaptive radius policies.
- Configs may provide `graph_policies` and `radio_scenarios` to evaluate fixed/adaptive policies and physical-layer sensitivity settings in one run while keeping run-wise split metadata explicit.
- Mission mobility uses offline scan, transit, and loiter waypoint plans per mission region with stochastic tracking, heading, and speed perturbations.
- Run-wise train/validation/test splits are grouped by the base `(mobility, N, seed)` identifier, so fixed/adaptive graph-policy and nominal/degraded radio variants of the same underlying trajectory cannot leak across folds. The exported `split_assignments.csv` documents the mapping.
- The pipeline exports manuscript-facing files such as `manuscript_tables.tex`, `manuscript_summary.json`, `report.md`, and `artifact_manifest.txt`.
- Resume mode checks per-model cache coverage and the configuration signature before skipping a seed, which makes interrupted or incrementally expanded runs safer to continue without reusing stale artefacts.
