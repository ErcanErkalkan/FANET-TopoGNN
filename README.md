# Kinetic-TopoGuard / FANET-TopoGNN Reproducibility Suite

This project rebuilds the experimental pipeline described in the manuscript
`Kinetic-TopoGuard: Risk-Aware Topological Connectivity Forecasting in Dynamic Flying Ad Hoc Networks`.

Repository: `ErcanErkalkan/FANET-TopoGNN`

GitHub: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Zenodo DOI: <https://doi.org/10.5281/zenodo.20226053>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20226053.svg)](https://doi.org/10.5281/zenodo.20226053)

Current compact artefact tag: `v0.1.12`

It creates a reproducible Python research workflow that can:

- simulate dynamic FANET trajectories under Random Waypoint, Gauss-Markov, and mission mobility,
- build snapshot datasets with graph statistics, fixed/adaptive connectivity labels, physical-layer link filtering, radio-scenario sweeps, and `H0` persistence-image features,
- optionally export snapshots as PyTorch Geometric `Data` objects through `fanet.pyg_utils`,
- train Kinetic-TopoGuard plus static, topological, neural GNN, and temporal baselines,
- include the exact union-find current-topology detection oracle as a non-forecasting baseline,
- compute compact-reporting metrics, confidence intervals, and paired statistical tests,
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

## Environment Setup

The default reproducibility environment is the base scientific stack used by the non-deep-learning pipeline:

```bash
conda env create -f environment.yml
conda activate fanet-topognn
python -m pip install -e .
python main.py --config configs/publication_compact.json
```

The same base stack can be installed with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional neural dependencies are separated from the base stack. Install PyTorch-only components with:

```bash
python -m pip install -e ".[deep]"
# or
python -m pip install -r requirements-deep.txt
```

Install the PyTorch Geometric adapter stack with:

```bash
python -m pip install -e ".[pyg]"
# or
python -m pip install -r requirements-pyg.txt
```

If the publication compact run is interrupted, resume it with:

```bash
python main.py --config configs/publication_compact.json --resume
```

On Windows PowerShell, the helper script runs the same current compact profile by default:

```powershell
scripts\run_paper_like.ps1
scripts\run_paper_like.ps1 -Resume
```

The older `paper_like*` profiles are retained only as legacy exploratory configurations. The manuscript-facing profile is:

```bash
python main.py --config configs/publication_compact.json
python main.py --config configs/publication_compact.json --resume
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
powershell -ExecutionPolicy Bypass -File scripts\sync_manuscript_assets.ps1 -Profile publication_compact -ManuscriptDir paper
```

The same script copies all generated PNG figures plus `manuscript_tables.tex`, `claims_summary.md`, and `manuscript_summary.json`.

## Outputs

Each run creates tables, figures, and a `summary.json` file in the configured output folder.

The compact manuscript-facing result bundle is tracked in this repository at:

```text
outputs/publication_compact/
```

This bundle contains the CSV, JSON, LaTeX, per-seed, figure, and runtime-profile artefacts used for manuscript cross-checking. Transient rerun logs are excluded from the public artefact bundle. Other generated output folders remain ignored by default.

## Zenodo Archiving

Zenodo metadata is included in `.zenodo.json`, and citation metadata is included in `CITATION.cff`.

The archived compact manuscript release is available at <https://doi.org/10.5281/zenodo.20226053>.

Use `docs/zenodo_release.md` for the repository `ErcanErkalkan/FANET-TopoGNN` Zenodo archival record and DOI maintenance notes.

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

## Release archive hygiene

The distributable source archive intentionally excludes the local `.git/` directory. Version, citation, and release metadata are recorded in `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, and the release notes.
