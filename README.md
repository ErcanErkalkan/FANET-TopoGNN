# Kinetic-TopoGuard / FANET-TopoGNN Reproducibility Suite

## Locked EAAI source-gated confirmation

Before inspecting any locked-confirmatory test result, the primary hypothesis was
registered as: **Source-Gated Kinetic-TopoGuard versus Current-state ExtraTrees on
fragmentation-event F1.** The frozen configuration and its SHA-256 lock are stored
in `configs/eaai_locked_confirmatory.json` and
`configs/eaai_locked_confirmatory.lock.json`. The primary superiority rule requires
a positive paired seed-level event-F1 difference, a seed-bootstrap 95% confidence
interval excluding zero, and a Holm-adjusted two-sided paired-permutation p-value below
0.05. The bootstrap interval summarizes effect uncertainty; the multiplicity-controlled
permutation result governs the confirmatory decision.
The locked test seeds must not be used to alter features, learners, calibration, or
threshold-selection rules.

## EAAI P1 evidence additions

The main manuscript now keeps eight decision-critical tables; detailed tables and their
machine-readable sources are mapped in `SUPPLEMENTARY_TABLE_INDEX.md` and packaged as
Supplementary Material S1.

Event-definition robustness is evaluated without refitting the model or retuning its
validation-selected threshold:

```bash
python scripts/run_event_protocol_sensitivity.py --workers 2
```

The executed five-seed output varies the warning window and alert refractory period over
0.3, 0.6, and 1.2 seconds and stratifies held-out runs by measured event-density tertiles.
It is descriptive sensitivity evidence, not part of the locked hypothesis family.

A separate 20-new-seed neural extension is prepared for Current-state ExtraTrees,
Kinetic-TopoGuard, FANET-TopoGNN, STGCN, and TGN:

```bash
python scripts/run_neural_20seed_extension.py --prepare-only
```

Its launcher verifies domain-separated unused seeds and refuses surrogate execution.
Because the current P1 build environment does not provide PyTorch, this extension remains
explicitly `prepared_not_executed`; the manuscript makes no claim from unexecuted or
partial neural results.

Run or resume the locked evaluation with:

```bash
python scripts/run_eaai_locked_confirmatory.py --workers 5 --resume
```

This project rebuilds the experimental pipeline described in the manuscript
`A Reproducible Benchmark and Cross-Domain Stress Test for Fragmentation-Event Forecasting in Unmanned Aerial Vehicle Networks`.

Repository: `ErcanErkalkan/FANET-TopoGNN`

GitHub: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Zenodo DOI: <https://doi.org/10.5281/zenodo.20226053>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20226053.svg)](https://doi.org/10.5281/zenodo.20226053)

Working-tree Python package version: `0.1.12`

Cited archived snapshot: `v0.1.0-q1-compact` at
<https://doi.org/10.5281/zenodo.20226053>. The DOI does not archive the current working
tree; the two version scopes must not be reported as the same release.

It creates a reproducible Python research workflow that can:

- simulate dynamic FANET trajectories under Random Waypoint, Gauss-Markov, and mission mobility,
- build snapshot datasets with graph statistics, fixed/adaptive connectivity labels, physical-layer link filtering, radio-scenario sweeps, and `H0` persistence-image features,
- optionally export snapshots as PyTorch Geometric `Data` objects through `fanet.pyg_utils`,
- train Kinetic-TopoGuard plus static, topological, neural GNN, and temporal baselines,
- include a current-state persistence forecast as a non-learning no-change reference,
- compute compact-reporting metrics, confidence intervals, and paired statistical tests,
- export publication-ready tables in CSV and LaTeX formats,
- generate figures for accuracy, lead time, latency, residuals, ablations, and network-level impact.
- aggregate results across multiple random seeds with confidence intervals and FDR-corrected significance tests.
- emit manuscript-ready CSV, LaTeX, Markdown, and artifact-manifest outputs.

The current manuscript-facing method is `Kinetic-TopoGuard`: a kinematic-topological forecaster that combines current connected-component state, velocity-projected link-margin features, `H0` persistence-image features, residual regression, and calibrated risk scoring. The older gated `FANET-TopoGNN` and concat variants are retained as neural topological comparators.

Risk labels use exact-horizon semantics: `frag_at_horizon = 1` means that the projected target snapshot at `t + forecast_horizon_steps` is fragmented (`beta0(t+h) > 1`). It does not mean that any intermediate tick inside `(t, t+h)` fragmented.

## Quick Start

Run the bounded smoke profile first. It is designed to finish in roughly 30 seconds on the reference CPU environment and verifies imports, dataset generation, evaluation, and output export:

```bash
python main.py --config configs/smoke_30s.json
python scripts/run_smoke_test.py
pytest -q tests/test_smoke.py
```

The broader exploratory quickstart remains available as:

```bash
python main.py --config configs/quickstart.json
```

Or on Windows PowerShell:

```powershell
scripts\setup_scientific_venv.ps1
scripts\run_quickstart.ps1
```

## Environment Setup

The release intentionally separates the lightweight scientific environment from the environment needed for the neural manuscript baselines.

### Base scientific environment

Use the base environment for dataset generation, Kinetic-TopoGuard, current-state persistence, shallow/tabular baselines, smoke tests, table/figure aggregation, and manuscript asset checks. This environment does **not** install PyTorch.

```bash
conda env create -f environment.yml
conda activate fanet-topognn
python -m pip install -e .
python main.py --config configs/quickstart.json
```

The same base stack can be installed with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Deep-learning environment for manuscript neural baselines

Install the deep extras before running configurations that include neural GNN or temporal model names such as `GCN`, `GAT`, `GraphSAGE`, `FANET-TopoGNN`, `T-GCN`, `STGCN`, or `TGN`:

```bash
python -m pip install -e ".[deep]"
# or
python -m pip install -r requirements-deep.txt
```

Install the PyTorch Geometric adapter stack only when exporting snapshots as PyG `Data` objects:

```bash
python -m pip install -e ".[pyg]"
# or
python -m pip install -r requirements-pyg.txt
```

Grouped run-block permutation explanations require only the base environment. TreeSHAP is
an optional secondary explanation and can be installed without making SHAP a core runtime
dependency:

```bash
python -m pip install -e ".[explain]"
```

### Fallback/surrogate behaviour

If PyTorch is not installed, the code keeps long runs executable by replacing neural model names with deterministic scikit-learn surrogate estimators that use the corresponding feature family. This fallback is useful for CI, smoke tests, and base-environment reproducibility checks, but it should **not** be described as a real PyTorch GNN or temporal-neural result. For manuscript-facing neural-baseline reruns, install `.[deep]` or `requirements-deep.txt` first.

New `summary.json` and `runtime_profile.json` files record `cache_version`, `model_backend`, `torch_available`, `torch_version`, `device`, and `surrogate_used`. Per-seed model tables also record `Model_Backend`. A compact artifact is manuscript-eligible only when all nine neural GNN/temporal task rows are recorded as `pytorch`, `surrogate_used` is false, and the current event/radio cache version is present. Shared-host runtime is retained for provenance and is not treated as an isolated speed benchmark.

Run the compact manuscript profile with the deep environment when neural rows are intended to be interpreted as neural baselines:

```bash
python -m pip install -e ".[deep]"
python main.py --config configs/publication_compact.json
```

If the publication compact run is interrupted, resume it with:

```bash
python main.py --config configs/publication_compact.json --resume
```

The non-selective five-seed baseline inventory and both manuscript tables are rebuilt from
executed artifacts with:

```bash
python scripts/build_neural_seed_extension.py
```

The builder requires all 14 declared models on seeds 7, 17, 27, 37, and 47, rejects any
PyTorch-named row whose backend is not `pytorch`, and writes missing/failed runs to
`training_failures.csv`. Legacy artifacts did not retain per-model training durations;
those cells remain explicitly unavailable. A versioned rerun that records these durations
can be resumed without modifying the legacy evidence:

```bash
python main.py --config configs/publication_neural_full_5seed.json --seed-workers 5 --resume
```

On Windows PowerShell, the helper script runs the same current compact profile by default:

```powershell
scripts\run_paper_like.ps1
scripts\run_paper_like.ps1 -Resume
```

The older `paper_like*` profiles are retained only as legacy exploratory configurations. The manuscript-facing profile remains `configs/publication_compact.json`; install the deep dependencies first whenever its neural rows will be reported as PyTorch GNN or temporal-neural baselines:

```bash
python -m pip install -r requirements-deep.txt
python main.py --config configs/publication_compact.json
python main.py --config configs/publication_compact.json --resume
```

To run a legacy profile explicitly, pass it through `-ConfigPath`:

```powershell
scripts\run_paper_like.ps1 -ConfigPath configs\paper_like.json
```

For a submission-oriented 20-seed confirmatory run that preserves the compact benchmark design and includes the proposed Kinetic-TopoGuard model, use:

```bash
python scripts/run_submission_20seed.py
python scripts/run_submission_20seed.py --resume
```

The launcher verifies 20 unique seeds and requires exactly the primary Kinetic-TopoGuard model, the selected shallow-ML comparator, and the non-learning current-state persistence reference. The profile preserves the compact mobility, graph-policy, radio-scenario, split, and horizon design. The expensive neural family remains in the verified compact benchmark plus its five-seed extension; the 20-seed profile is focused on the primary inferential comparison.

For the public real-flight motion transfer test, install the optional ROS bag reader, verify/download the three source bags, derive the aligned 10 Hz trace, and run the frozen-transfer evaluation:

```powershell
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts\download_external_validation.ps1
python scripts\extract_forestry_trace.py
python scripts\run_external_validation.py
```

The source bags remain ignored. The tracked derived trace, checksum manifest, transfer metrics, prediction records, protocol JSON, and trajectory/radius figure document the external evidence. Because the public bags contain motion but no RF or packet ground truth, the study is a real-field-motion transfer test under deterministic radius sensitivity, not a field-measured wireless-network test.

For measured three-UAV UWB topology transfer and the complementary protocol studies, use:

```powershell
python scripts/download_miluv_validation.py
python scripts/run_miluv_validation.py
python scripts/run_horizon_sweep.py --workers 2
python scripts/run_factorial_feature_ablation.py --workers 2
python scripts/run_packet_level_controller_validation.py --workers 2
python scripts/run_closed_loop_controller_validation.py --workers 5 --resume
python scripts/benchmark_end_to_end_latency.py
```

The closed-loop experiment is distinct from the one-snapshot reachability proxy. It carries
bounded relay state through time, reuses identical mobility, radio, and packet-arrival
realizations across six policies, and reports SimPy packet delivery, deadline, delay,
connectivity, recovery, and action-cost outcomes. It is a controlled simulation and does
not represent a field UAV controller or measured IP packet-delivery ratio.

The frozen-model explainability analysis is resumable and never feeds test explanations
back into model or threshold selection:

```powershell
python scripts/run_explainability.py --workers 5 --resume
```

If SHAP is unavailable or fails for an unsupported tree implementation, the protocol records
the reason and grouped run-block permutation importance still completes.

The MILUV downloader uses the official Figshare record, range-extracts only the six required Vicon/UWB CSV members, and verifies ZIP CRC and per-file SHA-256 hashes. MILUV supplies measured motion and inter-robot UWB first-path power, not IP packet-delivery labels. The single compatible sequence contains only one or two connected-to-fragmented events per evaluated threshold, so it is a sparse transfer stress test rather than strong external validation. The SimPy study is a separate simplified queue/contention experiment over simulated physical-radio graphs.

To copy generated figure assets into the manuscript package, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_manuscript_assets.ps1 -Profile publication_compact -ManuscriptDir paper
```

The same script copies all generated PNG figures plus `manuscript_tables.tex`, `claims_summary.md`, and `manuscript_summary.json`.

## Outputs

Each run creates tables, figures, and a `summary.json` file in the configured output folder.

Expected machine-readable files include `summary.json`, `runtime_profile.json`, `metrics_overall.csv`, `risk_metrics.csv`, `risk_threshold_sensitivity.csv`, `lead_time_summary.csv`, `network_metrics.csv`, and `artifact_manifest.txt`. Risk exports include PR-AUC, ROC-AUC, Brier score, expected calibration error, sample false alarms, one-to-one fragmentation-event precision/recall/F1, false alert events per minute, and threshold sensitivity. These values are computed only for current runs; they are not fabricated for legacy caches.

The compact manuscript-facing result bundle is tracked in this repository at:

```text
outputs/publication_compact/
```

This bundle contains the CSV, JSON, LaTeX, per-seed, figure, and runtime-profile artefacts used for manuscript cross-checking. The focused confirmatory outputs are stored under `outputs/paper_like_submission/`, and real-motion transfer outputs under `outputs/external_validation/`. Transient rerun logs are excluded from the public artefact bundle. Other exploratory output folders remain ignored by default.

## Zenodo Archiving

Zenodo metadata is included in `.zenodo.json`, and citation metadata is included in `CITATION.cff`.

The archived compact manuscript release `v0.1.0-q1-compact` is available at
<https://doi.org/10.5281/zenodo.20226053>. Package version `0.1.12` identifies the
current working tree and must not be attributed to that archived DOI.

Use `docs/zenodo_release.md` for the repository `ErcanErkalkan/FANET-TopoGNN` Zenodo archival record and DOI maintenance notes.

## Journal submission note

The double-anonymous manuscript PDF and `anonymous_supplementary.zip` exclude author-identifying metadata. The author-identifying `paper/title_page.tex` and public citation/archive metadata remain separate and must be uploaded only in the journal portal fields intended for author information.

The manuscript separates simulation, model-family, 20-seed confirmation, counterfactual real-motion transfer, measured AERPAW cellular evidence, transported 60 GHz sensitivity, measured MILUV three-UAV UWB topology, and simplified packet-level simulation. It does not claim measured field IP-PDR, onboard profiling, hardware-in-the-loop control, or deployment readiness.

## Technical notes

- This working tree is the curated executable reproducibility package for the manuscript. The cited DOI identifies the earlier `v0.1.0-q1-compact` snapshot; source code, configurations, and outputs added afterward belong to package version `0.1.12` until a new archive is minted.
- `H0` persistent homology is implemented directly from the minimum spanning tree of the point cloud, which is exact for connected-component persistence.
- The project now treats the task as future-horizon connectivity forecasting rather than same-timestep regression, which is required for meaningful early-warning analysis.
- The simulator implements log-distance path loss, run-persistent asymmetric hardware offsets, temporally correlated log-normal shadowing and fading, bidirectional link admission, optional SINR gating, and fixed/adaptive radius policies.
- Configs may provide `graph_policies` and `radio_scenarios` to evaluate fixed/adaptive policies and physical-layer sensitivity settings in one run while keeping run-wise split metadata explicit.
- Mission mobility uses offline scan, transit, and loiter waypoint plans per mission region with stochastic tracking, heading, and speed perturbations.
- Run-wise train/validation/test splits are grouped by the base `(mobility, N, seed)` identifier, so fixed/adaptive graph-policy and nominal/degraded radio variants of the same underlying trajectory cannot leak across folds. The exported `split_assignments.csv` documents the mapping.
- The pipeline exports manuscript-facing files such as `manuscript_tables.tex`, `manuscript_summary.json`, `report.md`, and `artifact_manifest.txt`.
- Resume mode checks per-model cache coverage and the configuration signature before skipping a seed, which makes interrupted or incrementally expanded runs safer to continue without reusing stale artefacts.

## Release archive hygiene

The distributable source archive intentionally excludes the local `.git/` directory. Version, citation, and release metadata are recorded in `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, and the release notes.
