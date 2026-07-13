# Anonymous supplementary execution notes

This note contains reproducibility detail moved out of the anonymous main manuscript. It contains no author-identifying repository, archive, or affiliation metadata.

## Package layout

- `main.py`: command-line entry point for simulation, training, evaluation, and report export.
- `fanet/`: mobility, topology, feature, model, training, evaluation, and reporting code.
- `configs/`: smoke, compact, and expanded submission profiles.
- `outputs/publication_compact/`: backend-verified compact model-family evidence.
- `outputs/paper_like_submission/`: completed focused 20-seed evidence.
- `outputs/external_validation/`: completed real-flight-motion transfer evidence.
- `outputs/aerpaw_cellular_validation/`: completed AERPAW aerial cellular RF/KPI and throughput evidence.
- `outputs/uav_to_uav_mmwave_validation/`: measured 60 GHz peer-link and transported sensitivity evidence.
- `outputs/miluv_validation/`: measured three-UAV motion/UWB topology transfer.
- `outputs/horizon_sweep/`, `outputs/factorial_feature_ablation/`: sensitivity and source-isolation studies.
- `outputs/packet_level_controller/`, `outputs/end_to_end_latency/`: packet and complete host-loop studies.
- `data/external_validation/`: derived traces, selected MILUV CSVs, and source/checksum manifests.
- `scripts/`: download, simulation, transfer, protocol, packaging, and audit launchers.

## Execution commands

Install the base scientific dependencies and run the smoke profile:

```text
python -m pip install -r requirements.txt
python main.py --config configs/smoke_30s.json
```

Install PyTorch before producing manuscript-facing neural baseline rows:

```text
python -m pip install -r requirements-deep.txt
python main.py --config configs/publication_compact.json
```

Run or resume the expanded 20-seed profile:

```text
python scripts/run_submission_20seed.py
python scripts/run_submission_20seed.py --resume
```

Reproduce the external flight-motion transfer study:

```text
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts\download_external_validation.ps1
python scripts/extract_forestry_trace.py
python scripts/run_external_validation.py
```

Reproduce the measured AERPAW cellular validation:

```text
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts\download_aerpaw_cellular_validation.ps1
python scripts/run_aerpaw_cellular_validation.py
```

Reproduce the peer-link, MILUV, sensitivity, packet, and latency studies:

```text
python scripts/run_uav_to_uav_mmwave_validation.py
python scripts/download_miluv_validation.py
python scripts/run_miluv_validation.py
python scripts/run_horizon_sweep.py --workers 2
python scripts/run_factorial_feature_ablation.py --workers 2
python scripts/run_packet_level_controller_validation.py --workers 2
python scripts/benchmark_end_to_end_latency.py
```

Generate manuscript tables, the offline replay, and the final audit:

```text
python scripts/select_operating_point.py
python scripts/generate_evidence_tables.py
python scripts/build_digital_twin_dashboard.py
python scripts/build_anonymous_supplementary.py
python scripts/audit_submission_readiness.py
```

Each completed run exports `summary.json`, `runtime_profile.json`, CSV and LaTeX tables, `report.md`, `claims_summary.md`, figures, and an artifact manifest beneath the configured output directory.

## Backend interpretation

New output summaries record `model_backend`, `torch_available`, `torch_version`, `device`, and `surrogate_used`. When PyTorch is unavailable, neural model names are executed by deterministic feature-based surrogate estimators so that lightweight checks remain executable. Such rows are not neural baseline results and must not be reported as PyTorch graph or temporal neural models.

The compact artifact is accepted only when its canonical summary records a verified PyTorch backend, `surrogate_used=false`, and the current cache version. The five-seed neural aggregate records source cache versions and rejects mixed stale/current inputs.

## Temporal baseline settings

The compact profile uses T-GCN, STGCN, and TGN with window size 5, learning rate 0.001, weight decay 0.0001, batch size 64, at most 18 epochs, and early stopping patience 5. STGCN uses temporal kernel size 3, T-GCN uses a GRU-style update, and TGN uses the implemented memory/message dimensions. Larger temporal windows and richer temporal-topological hybrids are outside the tracked compact evidence.

## Evidence boundary

The package keeps evidence layers separate. Forestry bags provide motion only. AERPAW provides UAV-to-infrastructure RF/KPI and iPerf measurements. The 60 GHz source provides measured UAV-to-UAV links, while its application to forestry motion is a transported, support-bounded sensitivity rather than calibration. MILUV provides direct three-UAV motion and UWB first-path-power topology but no IP packets. SimPy provides simplified packet-level PDR and delay over simulated physical-radio graphs. None of these sources establishes synchronized outdoor FANET IP-PDR, onboard timing/power, hardware-in-the-loop control, or deployment safety.
