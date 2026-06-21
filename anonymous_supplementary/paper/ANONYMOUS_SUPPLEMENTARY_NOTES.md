# Anonymous supplementary execution notes

This note contains reproducibility detail moved out of the anonymous main manuscript. It contains no author-identifying repository, archive, or affiliation metadata.

## Package layout

- `main.py`: command-line entry point for simulation, training, evaluation, and report export.
- `fanet/`: mobility, topology, feature, model, training, evaluation, and reporting code.
- `configs/`: smoke, compact, and expanded submission profiles.
- `outputs/publication_compact/`: backend-verified three-seed compact model-family evidence.
- `outputs/paper_like_submission/`: completed focused 20-seed evidence.
- `outputs/external_validation/`: completed real-flight-motion transfer evidence.
- `data/external_validation/derived/`: synchronized trace and source/checksum manifest.
- `scripts/`: smoke, resumable expanded-run, trace-extraction, external-validation, and audit launchers.

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

Each completed run exports `summary.json`, `runtime_profile.json`, CSV and LaTeX tables, `report.md`, `claims_summary.md`, figures, and an artifact manifest beneath the configured output directory.

## Backend interpretation

New output summaries record `model_backend`, `torch_available`, `torch_version`, `device`, and `surrogate_used`. When PyTorch is unavailable, neural model names are executed by deterministic feature-based surrogate estimators so that lightweight checks remain executable. Such rows are not neural baseline results and must not be reported as PyTorch graph or temporal neural models.

The compact artifact was freshly rerun with the metadata-producing pipeline. Its canonical summary records PyTorch 2.11.0 on CPU, `surrogate_used=false`, and nine PyTorch neural task rows. `provenance_rerun.json` records the validation and notes that concurrent wall-clock duration is not an isolated runtime benchmark.

## Temporal baseline settings

The compact profile uses T-GCN, STGCN, and TGN with window size 5, learning rate 0.001, weight decay 0.0001, batch size 64, at most 18 epochs, and early stopping patience 5. STGCN uses temporal kernel size 3, T-GCN uses a GRU-style update, and TGN uses the implemented memory/message dimensions. Larger temporal windows and richer temporal-topological hybrids are outside the tracked compact evidence.

## Evidence boundary

The manuscript separates three evidence layers: a three-seed compact neural-family benchmark, a focused 20-seed primary learned comparison, and a real-field-motion transfer test. The flight source has no measured RF or packet ground truth, so deterministic radius graphs are sensitivity assumptions. Hardware-in-the-loop testing, measured links, packet-level medium-access, onboard profiling, and deployment validation remain future work.
