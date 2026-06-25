# Anonymous supplementary reproducibility package

This review package contains the code, configurations, compact and 20-seed outputs, derived public flight trace, AERPAW aerial cellular outputs, external-transfer outputs, and focused tests needed to inspect and rerun the Kinetic-TopoGuard study. Author names, affiliations, email addresses, public repository identifiers, software-archive identifiers, and personal license fields have been removed.

## Environment

Python 3.12 is the reference version. Install the base scientific stack with:

```text
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install PyTorch before interpreting GCN, GAT, GraphSAGE, PI+MLP, FANET-TopoGNN, T-GCN, STGCN, or TGN rows as neural results:

```text
python -m pip install -r requirements-deep.txt
```

Without PyTorch, the pipeline can use deterministic scikit-learn surrogates for those model names. New summaries record the backend; surrogate rows are not valid neural baseline evidence.

## Smoke test

```text
python main.py --config configs/smoke_30s.json
python scripts/run_smoke_test.py
pytest -q tests/test_smoke.py
```

Expected smoke outputs are written beneath `outputs/smoke_30s/`, including `summary.json`, `metrics_overall.csv`, `risk_metrics.csv`, and `risk_threshold_sensitivity.csv`.

## Compact benchmark

```text
python -m pip install -r requirements-deep.txt
python main.py --config configs/publication_compact.json
python main.py --config configs/publication_compact.json --resume
```

The tracked `outputs/publication_compact/` bundle contains the three-seed compact model-family evidence used by the manuscript. Its fresh provenance rerun records all nine neural task rows as PyTorch 2.11.0 CPU executions and records `surrogate_used=false`. The concurrent rerun duration is provenance only, not an isolated speed benchmark.

## Expanded 20-seed benchmark

```text
python scripts/run_submission_20seed.py
python scripts/run_submission_20seed.py --resume
```

The launcher checks for 20 unique seeds and exactly the focused Kinetic-TopoGuard, shallow-ML, and union-find task set. The completed `outputs/paper_like_submission/` bundle contains all 20 per-seed results, aggregate tables, statistical tests, and generated figures. This focused base-environment run strengthens the primary learned comparison; it is not a 20-seed rerun of every compact neural baseline.

## Real-flight motion transfer

```text
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts\download_external_validation.ps1
python scripts/extract_forestry_trace.py
python scripts/run_external_validation.py
```

The three source ROS bags are verified against authoritative MD5 checksums and remain outside the package because of their size. The package includes the synchronized 10 Hz derived trace, source/checksum manifest, 107,730 prediction records, raw and physically clipped metrics, protocol JSON, and trajectory/radius figure. The public source DOI is `10.5281/zenodo.14701641` (CC BY 4.0).

## Measured AERPAW cellular validation

```text
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts\download_aerpaw_cellular_validation.ps1
python scripts/run_aerpaw_cellular_validation.py
```

The public AERPAW Dataset-22 and Dataset-23 CSV folders remain outside the package as third-party raw data. The package includes the derived LTE link-state metrics, iPerf throughput metrics, source/hash protocol JSON, manuscript table, and generated validation figure. This evidence uses measured UAV-to-cellular-infrastructure RF/KPI and throughput traces; it is not synchronized inter-UAV FANET packet-link validation.

## Output interpretation

`Union-Find detection oracle` uses current graph connectivity as a diagnostic reference and is not a future forecasting model. Forecasting and learned-model rankings must therefore be discussed separately from this row.

New runs export PR-AUC, ROC-AUC, Brier score, expected calibration error, false alarms per minute, and threshold-sensitivity data. The 600 ms lead-time ceiling in the compact profile is imposed by the six-step forecast horizon and is not a deployment guarantee.

## Evidence boundary

The evidence includes simulated-radio experiments, a real-field-motion transfer test, and measured AERPAW aerial cellular RF/KPI and throughput checks. The public flight bags do not contain measured RF or packet-reception labels, so their graph labels are counterfactual communication-radius sensitivities. The AERPAW cellular traces are UAV-to-infrastructure rather than peer-to-peer inter-UAV FANET labels. Hardware-in-the-loop experiments, synchronized inter-UAV wireless links, packet-level medium-access validation, onboard profiling, and deployment-level validation remain outside the completed package.
