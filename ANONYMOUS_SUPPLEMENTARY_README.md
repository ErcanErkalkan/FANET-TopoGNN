# Supplementary Material S1 — anonymous reproducibility package

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

The completed five-seed PyTorch model-family table is descriptive context only. A
20-new-seed neural extension is specified separately but remains unexecuted unless a
verified PyTorch backend is available; S1 contains no surrogate or fabricated result for
that pending protocol.

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

The tracked `outputs/publication_compact/` bundle contains the three-seed compact model-family evidence used by the manuscript. Its summary must record the current cache version, all nine neural task rows as PyTorch executions, and `surrogate_used=false`. Concurrent shared-host duration is provenance only, not an isolated speed benchmark.

## Expanded 20-seed benchmark

```text
python scripts/run_submission_20seed.py
python scripts/run_submission_20seed.py --resume
```

The launcher checks for 20 unique seeds and exactly the focused Kinetic-TopoGuard, shallow-ML, and current-state persistence task set. The completed `outputs/paper_like_submission/` bundle contains all 20 per-seed results, aggregate tables, statistical tests, and generated figures. This focused base-environment run strengthens the primary learned comparison; it is not a 20-seed rerun of every compact neural baseline.

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

`Current-state persistence baseline` carries the observed component count forward unchanged to the forecast horizon and is used as a non-learning no-change reference. This label does not imply field or onboard deployability.

Current runs export PR-AUC, ROC-AUC, Brier score, expected calibration error, sample false alarms, one-to-one fragmentation-event precision/recall/F1, false alert events per minute, and threshold-sensitivity data. The 600 ms nominal horizon is one point in the executed 0.2--2.0 s sweep and is not a deployment guarantee.

The event-protocol sensitivity keeps the fitted score and validation-selected threshold
fixed while varying the warning window and refractory period over 0.3, 0.6, and 1.2 s and
stratifying held-out runs by measured event-density tertiles. See
`outputs/event_protocol_sensitivity/` and `SUPPLEMENTARY_TABLE_INDEX.md`.

## Evidence boundary

The evidence includes temporally correlated simulated-radio experiments, a real-field-motion transfer test, measured AERPAW aerial cellular RF/KPI and throughput checks, measured 60 GHz peer-link modelling with an explicitly transported forestry sensitivity, and measured MILUV three-UAV UWB topology transfer. A simplified SimPy queue/contention experiment reports simulated packet-level PDR and delay over the physical-radio graphs. The forestry bags still lack RF labels, AERPAW is UAV-to-infrastructure, and MILUV first-path power is not IP packet delivery. The one compatible MILUV sequence contains only one or two connected-to-fragmented events per evaluated threshold; repeated seeds do not create independent measured scenarios. Hardware-in-the-loop control, synchronized outdoor FANET IP packets, onboard profiling, and deployment-level validation remain outside the completed package.
