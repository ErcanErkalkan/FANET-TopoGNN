# Accepted-Article Gap Closure

Reference article: `1-s2.0-S0952197626017422-main.pdf`, *Engineering Applications of Artificial Intelligence* 181 (2026) 115458.

## Format-Normalized Comparison

- Accepted bridge article: 13 publisher-formatted two-column pages and 10,914 extracted words.
- Kinetic-TopoGuard manuscript: 51 `elsarticle` preprint pages and 15,762 extracted words. The same source compiles to 25 pages in the Elsevier `5p,times,twocolumn` proof.
- Therefore, 51 versus 13 is not a valid direct comparison. The controlled comparison is 25 versus 13 pages. The FANET manuscript remains longer because it reports more model families, ablations, external datasets, statistical checks, and reproducibility detail.
- `paper/main_5p.pdf` is a format/length proof, not the anonymous portal manuscript. `paper/main_anonymized.pdf` remains the submission file.

## Gap-By-Gap Resolution

| Accepted-article strength | Previous FANET deficiency | Executed resolution | Measured/file evidence |
| --- | --- | --- | --- |
| Real measured case-study data | Real motion existed, but no measured RF layer | Integrated AERPAW aerial cellular data and the WiNES peer-to-peer UAV 60 GHz dataset | `outputs/aerpaw_cellular_validation/`, `outputs/uav_to_uav_mmwave_validation/` |
| Domain-model/data integration | Simulation, ML, and external checks were separate artefacts | Built one browser-based digital-twin evidence view binding simulation, 20-seed inference, real motion, A2A RF, AERPAW, and operating points | `outputs/digital_twin_dashboard/index.html` and `dashboard_payload.json` |
| Case-study prediction against measured response | External motion graphs were counterfactual | Added held-out long-distance A2A link viability and RF-calibrated expected topology on the forestry flight trace | Link F1 0.731 to 0.923; `uav_to_uav_mmwave_metrics.csv` |
| Operational decision rule | The manuscript discussed false alarms without selecting deployable thresholds | Executed threshold sweep and selected explicit alarm-rate-constrained operating points | Threshold 0.6: F1 0.754, 20.2 false alarms/min; threshold 0.7: F1 0.701, 9.6/min |
| Neural-family stability | Nine neural comparators had only three seeds | Executed two additional seeds for all nine PyTorch neural tasks and rebuilt a five-seed aggregate | 43,200 new snapshots; PyTorch backend; `outputs/publication_neural_5seed_extension/` |
| Confirmatory inference | Broad model table did not provide a high-seed primary comparison | Retained and audited the focused 20-seed Kinetic-TopoGuard versus shallow-ML study | MAE 0.308 versus 0.363; risk F1 0.587 versus 0.503 |
| Reproducible external-data workflow | Downloads, hashes, processing, and package gates were incomplete | Added checksum-verifying acquisition/analysis scripts, generated tables/figures, package inclusion, and readiness checks | `scripts/run_uav_to_uav_mmwave_validation.py`, `scripts/run_aerpaw_cellular_validation.py`, audit JSON |
| Submission-length interpretation | Single-column page count was compared with final two-column articles | Added an Elsevier 5p compile target and format-aware page gates | 51-page preprint; 25-page 5p proof; audit limits 55 and 30 pages |

## Quantitative Evidence Added

- WiNES UAV-to-UAV 60 GHz holdout: 1,756 long-distance test rows; baseline F1 0.731; logistic RF model F1 0.923; PR-AUC 0.985.
- RF-calibrated forestry topology: expected mean beta0 is 1.496, 1.915, and 2.339 at 5, 7, and 10 dB viability thresholds.
- AERPAW Dataset-22 LTE: F1 0.828 to 0.961. Dataset-23 LTE: F1 0.679 to 0.980.
- AERPAW Dataset-23 iPerf: MAE 223.0 to 75.2 Mbps; R2 0.713.
- Five-seed neural extension: Kinetic-TopoGuard MAE 0.309 and risk F1 0.560; the lowest neural-comparator MAE is GraphSAGE at 0.406.

## Claim Boundary After the Executed Work

The data gap is no longer closed only by manuscript wording: the repository now contains measured UAV-to-UAV RF, measured aerial cellular RF/throughput, real-flight motion, explicit operating-point selection, a unified dashboard, and five-seed neural-family execution. A synchronized multi-hop FANET campaign with packet reception, MAC contention, routing, and onboard hardware timing is a distinct physical experiment; no public dataset used here contains all of those synchronized labels. The manuscript does not claim that unexecuted experiment.

## Verification

- `python scripts/audit_submission_readiness.py`: format, evidence, metadata, and package gates.
- `python -m pytest -q --basetemp .pytest_tmp`: repository test suite.
- `paper/main.pdf`: 51-page preprint; `paper/main_anonymized.pdf`: 51-page anonymous manuscript.
- `paper/main_5p.pdf`: 25-page two-column format proof.
- `python scripts/build_anonymous_supplementary.py`: anonymous ZIP with external-data, operating-point, dashboard, and neural-extension evidence.
