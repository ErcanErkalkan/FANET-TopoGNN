# FANET-TopoGNN

Reproducible research code and frozen development evidence for fragmentation-warning reliability in **simulated flying ad hoc networks**.

## Research question

Can a fixed warning threshold maintain an analytical false-alarm budget across network sizes, mobility conditions and independent development blocks? The evaluation concerns simulated graph fragmentation, not packet delivery or real-world deployment.

## Frozen evaluation

| Quantity | Result |
|---|---:|
| Paired runs | 180 |
| Trajectory groups | 45 |
| Simulated exposure | 60 minutes |
| Fragmentation onsets | 1,114 |
| Matched onsets | 183 (16.4% recall) |
| False alarms | 88 (1.47/min) |
| Event F1 | 0.264 |
| Cells exceeding 2 false alarms/min | 6 of 15 block-by-mobility cells |
| 30-node scenarios | 82 onsets, 0 detected |

The fixed operating point is 0.15. The analytical 2/min budget is not a validated operational requirement. The retrospective History-4 comparison is post-hoc. None of the development experiments provides a prospective blind test, packet-level network assessment or evidence of prediction superiority.

## Repository layout

- `core/`: simulation, radio-link abstraction, models, metrics and reproducible code.
- `frozen/`: fixed experiment plans, calibration anchors and scripts.
- `audit/`: independent scoring, provenance verification and scientific regression tests.
- `data_derived/`: verified derived tables and sensitivity analyses.
- `snapshot/research.part00`–`research.part13`: transport chunks of the complete research-only ZIP, including large CSVs and verification receipts absent from the lightweight Git tree.
- `run.py`: reproducibility entry point.

**Research code and evidence only.** Article text, figures prepared for publication, supplementary manuscripts and submission documents are maintained separately from the public research repository.

## Reconstruct and verify

Use Python 3.11 or newer. From the repository root:

```bash
python assemble_snapshot.py
python -B run.py check
python -B run.py verify
python -B run.py test
```

The first command reconstructs and verifies the compressed research-only snapshot and restores the recorded CSVs and receipts without overwriting a different local version. The remaining commands verify integrity, audit records and tests. `run.py pilot` requires the documented CPU simulation dependencies and is a deterministic re-verification, not a new independent evaluation.

## Experimental boundaries

- Two-dimensional, thresholded received-power connectivity; no measured packet-delivery ratio, MAC-layer behavior or real UAV field trials.
- Frozen thresholds and event-matching logic; no model tuning on the reported development evaluation blocks.
- Four radio/graph variants share each trajectory realization and are not independent observations.
- Model input is a single graph snapshot; the target is a six-step endpoint, distinct from onset-anywhere matching.
- No prospective blind-holdout results or supported deployment threshold.

## Integrity and status

`MANIFEST.json` contains SHA-256 checksums for all included research files. The snapshot archive is authenticated by `assemble_snapshot.py`. Original experimental outputs, selection decisions and model weights were not regenerated or changed in preparing the code-only distribution. Research status: **development evidence / not deployment approved**.

The licensed software terms are retained in `core/LICENSE`. Citation of the research snapshot must identify a specific repository commit and snapshot hash; no DOI-registered archival release is claimed.
