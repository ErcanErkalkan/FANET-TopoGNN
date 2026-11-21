# FANET-TopoGNN

**Reproducible benchmark of fragmentation-warning reliability in simulated flying ad hoc networks.**

This repository hosts one current research package. The project investigates whether an alarm threshold selected on development data maintains a fixed false-alarm budget across mobility regimes, node counts, graph/radio conditions, and development blocks. Its code, dataset, experimental evidence and scientific conclusions are not tied to a publisher or venue. The manuscript presentation requirements are specified separately in `paper/manuscript/FORMAT_REQUIREMENTS.md`.

## Scope and principal findings

The frozen experiment contains **45 trajectory groups**, each evaluated under four paired radio/graph conditions (**180 runs**, **60 simulated minutes**). At a frozen threshold of **0.15**, 183 of 1,114 network-fragmentation onsets were matched (event recall **16.4%**, event F1 **0.264**) and there were 88 unmatched alarms (**1.47 false alarms/min** over full simulated exposure). The pooled rate is below the chosen analytical budget of two/min, while six of 15 block-by-mobility strata exceed it. At 30 nodes, the system emitted no alarms and missed all 82 events. The retrospective history-only reference captured more events but exceeded the pooled false-alarm budget. These results are descriptive development evidence, not independent blind test performance, packet-delivery validation, or proof of model superiority.

## Project layout

- `core/`: simulation, radio-graph construction, model and test code.
- `frozen/`: fixed experiment plan, training/selection inputs and calibration anchors (scientific provenance, **not competing releases**).
- `audit/`: verification tools, independent event scoring and reproducibility checks.
- `data_derived/`: reproducible summary tables and retrospective sensitivity analysis.
- `paper/manuscript/`: one current research draft and technical supplement; the scientific content remains neutral while the layout follows the specified publication format.
- `evidence/`: evidence and test receipts. Stage identifiers are audit provenance only, not active project versions.
- `run.py`: project entry point.

## Reproduce and verify

Python 3.11+ is sufficient for checksum validation and stored-result verification:

```bash
python run.py check
python run.py verify
python run.py test
```

Full CPU simulation reruns require the pinned Python / PyTorch / NumPy / SciPy / scikit-learn environment recorded in `frozen/anchor/`. Do not treat re-verification as new training or an independent holdout.

```bash
python run.py pilot
python run.py build-paper
```

The manuscript builder uses `pdflatex` and requires `bibtex` (or `bibtex8`). A separate XeLaTeX rendering is possible for exact system-font selection. New outputs are written under `outputs/` rather than overwriting frozen records.

## Scientific limitations

- The simulator models a **2-D received-power radio connectivity graph**, not PHY/MAC packet contention, routing convergence, throughput, packet delivery, or real UAV operations.
- The selected warning budget (2 false alarms per simulated minute) is an **analytical reference**, not an empirically validated service-level requirement.
- The model predicts a fixed-horizon endpoint state, whereas an alarm matches an onset anywhere in the lead window.
- The History-4 comparison was designed after inspecting original results; it is **post-hoc**, not confirmatory.
- Authors, disclosures, external data release and publication decisions remain subject to explicit human approval.

## Data access and citation

The full project snapshot includes recorded per-frame component-count truth and decision scores; it does not ship raw per-UAV coordinate trajectories, which are reproducible from the frozen generator. See `audit/reference_linux/`, `audit/reference_windows/` and `evidence/pilot_linux/`. Avoid citing historical GitHub or Zenodo records as if they archive the current snapshot. Use a snapshot-specific hash and versioned permalink after approving a public release.

## Licence and status

Project evidence remains **RESEARCH / AUTHOR REVIEW — NOT DEPLOYMENT APPROVED**. The original experimental records and selected thresholds have not been altered. `core/LICENSE` contains the preserved software licence. The latest package is the only active research snapshot; previous editorial/revision packages are superseded.

## Figure-enriched manuscript

The current manuscript contains five labeled vector figures. Methods/targets are explicitly schematic; experimental charts are tied to frozen CSVs (`paper/FIGURE_PROVENANCE.md`). The manuscript layout follows the chosen venue while the research findings and repository remain independent of the venue.
