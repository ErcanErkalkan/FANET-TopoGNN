# Real-flight motion external validation

This directory records the reproducible external-validation path based on the
public Zenodo dataset *Dataset of A Multi-Drone System Proof of Concept for
Forestry Applications* (DOI: `10.5281/zenodo.14701641`, CC BY 4.0).

The source contains field-recorded GNSS, IMU, LiDAR, odometry, and system data
from three UAVs. It does not contain packet-reception or RF link ground truth.
The validation therefore uses measured flight motion and deterministic
communication-radius sensitivity graphs. It must not be described as a
field-measured wireless-network evaluation.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_external_validation.ps1
python -m pip install -r requirements-external.txt
python scripts/extract_forestry_trace.py
python scripts/run_external_validation.py
```

The large original ROS bags remain ignored under `raw/`. The small aligned
trace and its provenance/checksum manifest are stored under `derived/`.
Evaluation tables, the protocol record, and the trajectory/radius figure are
written to `outputs/external_validation/`.

## Scope boundary

- Real evidence: synchronized positions and motion from a forestry field test.
- Time alignment: 10 Hz resampling preserves the synthetic benchmark's
  six-step, 0.6-second forecast horizon; gaps above 0.5 seconds are rejected.
- Counterfactual assumption: a link exists when pairwise distance is below the
  selected radius.
- Sensitivity radii: the 25th, 50th, and 75th percentiles of observed pairwise
  distance, selected without reference to model outcomes.
- Transfer protocol: models are fitted on the synthetic compact training split
  and evaluated without refitting on the measured flight trace.
- Still not covered: measured RF links, MAC contention, packet-level traffic,
  onboard execution, hardware-in-the-loop, or operational deployment.

## Attribution

Araujo, A.; Pontes Pizzino, C. A.; Couceiro, M.; Rocha, R. P. (2025),
*Dataset of A Multi-Drone System Proof of Concept for Forestry Applications*,
Zenodo. <https://doi.org/10.5281/zenodo.14701641>.
