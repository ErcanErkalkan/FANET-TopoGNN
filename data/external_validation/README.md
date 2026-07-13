# External validation datasets

This directory records reproducible external-validation paths based on public
flight and aerial cellular datasets.

## Real-flight motion transfer

The field-motion path uses the public Zenodo dataset *Dataset of A Multi-Drone
System Proof of Concept for Forestry Applications* (DOI:
`10.5281/zenodo.14701641`, CC BY 4.0).

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

## Measured AERPAW cellular RF/KPI validation

```powershell
python -m pip install -r requirements-external.txt
powershell -ExecutionPolicy Bypass -File scripts/download_aerpaw_cellular_validation.ps1
python scripts/run_aerpaw_cellular_validation.py
```

This path uses public AERPAW Dataset-22 and Dataset-23 CSV folders. The raw
Google Drive downloads remain ignored under `raw/`; the derived metrics,
protocol manifest, manuscript table, and validation figure are written to
`outputs/aerpaw_cellular_validation/` and `paper/*/generated/`.

The cellular evidence contains measured UAV-to-base-station LTE/5G KPIs and
iPerf throughput. It must not be described as synchronized inter-UAV FANET
packet-link validation.

## Measured UAV-to-UAV 60 GHz links

```powershell
python scripts/run_uav_to_uav_mmwave_validation.py
```

The WiNES source supplies measured peer-link channel observations. The held-out
test trains below 33 m and tests at 33 m or farther using a joint SNR and
received-power viability rule. Applying its distance-success relationship to
the forestry motion is a transported sensitivity, not same-site calibration.
Primary topology summaries require every pair to remain within the measured
6--40 m distance support.

## MILUV measured three-UAV UWB topology

```powershell
python scripts/download_miluv_validation.py
python scripts/run_miluv_validation.py
```

MILUV (dataset DOI `10.25452/figshare.plus.28386041.v1`) provides Vicon motion
and UWB first-path-power observations for three quadcopters. The downloader
range-extracts six required CSV members from the official archive and records
archive CRC plus per-file SHA-256 values. The evaluation uses causal
forward-hold reconstruction at 10 Hz and frozen simulation-trained models.
The resulting topology labels represent UWB ranging-message quality, not IP
packet delivery.

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
- Still not covered: synchronized outdoor FANET IP packets, onboard execution,
  hardware-in-the-loop control, or operational deployment.
- AERPAW cellular boundary: measured aerial RF/KPI and throughput are included,
  but peer-to-peer inter-UAV packet labels are not.
- MILUV boundary: inter-robot UWB RF quality is measured, but the environment is
  indoor, the swarm has three UAVs, and the labels are not IP PDR.

## Attribution

Araujo, A.; Pontes Pizzino, C. A.; Couceiro, M.; Rocha, R. P. (2025),
*Dataset of A Multi-Drone System Proof of Concept for Forestry Applications*,
Zenodo. <https://doi.org/10.5281/zenodo.14701641>.
