from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fanet.external_validation import (
    FlightTrace,
    build_trace_snapshots,
    load_flight_trace_csv,
    pairwise_distance_quantiles,
    trace_velocities,
)


def _trace() -> FlightTrace:
    timestamps = np.array([0.0, 1.0, 2.0])
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [14.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return FlightTrace(timestamps, ("uav6", "uav8", "uav9"), positions)


def test_trace_builder_labels_future_fragmentation() -> None:
    snapshots = build_trace_snapshots(_trace(), radius_m=6.5, horizon_steps=1, pi_resolution=4)
    assert len(snapshots) == 3
    assert snapshots[0].beta_current == 1.0
    assert snapshots[1].beta_current == 1.0
    assert snapshots[1].beta_target == 3.0
    assert snapshots[1].frag_at_horizon == 1
    assert snapshots[0].node_features.shape == (3, 6)
    assert np.allclose(trace_velocities(_trace())[:, 1, 0], 1.0)


def test_trace_csv_round_trip_and_distance_quantiles(tmp_path) -> None:
    rows = []
    trace = _trace()
    for time_idx, timestamp in enumerate(trace.timestamps_s):
        for vehicle_idx, vehicle_id in enumerate(trace.vehicle_ids):
            x, y, z = trace.positions_m[time_idx, vehicle_idx]
            rows.append({"timestamp_s": timestamp, "vehicle_id": vehicle_id, "x_m": x, "y_m": y, "z_m": z})
    path = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    loaded = load_flight_trace_csv(path)
    assert loaded.vehicle_ids == trace.vehicle_ids
    assert np.allclose(loaded.positions_m, trace.positions_m)
    quantiles = pairwise_distance_quantiles(loaded, (0.5,))
    assert quantiles[0.5] == pytest.approx(7.0)


def test_trace_validation_rejects_duplicate_timestamps() -> None:
    trace = _trace()
    invalid = FlightTrace(np.array([0.0, 1.0, 1.0]), trace.vehicle_ids, trace.positions_m)
    with pytest.raises(ValueError, match="strictly increasing"):
        invalid.validate()
