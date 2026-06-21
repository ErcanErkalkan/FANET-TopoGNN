from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Snapshot, graph_stats
from .geometry import normalize_positions, pairwise_distances
from .graph_utils import betti_zero
from .radio import build_fixed_adaptive_adjacencies
from .topology import persistence_image


@dataclass(frozen=True)
class FlightTrace:
    timestamps_s: np.ndarray
    vehicle_ids: tuple[str, ...]
    positions_m: np.ndarray

    def validate(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=float)
        positions = np.asarray(self.positions_m, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("A flight trace needs at least two timestamps")
        if positions.shape != (len(timestamps), len(self.vehicle_ids), 3):
            raise ValueError("positions_m must have shape [time, vehicle, 3]")
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(positions)):
            raise ValueError("Flight trace contains non-finite values")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Flight trace timestamps must be strictly increasing")


def load_flight_trace_csv(path: str | Path) -> FlightTrace:
    frame = pd.read_csv(path)
    required = {"timestamp_s", "vehicle_id", "x_m", "y_m", "z_m"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Flight trace CSV is missing columns: {sorted(missing)}")
    vehicle_ids = tuple(sorted(frame["vehicle_id"].astype(str).unique()))
    timestamps = np.sort(frame["timestamp_s"].astype(float).unique())
    expected = len(vehicle_ids) * len(timestamps)
    if len(frame) != expected:
        raise ValueError("Flight trace CSV must contain one row per timestamp and vehicle")
    positions = np.empty((len(timestamps), len(vehicle_ids), 3), dtype=np.float32)
    indexed = frame.set_index(["timestamp_s", "vehicle_id"])
    for time_idx, timestamp in enumerate(timestamps):
        for vehicle_idx, vehicle_id in enumerate(vehicle_ids):
            try:
                row = indexed.loc[(timestamp, vehicle_id)]
            except KeyError as exc:
                raise ValueError(f"Missing sample for {vehicle_id} at {timestamp}") from exc
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate sample for {vehicle_id} at {timestamp}")
            positions[time_idx, vehicle_idx] = row[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    trace = FlightTrace(timestamps_s=timestamps, vehicle_ids=vehicle_ids, positions_m=positions)
    trace.validate()
    return trace


def trace_velocities(trace: FlightTrace) -> np.ndarray:
    trace.validate()
    return np.gradient(
        trace.positions_m.astype(float),
        trace.timestamps_s.astype(float),
        axis=0,
        edge_order=1,
    ).astype(np.float32)


def build_trace_snapshots(
    trace: FlightTrace,
    radius_m: float,
    horizon_steps: int,
    pi_resolution: int = 16,
    pi_sigma: float = 20.0,
    pi_max_radius: float = 600.0,
    run_label: str = "forestry_field_trace",
) -> list[Snapshot]:
    """Build deterministic radius-graph snapshots from measured flight motion.

    The public field bags do not contain packet reception or RF ground truth.
    These snapshots therefore represent a communication-radius sensitivity
    analysis over measured positions, not a field-measured network trace.
    """
    trace.validate()
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be at least one")

    positions_all = trace.positions_m.astype(np.float32)
    velocities_all = trace_velocities(trace)
    dt = float(np.median(np.diff(trace.timestamps_s)))
    velocity_scale = max(float(np.quantile(np.linalg.norm(velocities_all, axis=2), 0.99)), 30.0)
    raw: list[dict] = []
    radius_cfg = {"link_model": "radius"}
    for time_idx, (positions, velocities) in enumerate(zip(positions_all, velocities_all)):
        distances = pairwise_distances(positions)
        adjacency, _ = build_fixed_adaptive_adjacencies(
            distances,
            float(radius_m),
            float(radius_m),
            np.random.default_rng(0),
            radius_cfg,
        )
        beta = float(betti_zero(adjacency))
        pi = persistence_image(positions, pi_resolution, pi_sigma, pi_max_radius).reshape(-1)
        node_features = np.concatenate(
            [normalize_positions(positions), velocities / velocity_scale],
            axis=1,
        ).astype(np.float32)
        raw.append(
            {
                "positions": positions,
                "velocities": velocities,
                "adjacency": adjacency.astype(np.float32),
                "beta": beta,
                "pi": pi.astype(np.float32),
                "node_features": node_features,
            }
        )

    snapshots: list[Snapshot] = []
    radius_label = f"r{float(radius_m):g}".replace(".", "p")
    run_id = f"{run_label}_{radius_label}"
    for time_idx, item in enumerate(raw):
        future_idx = min(time_idx + horizon_steps, len(raw) - 1)
        future_beta = float(raw[future_idx]["beta"])
        adjacency = item["adjacency"]
        edge_count = int(adjacency.sum() / 2)
        snapshots.append(
            Snapshot(
                run_id=run_id,
                split_group_id=run_id,
                time_index=time_idx,
                mobility="field_trace",
                n_nodes=len(trace.vehicle_ids),
                positions=item["positions"],
                velocities=item["velocities"],
                node_features=item["node_features"],
                adjacency=adjacency,
                adjacency_fixed=adjacency,
                adjacency_adaptive=adjacency.copy(),
                pi=item["pi"],
                stats=graph_stats(item["positions"], item["velocities"], adjacency, item["beta"]),
                beta_current=item["beta"],
                beta_target=future_beta,
                beta_fixed=item["beta"],
                beta_adaptive=item["beta"],
                radius=float(radius_m),
                radius_fixed=float(radius_m),
                radius_adaptive=float(radius_m),
                edge_count_fixed=edge_count,
                edge_count_adaptive=edge_count,
                link_model="radius_counterfactual",
                graph_policy="fixed",
                radio_scenario="field_motion_only",
                is_connected=int(item["beta"] == 1.0),
                future_time_index=future_idx,
                frag_at_horizon=int(future_beta > 1.0),
            )
        )
    return snapshots


def pairwise_distance_quantiles(trace: FlightTrace, quantiles: tuple[float, ...]) -> dict[float, float]:
    trace.validate()
    distances = []
    for positions in trace.positions_m:
        matrix = pairwise_distances(positions)
        distances.extend(matrix[np.triu_indices_from(matrix, k=1)].tolist())
    values = np.asarray(distances, dtype=float)
    return {float(q): float(np.quantile(values, q)) for q in quantiles}
