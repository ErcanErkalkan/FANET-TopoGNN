from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Iterable
import numpy as np
import pandas as pd

from .geometry import density, normalize_positions, pairwise_distances
from .graph_utils import avg_clustering_coefficient, betti_zero, degree_features, largest_component_ratio, shortest_path_matrix
from .mobility import init_gm, init_mission, init_rwp, step_gm, step_mission, step_rwp
from .radio import TemporalRadioState, build_fixed_adaptive_adjacencies
from .topology import persistence_image


@dataclass
class Snapshot:
    run_id: str
    split_group_id: str
    time_index: int
    mobility: str
    n_nodes: int
    positions: np.ndarray
    velocities: np.ndarray
    node_features: np.ndarray
    adjacency: np.ndarray
    adjacency_fixed: np.ndarray
    adjacency_adaptive: np.ndarray
    pi: np.ndarray
    stats: np.ndarray
    beta_current: float
    beta_target: float
    beta_fixed: float
    beta_adaptive: float
    radius: float
    radius_fixed: float
    radius_adaptive: float
    edge_count_fixed: int
    edge_count_adaptive: int
    link_model: str
    graph_policy: str
    radio_scenario: str
    is_connected: int
    future_time_index: int
    frag_at_horizon: int


def adaptive_radius(points: np.ndarray, base_radius: float, density_threshold: float, adaptive_scale: float) -> float:
    return base_radius * adaptive_scale if density(points) > density_threshold else base_radius


def _velocity_features(velocities: np.ndarray, reference_speed: float) -> np.ndarray:
    scale = max(reference_speed, 1.0)
    return (velocities / scale).astype(np.float32)


def graph_stats(points: np.ndarray, velocities: np.ndarray, adj: np.ndarray, beta_current: float) -> np.ndarray:
    dists = pairwise_distances(points)
    mean_deg, max_deg, min_deg = degree_features(adj)
    sp = shortest_path_matrix(adj)
    finite = sp[np.isfinite(sp)]
    tri = dists[np.triu_indices_from(dists, k=1)]
    speed = np.linalg.norm(velocities, axis=1)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    radial_norm = np.linalg.norm(centered, axis=1)
    radial_norm[radial_norm < 1e-6] = 1.0
    radial_unit = centered / radial_norm[:, None]
    radial_velocity = np.sum(radial_unit * velocities, axis=1)
    edge_density = adj.sum() / max(adj.shape[0] * max(adj.shape[0] - 1, 1), 1)
    return np.array(
        [
            density(points),
            float(tri.min()) if tri.size else 0.0,
            float(tri.mean()) if tri.size else 0.0,
            float(tri.max()) if tri.size else 0.0,
            mean_deg,
            max_deg,
            min_deg,
            largest_component_ratio(adj),
            avg_clustering_coefficient(adj),
            float(finite.mean()) if finite.size else 0.0,
            float(beta_current),
            float(speed.mean()),
            float(speed.std()),
            float(radial_norm.mean()),
            float(radial_velocity.mean()),
            float(radial_velocity.std()),
            float(edge_density),
        ],
        dtype=np.float32,
    )


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _graph_policies(sim_cfg: dict) -> list[str]:
    configured = sim_cfg.get("graph_policies")
    if configured is None:
        configured = [sim_cfg.get("graph_policy", "adaptive")]
    policies = [_normalise_label(policy) for policy in configured]
    invalid = [policy for policy in policies if policy not in {"fixed", "adaptive"}]
    if invalid:
        raise ValueError(f"Unknown graph_policies entries: {invalid}")
    return policies


def _scenario_physical_overrides(scenario: dict) -> dict:
    if "physical_layer" in scenario and isinstance(scenario["physical_layer"], dict):
        return dict(scenario["physical_layer"])
    return {key: value for key, value in scenario.items() if key != "name"}


def _radio_scenarios(sim_cfg: dict) -> list[tuple[str, dict]]:
    scenarios = sim_cfg.get("radio_scenarios")
    if not scenarios:
        return [(_normalise_label(sim_cfg.get("radio_scenario", "default")), copy.deepcopy(sim_cfg))]

    expanded: list[tuple[str, dict]] = []
    base_physical = dict(sim_cfg.get("physical_layer", {}))
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError("radio_scenarios entries must be objects")
        name = _normalise_label(scenario.get("name", f"scenario_{idx + 1}"))
        scenario_cfg = copy.deepcopy(sim_cfg)
        scenario_cfg.pop("radio_scenarios", None)
        physical = dict(base_physical)
        physical.update(_scenario_physical_overrides(scenario))
        scenario_cfg["physical_layer"] = physical
        scenario_cfg["radio_scenario"] = name
        expanded.append((name, scenario_cfg))
    return expanded


def _kinematic_run(sim_cfg: dict, mobility_name: str, n_nodes: int, run_seed: int) -> list[dict]:
    rng = np.random.default_rng(run_seed)
    area_size = sim_cfg["area_size"]
    dt = sim_cfg["dt"]
    speed_range = tuple(sim_cfg["speed_range"])
    if mobility_name == "rwp":
        state = init_rwp(rng, n_nodes, area_size, speed_range)
        step_fn = lambda st: step_rwp(st, rng, dt, area_size, speed_range)
    elif mobility_name == "gm":
        state = init_gm(rng, n_nodes, area_size, sim_cfg["gauss_markov_mu"])
        step_fn = lambda st: step_gm(st, rng, dt, area_size, sim_cfg["gauss_markov_alpha"], sim_cfg["gauss_markov_mu"], sim_cfg["gauss_markov_sigma"])
    else:
        state = init_mission(rng, n_nodes, area_size, sim_cfg["mission_regions"])
        step_fn = lambda st: step_mission(
            st,
            rng,
            dt,
            area_size,
            sim_cfg["mission_regions"],
            tracking_noise_m=sim_cfg.get("mission_tracking_noise_m", 2.0),
            heading_noise_rad=sim_cfg.get("mission_heading_noise_rad", 0.05),
            speed_jitter=sim_cfg.get("mission_speed_jitter", 0.08),
        )

    velocity_scale = max(speed_range[1], sim_cfg.get("gauss_markov_mu", speed_range[1]), 30.0)
    kinematic_items: list[dict] = []
    for time_index in range(sim_cfg["time_steps"]):
        positions = state.position.astype(np.float32)
        velocities = state.velocity.astype(np.float32)
        distances = pairwise_distances(positions)
        pi = persistence_image(
            positions,
            sim_cfg["pi_resolution"],
            sim_cfg["pi_sigma"],
            sim_cfg["pi_max_radius"],
        ).reshape(-1)
        node_features = np.concatenate(
            [normalize_positions(positions), _velocity_features(velocities, velocity_scale)],
            axis=1,
        ).astype(np.float32)
        kinematic_items.append(
            {
                "time_index": time_index,
                "positions": positions,
                "velocities": velocities,
                "distances": distances,
                "pi": pi.astype(np.float32),
                "node_features": node_features,
            }
        )
        state = step_fn(state)
    return kinematic_items


def _raw_run(
    sim_cfg: dict,
    mobility_name: str,
    n_nodes: int,
    run_seed: int,
    kinematic_items: list[dict] | None = None,
) -> list[dict]:
    graph_policy = _normalise_label(sim_cfg.get("graph_policy", "adaptive"))
    link_model = str(sim_cfg.get("link_model", "radius")).lower()
    radio_scenario = _normalise_label(sim_cfg.get("radio_scenario", "default"))
    raw_items: list[dict] = []
    split_group_id = f"{mobility_name}_N{n_nodes}_seed{run_seed}"
    run_id = f"{mobility_name}_N{n_nodes}_{graph_policy}_{radio_scenario}_seed{run_seed}"
    base_items = kinematic_items or _kinematic_run(sim_cfg, mobility_name, n_nodes, run_seed)
    radio_state = TemporalRadioState()
    for base in base_items:
        time_index = int(base["time_index"])
        positions = base["positions"]
        velocities = base["velocities"]
        distances = base["distances"]
        radius_fixed = float(sim_cfg["base_radius"])
        radius_adaptive = adaptive_radius(positions, radius_fixed, sim_cfg["density_threshold"], sim_cfg["adaptive_scale"])
        link_seed = int((run_seed * 1_000_003 + time_index) % (2**63 - 1))
        adjacency_fixed, adjacency_adaptive = build_fixed_adaptive_adjacencies(
            distances,
            radius_fixed,
            radius_adaptive,
            np.random.default_rng(link_seed),
            sim_cfg,
            state=radio_state,
        )
        beta_fixed = float(betti_zero(adjacency_fixed))
        beta_adaptive = float(betti_zero(adjacency_adaptive))
        if graph_policy == "fixed":
            adjacency = adjacency_fixed
            beta_current = beta_fixed
            radius = radius_fixed
        elif graph_policy == "adaptive":
            adjacency = adjacency_adaptive
            beta_current = beta_adaptive
            radius = radius_adaptive
        else:
            raise ValueError(f"Unknown graph_policy: {graph_policy}")
        raw_items.append(
            {
                "run_id": run_id,
                "split_group_id": split_group_id,
                "time_index": time_index,
                "mobility": mobility_name,
                "n_nodes": n_nodes,
                "positions": positions,
                "velocities": velocities,
                "node_features": base["node_features"],
                "adjacency": adjacency.astype(np.float32),
                "adjacency_fixed": adjacency_fixed.astype(np.float32),
                "adjacency_adaptive": adjacency_adaptive.astype(np.float32),
                "pi": base["pi"],
                "beta_current": beta_current,
                "beta_fixed": beta_fixed,
                "beta_adaptive": beta_adaptive,
                "radius": float(radius),
                "radius_fixed": radius_fixed,
                "radius_adaptive": float(radius_adaptive),
                "edge_count_fixed": int(adjacency_fixed.sum() / 2),
                "edge_count_adaptive": int(adjacency_adaptive.sum() / 2),
                "link_model": link_model,
                "graph_policy": graph_policy,
                "radio_scenario": radio_scenario,
            }
        )
    return raw_items


def simulate_run(
    sim_cfg: dict,
    mobility_name: str,
    n_nodes: int,
    run_seed: int,
    kinematic_items: list[dict] | None = None,
) -> list[Snapshot]:
    raw_items = _raw_run(sim_cfg, mobility_name, n_nodes, run_seed, kinematic_items)
    horizon = int(sim_cfg.get("forecast_horizon_steps", sim_cfg.get("warning_horizon_steps", 1)))
    snapshots: list[Snapshot] = []
    for idx, item in enumerate(raw_items):
        future_idx = min(idx + horizon, len(raw_items) - 1)
        future_beta = float(raw_items[future_idx]["beta_current"])
        frag_at_horizon = int(future_beta > 1)
        stats = graph_stats(item["positions"], item["velocities"], item["adjacency"], item["beta_current"])
        snapshots.append(
            Snapshot(
                run_id=item["run_id"],
                split_group_id=item["split_group_id"],
                time_index=item["time_index"],
                mobility=item["mobility"],
                n_nodes=item["n_nodes"],
                positions=item["positions"],
                velocities=item["velocities"],
                node_features=item["node_features"],
                adjacency=item["adjacency"],
                adjacency_fixed=item["adjacency_fixed"],
                adjacency_adaptive=item["adjacency_adaptive"],
                pi=item["pi"],
                stats=stats,
                beta_current=item["beta_current"],
                beta_target=future_beta,
                beta_fixed=item["beta_fixed"],
                beta_adaptive=item["beta_adaptive"],
                radius=item["radius"],
                radius_fixed=item["radius_fixed"],
                radius_adaptive=item["radius_adaptive"],
                edge_count_fixed=item["edge_count_fixed"],
                edge_count_adaptive=item["edge_count_adaptive"],
                link_model=item["link_model"],
                graph_policy=item["graph_policy"],
                radio_scenario=item["radio_scenario"],
                is_connected=int(item["beta_current"] == 1),
                future_time_index=raw_items[future_idx]["time_index"],
                frag_at_horizon=frag_at_horizon,
            )
        )
    return snapshots


def build_dataset(sim_cfg: dict, seed: int) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    radio_scenarios = _radio_scenarios(sim_cfg)
    graph_policies = _graph_policies(sim_cfg)
    kinematic_cache = {}
    for mobility_idx, mobility_name in enumerate(sim_cfg["mobility_models"]):
        for size_idx, n_nodes in enumerate(sim_cfg["swarm_sizes"]):
            for run_idx in range(sim_cfg["runs_per_setting"]):
                run_seed = seed + mobility_idx * 10000 + size_idx * 100 + run_idx
                kinematic_cache[(mobility_name, n_nodes, run_seed)] = _kinematic_run(
                    sim_cfg,
                    mobility_name,
                    n_nodes,
                    run_seed,
                )
    for scenario_name, scenario_cfg in radio_scenarios:
        for graph_policy in graph_policies:
            run_cfg = copy.deepcopy(scenario_cfg)
            run_cfg["graph_policy"] = graph_policy
            run_cfg["radio_scenario"] = scenario_name
            for mobility_idx, mobility_name in enumerate(sim_cfg["mobility_models"]):
                for size_idx, n_nodes in enumerate(sim_cfg["swarm_sizes"]):
                    for run_idx in range(sim_cfg["runs_per_setting"]):
                        run_seed = seed + mobility_idx * 10000 + size_idx * 100 + run_idx
                        snapshots.extend(
                            simulate_run(
                                run_cfg,
                                mobility_name,
                                n_nodes,
                                run_seed,
                                kinematic_cache[(mobility_name, n_nodes, run_seed)],
                            )
                        )
    return snapshots


def relabel_forecast_horizon(snapshots: list[Snapshot], horizon_steps: int) -> list[Snapshot]:
    """Reuse fixed trajectories while changing only deterministic future labels."""
    horizon = max(int(horizon_steps), 0)
    relabeled: list[Snapshot] = []
    for sequence in split_by_run(snapshots).values():
        ordered = sorted(sequence, key=lambda item: item.time_index)
        for idx, snapshot in enumerate(ordered):
            future = ordered[min(idx + horizon, len(ordered) - 1)]
            relabeled.append(
                replace(
                    snapshot,
                    beta_target=float(future.beta_current),
                    future_time_index=int(future.time_index),
                    frag_at_horizon=int(future.beta_current > 1),
                )
            )
    return relabeled


def to_frame(snapshots: Iterable[Snapshot]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": s.run_id,
                "split_group_id": s.split_group_id,
                "time_index": s.time_index,
                "mobility": s.mobility,
                "n_nodes": s.n_nodes,
                "beta_current": s.beta_current,
                "beta_target": s.beta_target,
                "beta_fixed": s.beta_fixed,
                "beta_adaptive": s.beta_adaptive,
                "future_time_index": s.future_time_index,
                "radius": s.radius,
                "radius_fixed": s.radius_fixed,
                "radius_adaptive": s.radius_adaptive,
                "edge_count": s.edge_count_fixed if s.graph_policy == "fixed" else s.edge_count_adaptive,
                "edge_count_fixed": s.edge_count_fixed,
                "edge_count_adaptive": s.edge_count_adaptive,
                "link_model": s.link_model,
                "graph_policy": s.graph_policy,
                "radio_scenario": s.radio_scenario,
                "is_connected": s.is_connected,
                "frag_at_horizon": s.frag_at_horizon,
            }
            for s in snapshots
        ]
    )


def split_by_run(snapshots: list[Snapshot]) -> dict[str, list[Snapshot]]:
    runs: dict[str, list[Snapshot]] = {}
    for snap in snapshots:
        runs.setdefault(snap.run_id, []).append(snap)
    return runs


def _split_counts(n_items: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    n_train = max(1, int(round(train_fraction * n_items)))
    n_val = max(1, int(round(val_fraction * n_items))) if n_items >= 3 else 0
    n_test = n_items - n_train - n_val
    while n_test < (1 if n_items >= 2 else 0) and n_train > 1:
        n_train -= 1
        n_test += 1
    while n_test < (1 if n_items >= 3 else 0) and n_val > 1:
        n_val -= 1
        n_test += 1
    return n_train, n_val, max(n_test, 0)


def _stable_group_seed(split_seed: int, key: tuple[object, ...]) -> int:
    text = "|".join(str(item) for item in key)
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(text))
    return int((split_seed + offset) % (2**63 - 1))


def assign_run_splits(
    snapshots: list[Snapshot],
    split_seed: int = 0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    stratify_by: tuple[str, ...] = ("mobility",),
) -> tuple[dict[str, str], pd.DataFrame]:
    run_map = split_by_run(snapshots)
    run_meta = {}
    for run_id, items in run_map.items():
        first = items[0]
        run_meta[run_id] = {
            "run_id": run_id,
            "split_group_id": first.split_group_id,
            "mobility": first.mobility,
            "n_nodes": first.n_nodes,
            "link_model": first.link_model,
            "graph_policy": first.graph_policy,
            "radio_scenario": first.radio_scenario,
            "snapshots": len(items),
        }
    split_units: dict[str, dict] = {}
    for run_id, meta in run_meta.items():
        unit = split_units.setdefault(
            meta["split_group_id"],
            {
                "split_group_id": meta["split_group_id"],
                "mobility": meta["mobility"],
                "n_nodes": meta["n_nodes"],
                "link_model": meta["link_model"],
                "run_ids": [],
            },
        )
        unit["run_ids"].append(run_id)

    variant_columns = {"graph_policy", "radio_scenario", "run_id", "snapshots"}
    effective_stratify = tuple(name for name in stratify_by if name not in variant_columns)
    if not effective_stratify:
        effective_stratify = ("mobility",)

    groups: dict[tuple[object, ...], list[str]] = {}
    for split_group_id, meta in split_units.items():
        key = tuple(meta[name] for name in effective_stratify)
        groups.setdefault(key, []).append(split_group_id)

    mapping: dict[str, str] = {}
    rows = []
    for key, ids in sorted(groups.items(), key=lambda item: item[0]):
        ids = sorted(ids)
        rng = np.random.default_rng(_stable_group_seed(split_seed, key))
        shuffled = np.asarray(ids, dtype=object)
        rng.shuffle(shuffled)
        ids = shuffled.tolist()
        n_train, n_val, _ = _split_counts(len(ids), train_fraction, val_fraction)
        for idx, split_group_id in enumerate(ids):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            for run_id in split_units[split_group_id]["run_ids"]:
                mapping[run_id] = split
    for run_id in sorted(run_meta):
        row = dict(run_meta[run_id])
        row["split"] = mapping[run_id]
        rows.append(row)
    return mapping, pd.DataFrame(rows)


def train_val_test_split(
    snapshots: list[Snapshot],
    split_seed: int = 0,
    stratify_by: tuple[str, ...] = ("mobility",),
    return_mapping: bool = False,
) -> tuple[list[Snapshot], list[Snapshot], list[Snapshot]] | tuple[list[Snapshot], list[Snapshot], list[Snapshot], pd.DataFrame]:
    run_map = split_by_run(snapshots)
    split_mapping, split_frame = assign_run_splits(snapshots, split_seed=split_seed, stratify_by=stratify_by)
    train, val, test = [], [], []
    for run_id, items in run_map.items():
        split = split_mapping[run_id]
        if split == "train":
            train.extend(items)
        elif split == "val":
            val.extend(items)
        else:
            test.extend(items)
    if not train or not val or not test:
        raise ValueError("Run-wise split produced an empty train, validation, or test partition")
    if return_mapping:
        return train, val, test, split_frame
    return train, val, test
