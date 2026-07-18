from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .dataset import Snapshot
from .evaluation import _controlled_physical_adjacency, _select_relays
from .graph_utils import connected_components


POLICIES = (
    "No intervention",
    "Reactive intervention",
    "Persistence-triggered intervention",
    "Current-state ExtraTrees intervention",
    "Original Kinetic-TopoGuard intervention",
    "Source-Gated Kinetic-TopoGuard intervention",
)


@dataclass(frozen=True)
class RelayActionConfig:
    dt_s: float = 0.1
    max_speed_mps: float = 30.0
    max_acceleration_mps2: float = 12.0
    reaction_delay_steps: int = 2
    action_duration_steps: int = 12
    cooldown_steps: int = 6
    max_relays: int = 2
    radius_boost: float = 1.15
    relay_link_budget_boost_db: float = 3.0
    min_trigger_separation_steps: int = 2
    energy_speed_weight: float = 1.0
    energy_acceleration_weight: float = 0.1


@dataclass
class ControllerState:
    offsets: np.ndarray
    commanded_velocities: np.ndarray
    relays: list[int] = field(default_factory=list)
    pending_steps: int = -1
    active_steps: int = 0
    cooldown_steps: int = 0
    consecutive_trigger_steps: int = 0
    intervention_count: int = 0
    cumulative_travel_m: float = 0.0
    cumulative_energy_proxy: float = 0.0


def initial_state(n_nodes: int) -> ControllerState:
    return ControllerState(
        offsets=np.zeros((int(n_nodes), 2), dtype=float),
        commanded_velocities=np.zeros((int(n_nodes), 2), dtype=float),
    )


def traffic_realization_seed(seed: int, run_id: str, time_index: int, load: int) -> int:
    payload = f"{int(seed)}:{run_id}:{int(time_index)}:{int(load)}:packet-arrivals"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") % (2**63 - 1)


def _relay_targets(
    positions: np.ndarray,
    adjacency: np.ndarray,
    relays: list[int],
) -> dict[int, np.ndarray]:
    components = sorted(connected_components(adjacency), key=len, reverse=True)
    if len(components) < 2:
        return {relay: positions[relay].copy() for relay in relays}
    centroids = [positions[component].mean(axis=0) for component in components]
    targets: dict[int, np.ndarray] = {}
    for relay in relays:
        own = next((index for index, component in enumerate(components) if relay in component), 0)
        other = min(
            (index for index in range(len(components)) if index != own),
            key=lambda index: float(np.linalg.norm(centroids[index] - centroids[own])),
        )
        targets[relay] = 0.5 * (centroids[own] + centroids[other])
    return targets


def _bounded_velocity_update(
    position: np.ndarray,
    current_velocity: np.ndarray,
    target: np.ndarray,
    config: RelayActionConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    direction = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-12:
        desired = np.zeros(2, dtype=float)
    else:
        desired_speed = min(float(config.max_speed_mps), distance / max(float(config.dt_s), 1e-12))
        desired = direction / distance * desired_speed
    delta_v = desired - current_velocity
    max_delta = float(config.max_acceleration_mps2) * float(config.dt_s)
    delta_norm = float(np.linalg.norm(delta_v))
    if delta_norm > max_delta > 0.0:
        delta_v *= max_delta / delta_norm
    velocity = current_velocity + delta_v
    speed = float(np.linalg.norm(velocity))
    if speed > float(config.max_speed_mps) > 0.0:
        velocity *= float(config.max_speed_mps) / speed
        speed = float(config.max_speed_mps)
    acceleration = float(np.linalg.norm(velocity - current_velocity)) / max(float(config.dt_s), 1e-12)
    displacement = velocity * float(config.dt_s)
    return velocity, displacement, speed, acceleration


def controller_step(
    snapshot: Snapshot,
    state: ControllerState,
    *,
    policy: str,
    risk_score: float,
    threshold: float,
    config: RelayActionConfig,
    sim_config: dict,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply one causal controller step using only current/past state and score."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if state.offsets.shape != snapshot.positions.shape:
        raise ValueError("controller state and snapshot node dimensions differ")

    base_adjacency = (snapshot.adjacency > 0).astype(np.float32)
    if policy == "No intervention":
        return base_adjacency, {
            "triggered": False,
            "action_started": False,
            "action_active": False,
            "relays": [],
            "travel_m": 0.0,
            "energy_proxy": 0.0,
            "max_speed_mps": 0.0,
            "max_acceleration_mps2": 0.0,
            "information_time_index": int(snapshot.time_index),
        }

    triggered = bool(float(risk_score) >= float(threshold))
    state.consecutive_trigger_steps = state.consecutive_trigger_steps + 1 if triggered else 0
    eligible_trigger = triggered
    if policy == "Reactive intervention":
        eligible_trigger = state.consecutive_trigger_steps >= int(config.min_trigger_separation_steps)

    action_started = False
    if eligible_trigger and state.pending_steps < 0 and state.active_steps <= 0 and state.cooldown_steps <= 0:
        state.pending_steps = int(config.reaction_delay_steps)
    if state.pending_steps >= 0:
        if state.pending_steps == 0:
            components = sorted(connected_components(base_adjacency), key=len, reverse=True)
            state.relays = _select_relays(base_adjacency, components, int(config.max_relays))
            state.active_steps = int(config.action_duration_steps)
            state.pending_steps = -1
            state.intervention_count += 1
            action_started = True
        else:
            state.pending_steps -= 1

    positions = snapshot.positions.astype(float) + state.offsets
    active = state.active_steps > 0 and bool(state.relays)
    targets = _relay_targets(positions, base_adjacency, state.relays) if active else {
        relay: positions[relay].copy() for relay in state.relays
    }
    step_travel = 0.0
    step_energy = 0.0
    max_speed = 0.0
    max_acceleration = 0.0
    moving_relays: list[int] = []
    for relay in list(state.relays):
        velocity, displacement, speed, acceleration = _bounded_velocity_update(
            positions[relay],
            state.commanded_velocities[relay],
            targets[relay],
            config,
        )
        state.commanded_velocities[relay] = velocity
        state.offsets[relay] += displacement
        step_travel += float(np.linalg.norm(displacement))
        step_energy += float(config.dt_s) * (
            float(config.energy_speed_weight) * speed**2
            + float(config.energy_acceleration_weight) * acceleration**2
        )
        max_speed = max(max_speed, speed)
        max_acceleration = max(max_acceleration, acceleration)
        if speed > 1e-9:
            moving_relays.append(relay)

    if state.active_steps > 0:
        state.active_steps -= 1
        if state.active_steps == 0:
            state.cooldown_steps = int(config.cooldown_steps)
    elif state.cooldown_steps > 0:
        state.cooldown_steps -= 1
    if not active and not moving_relays and state.active_steps <= 0:
        state.relays = []

    state.cumulative_travel_m += step_travel
    state.cumulative_energy_proxy += step_energy
    effective_positions = snapshot.positions.astype(float) + state.offsets
    controlled = _controlled_physical_adjacency(
        snapshot,
        effective_positions,
        float(config.radius_boost),
        state.relays if active else [],
        sim_config,
        float(config.relay_link_budget_boost_db),
    )
    return controlled, {
        "triggered": triggered,
        "action_started": action_started,
        "action_active": active,
        "relays": list(state.relays),
        "travel_m": step_travel,
        "energy_proxy": step_energy,
        "max_speed_mps": max_speed,
        "max_acceleration_mps2": max_acceleration,
        "information_time_index": int(snapshot.time_index),
    }


def validate_paired_initial_conditions(records: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, str, int, int], set[str]] = {}
    for record in records:
        key = (
            int(record["seed"]),
            str(record["run_id"]),
            int(record["time_index"]),
            int(record["traffic_load"]),
        )
        grouped.setdefault(key, set()).add(str(record["initial_condition_hash"]))
    bad = [key for key, values in grouped.items() if len(values) != 1]
    if bad:
        raise RuntimeError(f"unpaired initial conditions: {bad[:5]}")
