from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import clip_positions, heading_to_vector, random_waypoints


@dataclass
class MobilityState:
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray | None = None
    heading: np.ndarray | None = None
    speed: np.ndarray | None = None
    mission_plans: list[list[dict[str, object]]] | None = None
    mission_index: np.ndarray | None = None
    mission_age: np.ndarray | None = None


def init_rwp(rng: np.random.Generator, n_nodes: int, area_size: float, speed_range: tuple[float, float]) -> MobilityState:
    pos = rng.uniform(0.0, area_size, size=(n_nodes, 2))
    target = random_waypoints(rng, n_nodes, area_size)
    speed = rng.uniform(speed_range[0], speed_range[1], size=n_nodes)
    direction = target - pos
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    vel = direction / norms * speed[:, None]
    return MobilityState(pos, vel, target=target, speed=speed)


def step_rwp(state: MobilityState, rng: np.random.Generator, dt: float, area_size: float, speed_range: tuple[float, float]) -> MobilityState:
    pos = state.position.copy()
    target = state.target.copy()
    speed = state.speed.copy()
    for i in range(len(pos)):
        delta = target[i] - pos[i]
        distance = np.linalg.norm(delta)
        if distance < speed[i] * dt:
            target[i] = rng.uniform(0.0, area_size, size=2)
            speed[i] = rng.uniform(speed_range[0], speed_range[1])
            delta = target[i] - pos[i]
            distance = np.linalg.norm(delta)
        pos[i] += delta / max(distance, 1e-6) * speed[i] * dt
    vel = target - pos
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    vel = vel / norms * speed[:, None]
    return MobilityState(clip_positions(pos, area_size), vel, target=target, speed=speed)


def init_gm(rng: np.random.Generator, n_nodes: int, area_size: float, mu_speed: float) -> MobilityState:
    pos = rng.uniform(0.0, area_size, size=(n_nodes, 2))
    heading = rng.uniform(0.0, 2.0 * np.pi, size=n_nodes)
    speed = np.full(n_nodes, mu_speed, dtype=float)
    vel = heading_to_vector(heading, speed)
    return MobilityState(pos, vel, heading=heading, speed=speed)


def step_gm(state: MobilityState, rng: np.random.Generator, dt: float, area_size: float, alpha: float, mu_speed: float, sigma: float) -> MobilityState:
    heading = alpha * state.heading + (1.0 - alpha) * rng.uniform(0.0, 2.0 * np.pi, size=len(state.heading))
    speed = alpha * state.speed + (1.0 - alpha) * mu_speed + rng.normal(0.0, sigma * 0.1, size=len(state.speed))
    speed = np.clip(speed, 5.0, 35.0)
    vel = heading_to_vector(heading, speed)
    pos = clip_positions(state.position + vel * dt, area_size)
    bounce = (pos <= 0.0) | (pos >= area_size)
    vel[bounce] *= -1.0
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    speed = np.linalg.norm(vel, axis=1)
    return MobilityState(clip_positions(pos, area_size), vel, heading=heading, speed=speed)


def _mission_region_bounds(region: int, regions: int, area_size: float) -> tuple[float, float]:
    band = area_size / max(regions, 1)
    left = region * band
    right = min((region + 1) * band, area_size)
    return left, right


def _mission_waypoint(point: np.ndarray, kind: str, speed_low: float, speed_high: float, max_age: int) -> dict[str, object]:
    return {
        "point": point.astype(float),
        "kind": kind,
        "speed_low": float(speed_low),
        "speed_high": float(speed_high),
        "max_age": int(max_age),
    }


def _build_mission_plan(rng: np.random.Generator, region: int, area_size: float, regions: int) -> list[dict[str, object]]:
    left, right = _mission_region_bounds(region, regions, area_size)
    width = max(right - left, 1.0)
    margin = min(45.0, width * 0.18, area_size * 0.08)
    x_low = left + margin
    x_high = right - margin
    y_low = margin
    y_high = area_size - margin
    scan_x = np.linspace(x_low, x_high, num=4)
    plan: list[dict[str, object]] = []
    for idx, x_coord in enumerate(scan_x):
        start_y, end_y = (y_low, y_high) if idx % 2 == 0 else (y_high, y_low)
        plan.append(_mission_waypoint(np.array([x_coord, start_y]), "transit", 12.0, 24.0, 80))
        plan.append(_mission_waypoint(np.array([x_coord, end_y]), "scan", 9.0, 18.0, 130))

    centre = np.array(
        [
            rng.uniform(x_low, x_high),
            rng.uniform(area_size * 0.25, area_size * 0.75),
        ],
        dtype=float,
    )
    loiter_radius = min(width * 0.18, area_size * 0.12)
    plan.append(_mission_waypoint(centre, "transit", 12.0, 24.0, 90))
    for angle in np.linspace(0.0, 2.0 * np.pi, num=8, endpoint=False):
        point = centre + loiter_radius * np.array([np.cos(angle), np.sin(angle)])
        point[0] = np.clip(point[0], left + margin, right - margin)
        point[1] = np.clip(point[1], margin, area_size - margin)
        plan.append(_mission_waypoint(point, "loiter", 7.0, 14.0, 70))
    return plan


def init_mission(rng: np.random.Generator, n_nodes: int, area_size: float, regions: int) -> MobilityState:
    region_ids = np.asarray([i % max(regions, 1) for i in range(n_nodes)], dtype=int)
    plans = [_build_mission_plan(rng, int(region), area_size, regions) for region in region_ids]
    pos = np.zeros((n_nodes, 2), dtype=float)
    target = np.zeros_like(pos)
    for i, region in enumerate(region_ids):
        left, right = _mission_region_bounds(int(region), regions, area_size)
        pos[i] = np.array([rng.uniform(left, right), rng.uniform(0.0, area_size)], dtype=float)
        target[i] = np.asarray(plans[i][0]["point"], dtype=float)
    speed = rng.uniform(8.0, 22.0, size=n_nodes)
    vel = target - pos
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    vel = vel / norms * speed[:, None]
    return MobilityState(
        pos,
        vel,
        target=target,
        speed=speed,
        mission_plans=plans,
        mission_index=np.zeros(n_nodes, dtype=int),
        mission_age=np.zeros(n_nodes, dtype=int),
    )


def step_mission(
    state: MobilityState,
    rng: np.random.Generator,
    dt: float,
    area_size: float,
    regions: int,
    tracking_noise_m: float = 2.0,
    heading_noise_rad: float = 0.05,
    speed_jitter: float = 0.08,
) -> MobilityState:
    pos = state.position.copy()
    target = state.target.copy()
    speed = state.speed.copy()
    plans = state.mission_plans
    if plans is None or state.mission_index is None or state.mission_age is None:
        return init_mission(rng, len(pos), area_size, regions)
    mission_index = state.mission_index.copy()
    mission_age = state.mission_age.copy()

    vel = np.zeros_like(pos)
    for i in range(len(pos)):
        waypoint = plans[i][int(mission_index[i])]
        nominal_target = np.asarray(waypoint["point"], dtype=float)
        noisy_target = nominal_target + rng.normal(0.0, tracking_noise_m, size=2)
        noisy_target = clip_positions(noisy_target.reshape(1, 2), area_size).reshape(2)
        delta = noisy_target - pos[i]
        distance = float(np.linalg.norm(delta))
        max_age = int(waypoint["max_age"])
        arrival_radius = 18.0 if waypoint["kind"] != "loiter" else 12.0
        if distance < arrival_radius or mission_age[i] >= max_age:
            mission_index[i] = (mission_index[i] + 1) % len(plans[i])
            mission_age[i] = 0
            waypoint = plans[i][int(mission_index[i])]
            nominal_target = np.asarray(waypoint["point"], dtype=float)
            noisy_target = clip_positions((nominal_target + rng.normal(0.0, tracking_noise_m, size=2)).reshape(1, 2), area_size).reshape(2)
            delta = noisy_target - pos[i]
            distance = float(np.linalg.norm(delta))

        heading = np.arctan2(delta[1], delta[0]) + rng.normal(0.0, heading_noise_rad)
        desired_speed = rng.uniform(float(waypoint["speed_low"]), float(waypoint["speed_high"]))
        speed[i] = np.clip(desired_speed * (1.0 + rng.normal(0.0, speed_jitter)), 5.0, 30.0)
        vel[i] = heading_to_vector(np.asarray([heading]), np.asarray([speed[i]])).reshape(2)
        pos[i] = pos[i] + vel[i] * dt
        target[i] = nominal_target
        mission_age[i] += 1

    return MobilityState(
        clip_positions(pos, area_size),
        vel,
        target=target,
        speed=speed,
        mission_plans=plans,
        mission_index=mission_index,
        mission_age=mission_age,
    )
