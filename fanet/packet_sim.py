from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simpy

from .graph_utils import shortest_path_matrix


@dataclass(frozen=True)
class PacketSimulationConfig:
    tick_duration_s: float = 0.1
    packets_per_tick: int = 16
    packet_bytes: int = 1200
    bitrate_mbps: float = 6.0
    queue_limit: int = 64
    difs_us: float = 34.0
    slot_us: float = 9.0
    max_backoff_slots: int = 15
    propagation_us_per_hop: float = 5.0


def _shortest_path(adjacency: np.ndarray, source: int, target: int) -> list[int] | None:
    if source == target:
        return [source]
    queue = [source]
    parent = {source: -1}
    for node in queue:
        for neighbour in np.flatnonzero(adjacency[node] > 0).astype(int):
            if int(neighbour) in parent:
                continue
            parent[int(neighbour)] = node
            if int(neighbour) == target:
                path = [target]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(int(neighbour))
    return None


def simulate_packet_tick(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    config: PacketSimulationConfig,
) -> dict[str, float]:
    """Simulate routed packets over one topology snapshot with SimPy.

    The channel is a conservative single collision domain. CSMA-style random
    backoff and a finite MAC queue determine contention delay and deadline
    drops; the supplied physical graph determines route availability.
    """
    n_nodes = int(adjacency.shape[0])
    env = simpy.Environment()
    channel = simpy.Resource(env, capacity=1)
    waiting_transmissions = 0
    results = {
        "generated": 0,
        "delivered": 0,
        "no_route_drops": 0,
        "queue_drops": 0,
        "deadline_drops": 0,
        "delay_s": [],
        "hops": [],
    }
    tx_s = float(config.packet_bytes * 8) / max(float(config.bitrate_mbps) * 1e6, 1.0)
    difs_s = float(config.difs_us) * 1e-6
    slot_s = float(config.slot_us) * 1e-6
    propagation_s = float(config.propagation_us_per_hop) * 1e-6

    def transmit_packet(path: list[int]) -> object:
        nonlocal waiting_transmissions
        start = env.now
        for _left, _right in zip(path[:-1], path[1:]):
            backoff = int(rng.integers(0, int(config.max_backoff_slots) + 1))
            yield env.timeout(difs_s + backoff * slot_s)
            if waiting_transmissions >= int(config.queue_limit):
                results["queue_drops"] += 1
                return
            waiting_transmissions += 1
            with channel.request() as request:
                yield request
                waiting_transmissions -= 1
                yield env.timeout(tx_s + propagation_s)
            if env.now > float(config.tick_duration_s):
                results["deadline_drops"] += 1
                return
        results["delivered"] += 1
        results["delay_s"].append(float(env.now - start))
        results["hops"].append(float(max(len(path) - 1, 0)))

    traffic_pairs = []
    for _ in range(int(config.packets_per_tick)):
        source = int(rng.integers(0, n_nodes))
        target = int(rng.integers(0, n_nodes - 1))
        if target >= source:
            target += 1
        traffic_pairs.append((source, target))

    for source, target in traffic_pairs:
        results["generated"] += 1
        path = _shortest_path(adjacency, source, target)
        if path is None:
            results["no_route_drops"] += 1
            continue
        env.process(transmit_packet(path))
    env.run(until=float(config.tick_duration_s) + 1e-12)
    unfinished = (
        results["generated"]
        - results["delivered"]
        - results["no_route_drops"]
        - results["queue_drops"]
        - results["deadline_drops"]
    )
    results["deadline_drops"] += max(int(unfinished), 0)
    delay_values = np.asarray(results.pop("delay_s"), dtype=float)
    hop_values = np.asarray(results.pop("hops"), dtype=float)
    generated = max(int(results["generated"]), 1)
    return {
        **{key: float(value) for key, value in results.items()},
        "pdr": float(results["delivered"] / generated),
        "mean_delay_ms": float(delay_values.mean() * 1000.0) if delay_values.size else float("nan"),
        "p95_delay_ms": float(np.quantile(delay_values, 0.95) * 1000.0) if delay_values.size else float("nan"),
        "delay_samples_ms": (delay_values * 1000.0).tolist(),
        "mean_hops": float(hop_values.mean()) if hop_values.size else float("nan"),
        "connected_pair_ratio": float(np.isfinite(shortest_path_matrix(adjacency)).sum() - n_nodes)
        / max(n_nodes * (n_nodes - 1), 1),
    }
