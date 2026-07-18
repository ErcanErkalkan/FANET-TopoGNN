from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simpy

from .graph_utils import shortest_path_matrix


@dataclass(frozen=True)
class PacketSimulationConfig:
    tick_duration_s: float = 0.1
    packet_deadline_s: float = 0.1
    packets_per_tick: int = 16
    packet_bytes: int = 1200
    bitrate_mbps: float = 6.0
    queue_limit: int = 64
    difs_us: float = 34.0
    slot_us: float = 9.0
    max_backoff_slots: int = 15
    propagation_us_per_hop: float = 5.0
    link_failure_probability_per_attempt: float = 0.0
    intervention_transition_failure_probability_per_attempt: float = 0.0
    max_retransmissions: int = 0


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
    traffic_pairs: list[tuple[int, int]] | None = None,
    transition_nodes: list[int] | None = None,
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
        "link_failure_drops": 0,
        "intervention_transition_drops": 0,
        "routable_packets": 0,
        "retransmission_attempts": 0,
        "delay_s": [],
        "queue_delay_s": [],
        "queue_occupancy": [],
        "hops": [],
    }
    tx_s = float(config.packet_bytes * 8) / max(float(config.bitrate_mbps) * 1e6, 1.0)
    difs_s = float(config.difs_us) * 1e-6
    slot_s = float(config.slot_us) * 1e-6
    propagation_s = float(config.propagation_us_per_hop) * 1e-6
    deadline_s = float(config.packet_deadline_s)
    transition_node_set = {int(node) for node in (transition_nodes or [])}
    if not 0.0 <= float(config.link_failure_probability_per_attempt) <= 1.0:
        raise ValueError("link_failure_probability_per_attempt must be in [0, 1]")
    if not 0.0 <= float(config.intervention_transition_failure_probability_per_attempt) <= 1.0:
        raise ValueError("intervention transition failure probability must be in [0, 1]")
    if int(config.max_retransmissions) < 0:
        raise ValueError("max_retransmissions must be non-negative")

    def transmit_packet(path: list[int]) -> object:
        nonlocal waiting_transmissions
        start = env.now
        queue_delay = 0.0
        for left, right in zip(path[:-1], path[1:]):
            for attempt in range(int(config.max_retransmissions) + 1):
                if attempt:
                    results["retransmission_attempts"] += 1
                backoff = int(rng.integers(0, int(config.max_backoff_slots) + 1))
                contention_wait = difs_s + backoff * slot_s
                yield env.timeout(contention_wait)
                queue_delay += contention_wait
                if waiting_transmissions >= int(config.queue_limit):
                    results["queue_drops"] += 1
                    return
                waiting_transmissions += 1
                results["queue_occupancy"].append(float(waiting_transmissions))
                requested_at = env.now
                transition_failure = False
                link_failure = False
                with channel.request() as request:
                    yield request
                    queue_delay += float(env.now - requested_at)
                    waiting_transmissions -= 1
                    results["queue_occupancy"].append(float(waiting_transmissions))
                    if env.now >= deadline_s:
                        results["deadline_drops"] += 1
                        return
                    is_transition_hop = int(left) in transition_node_set or int(right) in transition_node_set
                    transition_failure = bool(
                        is_transition_hop
                        and rng.random() < float(config.intervention_transition_failure_probability_per_attempt)
                    )
                    if not transition_failure:
                        link_failure = bool(
                            rng.random() < float(config.link_failure_probability_per_attempt)
                        )
                    if not transition_failure and not link_failure:
                        yield env.timeout(tx_s + propagation_s)
                if transition_failure or link_failure:
                    if attempt < int(config.max_retransmissions):
                        continue
                    key = "intervention_transition_drops" if transition_failure else "link_failure_drops"
                    results[key] += 1
                    return
                break
            if env.now > deadline_s:
                results["deadline_drops"] += 1
                return
        results["delivered"] += 1
        results["delay_s"].append(float(env.now - start))
        results["queue_delay_s"].append(float(queue_delay))
        results["hops"].append(float(max(len(path) - 1, 0)))

    if traffic_pairs is None:
        traffic_pairs = generate_traffic_pairs(n_nodes, int(config.packets_per_tick), rng)
    else:
        traffic_pairs = [(int(source), int(target)) for source, target in traffic_pairs]
        if any(source == target or source < 0 or target < 0 or source >= n_nodes or target >= n_nodes for source, target in traffic_pairs):
            raise ValueError("traffic_pairs must contain distinct valid node indices")

    for source, target in traffic_pairs:
        results["generated"] += 1
        path = _shortest_path(adjacency, source, target)
        if path is None:
            results["no_route_drops"] += 1
            continue
        results["routable_packets"] += 1
        env.process(transmit_packet(path))
    env.run(until=deadline_s + 1e-12)
    unfinished = (
        results["generated"]
        - results["delivered"]
        - results["no_route_drops"]
        - results["queue_drops"]
        - results["deadline_drops"]
        - results["link_failure_drops"]
        - results["intervention_transition_drops"]
    )
    results["deadline_drops"] += max(int(unfinished), 0)
    delay_values = np.asarray(results.pop("delay_s"), dtype=float)
    queue_delay_values = np.asarray(results.pop("queue_delay_s"), dtype=float)
    queue_occupancy_values = np.asarray(results.pop("queue_occupancy"), dtype=float)
    hop_values = np.asarray(results.pop("hops"), dtype=float)
    generated = max(int(results["generated"]), 1)
    return {
        **{key: float(value) for key, value in results.items()},
        "pdr": float(results["delivered"] / generated),
        "mean_delay_ms": float(delay_values.mean() * 1000.0) if delay_values.size else float("nan"),
        "p95_delay_ms": float(np.quantile(delay_values, 0.95) * 1000.0) if delay_values.size else float("nan"),
        "p50_delay_ms": float(np.quantile(delay_values, 0.50) * 1000.0) if delay_values.size else float("nan"),
        "p99_delay_ms": float(np.quantile(delay_values, 0.99) * 1000.0) if delay_values.size else float("nan"),
        "delay_samples_ms": (delay_values * 1000.0).tolist(),
        "queue_delay_samples_ms": (queue_delay_values * 1000.0).tolist(),
        "mean_queue_delay_ms": float(queue_delay_values.mean() * 1000.0) if queue_delay_values.size else float("nan"),
        "mean_queue_occupancy": float(queue_occupancy_values.mean()) if queue_occupancy_values.size else 0.0,
        "max_queue_occupancy": float(queue_occupancy_values.max()) if queue_occupancy_values.size else 0.0,
        "queue_occupancy_observations": float(queue_occupancy_values.size),
        "mean_hops": float(hop_values.mean()) if hop_values.size else float("nan"),
        "connected_pair_ratio": float(np.isfinite(shortest_path_matrix(adjacency)).sum() - n_nodes)
        / max(n_nodes * (n_nodes - 1), 1),
    }


def generate_traffic_pairs(
    n_nodes: int,
    packets_per_tick: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate a packet-arrival realization independently of network policy."""
    if n_nodes < 2:
        raise ValueError("at least two nodes are required")
    pairs: list[tuple[int, int]] = []
    for _ in range(int(packets_per_tick)):
        source = int(rng.integers(0, n_nodes))
        target = int(rng.integers(0, n_nodes - 1))
        if target >= source:
            target += 1
        pairs.append((source, target))
    return pairs
