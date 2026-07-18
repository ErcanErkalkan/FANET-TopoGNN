from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fanet.closed_loop import (
    RelayActionConfig,
    controller_step,
    initial_state,
    traffic_realization_seed,
    validate_paired_initial_conditions,
)
from fanet.dataset import Snapshot
from fanet.packet_sim import PacketSimulationConfig, generate_traffic_pairs, simulate_packet_tick


def _snapshot(time_index: int = 0, frag_at_horizon: int = 1) -> Snapshot:
    adjacency = np.asarray(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.float32,
    )
    return Snapshot(
        run_id="closed_loop_seed7",
        split_group_id="closed_loop_seed7",
        time_index=time_index,
        mobility="unit",
        n_nodes=4,
        positions=np.asarray([[0, 0], [20, 0], [200, 0], [220, 0]], dtype=np.float32),
        velocities=np.zeros((4, 2), dtype=np.float32),
        node_features=np.zeros((4, 4), dtype=np.float32),
        adjacency=adjacency,
        adjacency_fixed=adjacency,
        adjacency_adaptive=adjacency,
        pi=np.zeros(4, dtype=np.float32),
        stats=np.zeros(17, dtype=np.float32),
        beta_current=2.0,
        beta_target=2.0,
        beta_fixed=2.0,
        beta_adaptive=2.0,
        radius=80.0,
        radius_fixed=80.0,
        radius_adaptive=80.0,
        edge_count_fixed=2,
        edge_count_adaptive=2,
        link_model="unit",
        graph_policy="fixed",
        radio_scenario="unit",
        is_connected=0,
        future_time_index=time_index + 6,
        frag_at_horizon=frag_at_horizon,
    )


def test_controller_does_not_read_future_ground_truth() -> None:
    config = RelayActionConfig(reaction_delay_steps=0)
    first_state = initial_state(4)
    second_state = initial_state(4)
    first, first_log = controller_step(
        _snapshot(frag_at_horizon=0), first_state,
        policy="Original Kinetic-TopoGuard intervention", risk_score=0.9,
        threshold=0.5, config=config, sim_config=None,
    )
    second, second_log = controller_step(
        _snapshot(frag_at_horizon=1), second_state,
        policy="Original Kinetic-TopoGuard intervention", risk_score=0.9,
        threshold=0.5, config=config, sim_config=None,
    )
    assert np.array_equal(first, second)
    assert first_log == second_log
    assert first_log["information_time_index"] == 0


def test_paired_initial_conditions_require_identical_hashes() -> None:
    base = {
        "seed": 7, "run_id": "run", "time_index": 0, "traffic_load": 8,
        "initial_condition_hash": "same",
    }
    validate_paired_initial_conditions([{**base, "policy": "a"}, {**base, "policy": "b"}])
    with pytest.raises(RuntimeError, match="unpaired initial conditions"):
        validate_paired_initial_conditions([
            {**base, "policy": "a"},
            {**base, "policy": "b", "initial_condition_hash": "different"},
        ])


def test_relay_speed_and_acceleration_are_bounded() -> None:
    config = RelayActionConfig(
        dt_s=0.2, max_speed_mps=4.0, max_acceleration_mps2=1.5,
        reaction_delay_steps=0, action_duration_steps=8,
    )
    state = initial_state(4)
    for time_index in range(8):
        _, log = controller_step(
            _snapshot(time_index), state,
            policy="Persistence-triggered intervention", risk_score=1.0,
            threshold=0.5, config=config, sim_config=None,
        )
        assert log["max_speed_mps"] <= config.max_speed_mps + 1e-10
        assert log["max_acceleration_mps2"] <= config.max_acceleration_mps2 + 1e-10


def test_no_intervention_emits_no_action() -> None:
    state = initial_state(4)
    adjacency, log = controller_step(
        _snapshot(), state,
        policy="No intervention", risk_score=1.0, threshold=0.0,
        config=RelayActionConfig(), sim_config=None,
    )
    assert np.array_equal(adjacency, _snapshot().adjacency)
    assert not log["action_started"]
    assert not log["action_active"]
    assert state.intervention_count == 0
    assert state.cumulative_travel_m == 0.0


def test_packet_arrivals_and_smoke_result_are_deterministic() -> None:
    seed = traffic_realization_seed(7, "run", 3, 8)
    pairs_a = generate_traffic_pairs(4, 8, np.random.default_rng(seed))
    pairs_b = generate_traffic_pairs(4, 8, np.random.default_rng(seed))
    assert pairs_a == pairs_b
    adjacency = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    config = PacketSimulationConfig(packets_per_tick=8)
    first = simulate_packet_tick(adjacency, np.random.default_rng(99), config, traffic_pairs=pairs_a)
    second = simulate_packet_tick(adjacency, np.random.default_rng(99), config, traffic_pairs=pairs_b)
    assert first == second
    assert len(first["queue_delay_samples_ms"]) == int(first["delivered"])


def test_drop_accounting_includes_all_mutually_exclusive_causes() -> None:
    disconnected = np.zeros((4, 4), dtype=np.float32)
    result = simulate_packet_tick(
        disconnected,
        np.random.default_rng(7),
        PacketSimulationConfig(packets_per_tick=8),
        traffic_pairs=[(0, 1)] * 8,
    )
    accounted = (
        result["delivered"] + result["no_route_drops"] + result["queue_drops"]
        + result["deadline_drops"] + result["link_failure_drops"]
        + result["intervention_transition_drops"]
    )
    assert accounted == result["generated"]
    assert result["no_route_drops"] == 8.0
    assert result["delivered"] == 0.0


def test_packet_deadline_marks_unfinished_routable_packet() -> None:
    connected = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    result = simulate_packet_tick(
        connected,
        np.random.default_rng(11),
        PacketSimulationConfig(
            packets_per_tick=1,
            packet_bytes=1200,
            bitrate_mbps=0.001,
            packet_deadline_s=0.001,
            max_backoff_slots=0,
        ),
        traffic_pairs=[(0, 1)],
    )
    assert result["routable_packets"] == 1.0
    assert result["delivered"] == 0.0
    assert result["deadline_drops"] == 1.0
