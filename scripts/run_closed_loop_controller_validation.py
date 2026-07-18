from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.closed_loop import (
    POLICIES,
    RelayActionConfig,
    controller_step,
    initial_state,
    traffic_realization_seed,
    validate_paired_initial_conditions,
)
from fanet.config import load_config
from fanet.dataset import Snapshot, build_dataset, train_val_test_split
from fanet.evaluation import predict_generic
from fanet.graph_utils import betti_zero
from fanet.packet_sim import PacketSimulationConfig, generate_traffic_pairs, simulate_packet_tick
from fanet.provenance import build_file_manifest, relative_repo_path
from fanet.source_gated import (
    SourceGatedKineticTopoGuard,
    fit_current_state_extratrees,
)
from fanet.training import TrainResult, fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs" / "closed_loop_controller.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "closed_loop_controller_packet_v2"
HIGHER_IS_BETTER = {
    "Packet_Delivery_Ratio": True,
    "Packet_Deadline_Success": True,
    "Latency_P50_ms": False,
    "Latency_P95_ms": False,
    "Latency_P99_ms": False,
    "Mean_Queue_Occupancy": False,
    "No_Route_Drop_Rate": False,
    "Queue_Overflow_Drop_Rate": False,
    "Deadline_Miss_Drop_Rate": False,
    "Link_Failure_Drop_Rate": False,
    "Intervention_Transition_Drop_Rate": False,
    "Connected_Time_Ratio": True,
    "Outage_Duration_s": False,
    "Fragmentation_Event_Count": False,
    "Mean_Recovery_Time_s": False,
    "Queue_Delay_ms": False,
    "End_to_End_Delay_ms": False,
}


def _split(config, snapshots: list[Snapshot]):
    stratify = list(config.sim.get("split_stratify_by", ["mobility"]))
    for field in ("graph_policy", "radio_scenario"):
        if len({getattr(snapshot, field) for snapshot in snapshots}) > 1 and field not in stratify:
            stratify.append(field)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(config.sim["split_seed"]),
        stratify_by=tuple(stratify),
    )
    run_sets = [{snapshot.run_id for snapshot in split} for split in (train, validation, test)]
    overlap = {
        "train_validation": sorted(run_sets[0] & run_sets[1]),
        "train_test": sorted(run_sets[0] & run_sets[2]),
        "validation_test": sorted(run_sets[1] & run_sets[2]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"run_id leakage: {overlap}")
    return train, validation, test, stratify, overlap


def _score_map(result: TrainResult, snapshots: list[Snapshot]) -> tuple[dict[tuple[str, int], float], float]:
    _, scores, _, aligned, threshold = predict_generic(result, snapshots)
    return {
        (snapshot.run_id, int(snapshot.time_index)): float(score)
        for snapshot, score in zip(aligned, scores)
    }, float(threshold)


def _source_model(seed: int, train, validation, horizon: int, dt: float, parameters: dict):
    artifact = ROOT / "outputs" / "source_gated_development" / "per_seed" / f"seed_{seed}" / "source_gated_model.pkl"
    if artifact.is_file():
        model = SourceGatedKineticTopoGuard.load(artifact)
        if int(model.seed) != int(seed) or int(model.horizon_steps) != int(horizon):
            raise RuntimeError(f"frozen source-gated artifact does not match seed/horizon: {artifact}")
        return TrainResult(model=model, model_name="Source-Gated Kinetic-TopoGuard", inference_ms=0.0), artifact
    from fanet.source_gated import fit_source_gated_kinetic_topoguard

    return (
        fit_source_gated_kinetic_topoguard(
            train, validation, horizon, dt, seed, parameters=parameters
        ),
        None,
    )


def _initial_hash(snapshot: Snapshot, traffic_pairs: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    digest.update(snapshot.run_id.encode("utf-8"))
    digest.update(np.asarray([snapshot.time_index], dtype=np.int64).tobytes())
    digest.update(np.asarray(snapshot.positions, dtype=np.float32).tobytes())
    digest.update(np.asarray(snapshot.velocities, dtype=np.float32).tobytes())
    digest.update(np.asarray(snapshot.adjacency, dtype=np.float32).tobytes())
    digest.update(np.asarray(traffic_pairs, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def _event_postprocess(logs: list[dict], horizon_steps: int) -> tuple[int, int]:
    logs.sort(key=lambda row: int(row["time_index"]))
    event_indices = [
        index for index in range(1, len(logs))
        if not bool(logs[index - 1]["base_fragmented"]) and bool(logs[index]["base_fragmented"])
    ]
    starts = [index for index, row in enumerate(logs) if bool(row["action_started"])]
    unnecessary = 0
    for start in starts:
        if not any(start < event <= start + int(horizon_steps) for event in event_indices):
            unnecessary += 1
            logs[start]["unnecessary_intervention"] = True
    missed = 0
    for event in event_indices:
        if not any(
            bool(logs[index]["action_active"]) or bool(logs[index]["action_started"])
            for index in range(max(0, event - int(horizon_steps)), event + 1)
        ):
            missed += 1
            logs[event]["missed_intervention"] = True
    return unnecessary, missed


def _run_policy_load(
    seed: int,
    policy: str,
    load: int,
    runs: dict[str, list[Snapshot]],
    score_map: dict[tuple[str, int], float],
    threshold: float,
    action_config: RelayActionConfig,
    sim_config: dict,
    stride: int,
    horizon_steps: int,
    packet_parameters: dict,
) -> tuple[list[dict], list[dict]]:
    action_rows: list[dict] = []
    density_accumulators: dict[int, dict[str, object]] = {}
    for run_id, sequence in sorted(runs.items()):
        sampled = sorted(sequence, key=lambda snapshot: snapshot.time_index)[:: max(int(stride), 1)]
        if not sampled:
            continue
        density = int(sampled[0].n_nodes)
        acc = density_accumulators.setdefault(
            density,
            {
                "generated": 0.0, "delivered": 0.0, "routable_packets": 0.0,
                "deadline_drops": 0.0, "no_route_drops": 0.0, "queue_drops": 0.0,
                "link_failure_drops": 0.0, "intervention_transition_drops": 0.0,
                "queue_delay_sum_ms": 0.0, "end_to_end_delay_sum_ms": 0.0,
                "delivered_delay_count": 0, "ticks": 0, "connected_ticks": 0,
                "fragmentation_events": 0, "outage_ticks": 0, "recoveries": [],
                "interventions": 0, "unnecessary": 0, "missed": 0,
                "travel": 0.0, "energy": 0.0, "active_ticks": 0,
                "delay_samples_ms": [], "queue_occupancy_sum": 0.0,
                "queue_occupancy_observations": 0, "max_queue_occupancy": 0.0,
            },
        )
        state = initial_state(sampled[0].n_nodes)
        run_logs: list[dict] = []
        previous_connected = True
        outage_start: int | None = None
        recovery_times: list[float] = []
        for sample_index, snapshot in enumerate(sampled):
            key = (snapshot.run_id, int(snapshot.time_index))
            if policy == "No intervention":
                score, policy_threshold = 0.0, float("inf")
            elif policy == "Reactive intervention":
                score, policy_threshold = float(snapshot.beta_current > 1.0), 0.5
            elif policy == "Persistence-triggered intervention":
                score, policy_threshold = float(snapshot.beta_current > 1.0), 0.5
            else:
                score, policy_threshold = float(score_map.get(key, 0.0)), float(threshold)
            adjacency, action = controller_step(
                snapshot,
                state,
                policy=policy,
                risk_score=score,
                threshold=policy_threshold,
                config=action_config,
                sim_config=sim_config,
            )
            arrival_seed = traffic_realization_seed(seed, run_id, int(snapshot.time_index), int(load))
            pairs = generate_traffic_pairs(
                snapshot.n_nodes,
                int(load),
                np.random.default_rng(arrival_seed),
            )
            packet_rng = np.random.default_rng((arrival_seed + 1_000_003) % (2**63 - 1))
            packet = simulate_packet_tick(
                adjacency,
                packet_rng,
                PacketSimulationConfig(
                    tick_duration_s=float(action_config.dt_s),
                    packet_deadline_s=float(packet_parameters["packet_deadline_s"]),
                    packets_per_tick=int(load),
                    packet_bytes=int(packet_parameters["packet_bytes"]),
                    bitrate_mbps=float(packet_parameters["bitrate_mbps"]),
                    queue_limit=int(packet_parameters["queue_limit"]),
                    difs_us=float(packet_parameters["difs_us"]),
                    slot_us=float(packet_parameters["slot_us"]),
                    max_backoff_slots=int(packet_parameters["max_backoff_slots"]),
                    propagation_us_per_hop=float(packet_parameters["propagation_us_per_hop"]),
                    link_failure_probability_per_attempt=float(packet_parameters["link_failure_probability_per_attempt"]),
                    intervention_transition_failure_probability_per_attempt=float(packet_parameters["intervention_transition_failure_probability_per_attempt"]),
                    max_retransmissions=int(packet_parameters["max_retransmissions"]),
                ),
                traffic_pairs=pairs,
                transition_nodes=list(action["relays"]) if action["action_active"] else [],
            )
            acc["delay_samples_ms"].extend(float(value) for value in packet["delay_samples_ms"])
            occupancy_count = int(packet["queue_occupancy_observations"])
            acc["queue_occupancy_sum"] = float(acc["queue_occupancy_sum"]) + float(packet["mean_queue_occupancy"]) * occupancy_count
            acc["queue_occupancy_observations"] = int(acc["queue_occupancy_observations"]) + occupancy_count
            acc["max_queue_occupancy"] = max(float(acc["max_queue_occupancy"]), float(packet["max_queue_occupancy"]))
            connected = bool(betti_zero(adjacency) == 1)
            if previous_connected and not connected:
                outage_start = sample_index
            elif not previous_connected and connected and outage_start is not None:
                recovery_times.append((sample_index - outage_start) * float(action_config.dt_s) * int(stride))
                outage_start = None
            previous_connected = connected
            record = {
                "seed": int(seed),
                "policy": policy,
                "run_id": run_id,
                "time_index": int(snapshot.time_index),
                "traffic_load": int(load),
                "topology_density_nodes": int(snapshot.n_nodes),
                "risk_score": float(score),
                "selected_threshold": float(policy_threshold),
                "base_fragmented": bool(snapshot.beta_current > 1.0),
                "controlled_fragmented": not connected,
                "initial_condition_hash": _initial_hash(snapshot, pairs),
                "packet_arrival_seed": int(arrival_seed),
                "generated": float(packet["generated"]),
                "delivered": float(packet["delivered"]),
                "deadline_drops": float(packet["deadline_drops"]),
                "link_failure_drops": float(packet["link_failure_drops"]),
                "intervention_transition_drops": float(packet["intervention_transition_drops"]),
                "routable_packets": float(packet["routable_packets"]),
                "queue_drops": float(packet["queue_drops"]),
                "no_route_drops": float(packet["no_route_drops"]),
                "queue_delay_sum_ms": float(np.nansum(packet["queue_delay_samples_ms"])),
                "end_to_end_delay_sum_ms": float(np.nansum(packet["delay_samples_ms"])),
                "delivered_delay_count": int(len(packet["delay_samples_ms"])),
                "latency_p50_ms": float(packet["p50_delay_ms"]),
                "latency_p95_ms": float(packet["p95_delay_ms"]),
                "latency_p99_ms": float(packet["p99_delay_ms"]),
                "mean_queue_occupancy": float(packet["mean_queue_occupancy"]),
                "max_queue_occupancy": float(packet["max_queue_occupancy"]),
                "triggered": bool(action["triggered"]),
                "action_started": bool(action["action_started"]),
                "action_active": bool(action["action_active"]),
                "relay_ids": ";".join(str(value) for value in action["relays"]),
                "relay_travel_m": float(action["travel_m"]),
                "action_energy_proxy": float(action["energy_proxy"]),
                "commanded_max_speed_mps": float(action["max_speed_mps"]),
                "commanded_max_acceleration_mps2": float(action["max_acceleration_mps2"]),
                "information_time_index": int(action["information_time_index"]),
                "future_label_read_by_policy": False,
                "unnecessary_intervention": False,
                "missed_intervention": False,
            }
            run_logs.append(record)
            action_rows.append(record)
        unnecessary, missed = _event_postprocess(run_logs, horizon_steps)
        for row in run_logs:
            for name in ("generated", "delivered", "routable_packets", "deadline_drops", "no_route_drops", "queue_drops", "link_failure_drops", "intervention_transition_drops", "queue_delay_sum_ms", "end_to_end_delay_sum_ms"):
                acc[name] = float(acc[name]) + float(row[name])
            acc["delivered_delay_count"] = int(acc["delivered_delay_count"]) + int(row["delivered_delay_count"])
            acc["ticks"] = int(acc["ticks"]) + 1
            acc["connected_ticks"] = int(acc["connected_ticks"]) + int(not row["controlled_fragmented"])
            acc["outage_ticks"] = int(acc["outage_ticks"]) + int(row["controlled_fragmented"])
            acc["interventions"] = int(acc["interventions"]) + int(row["action_started"])
            acc["travel"] = float(acc["travel"]) + float(row["relay_travel_m"])
            acc["energy"] = float(acc["energy"]) + float(row["action_energy_proxy"])
            acc["active_ticks"] = int(acc["active_ticks"]) + int(row["action_active"])
        acc["fragmentation_events"] = int(acc["fragmentation_events"]) + sum(
            1 for index in range(1, len(run_logs))
            if not run_logs[index - 1]["controlled_fragmented"] and run_logs[index]["controlled_fragmented"]
        )
        acc["recoveries"].extend(recovery_times)
        acc["unnecessary"] = int(acc["unnecessary"]) + unnecessary
        acc["missed"] = int(acc["missed"]) + missed

    rows = []
    for density, acc in sorted(density_accumulators.items()):
        generated = max(float(acc["generated"]), 1.0)
        routable = max(float(acc["routable_packets"]), 1.0)
        delivered = max(int(acc["delivered_delay_count"]), 1)
        delay_values = np.asarray(acc["delay_samples_ms"], dtype=float)
        simulated_seconds = int(acc["ticks"]) * float(action_config.dt_s) * int(stride)
        rows.append({
            "seed": int(seed), "Policy": policy, "Traffic_Load": int(load),
            "Topology_Density_Nodes": int(density),
            "Packet_Delivery_Ratio": float(acc["delivered"]) / generated,
            "Packet_Deadline_Success": float(acc["delivered"]) / routable,
            "Latency_P50_ms": float(np.quantile(delay_values, 0.50)) if delay_values.size else float("nan"),
            "Latency_P95_ms": float(np.quantile(delay_values, 0.95)) if delay_values.size else float("nan"),
            "Latency_P99_ms": float(np.quantile(delay_values, 0.99)) if delay_values.size else float("nan"),
            "Mean_Queue_Occupancy": float(acc["queue_occupancy_sum"]) / max(int(acc["queue_occupancy_observations"]), 1),
            "Max_Queue_Occupancy": float(acc["max_queue_occupancy"]),
            "No_Route_Drop_Rate": float(acc["no_route_drops"]) / generated,
            "Queue_Overflow_Drop_Rate": float(acc["queue_drops"]) / generated,
            "Deadline_Miss_Drop_Rate": float(acc["deadline_drops"]) / generated,
            "Link_Failure_Drop_Rate": float(acc["link_failure_drops"]) / generated,
            "Intervention_Transition_Drop_Rate": float(acc["intervention_transition_drops"]) / generated,
            "Drop_Accounting_Error": abs(
                float(acc["generated"]) - float(acc["delivered"])
                - float(acc["no_route_drops"]) - float(acc["queue_drops"])
                - float(acc["deadline_drops"]) - float(acc["link_failure_drops"])
                - float(acc["intervention_transition_drops"])
            ),
            "Outage_Duration_s": int(acc["outage_ticks"]) * float(action_config.dt_s) * int(stride),
            "Connected_Time_Ratio": int(acc["connected_ticks"]) / max(int(acc["ticks"]), 1),
            "Fragmentation_Event_Count": int(acc["fragmentation_events"]),
            "Mean_Recovery_Time_s": float(np.mean(acc["recoveries"])) if acc["recoveries"] else 0.0,
            "Intervention_Count": int(acc["interventions"]),
            "Unnecessary_Intervention_Count": int(acc["unnecessary"]),
            "Missed_Intervention_Count": int(acc["missed"]),
            "Relay_Travel_Distance_m": float(acc["travel"]),
            "Action_Energy_Proxy": float(acc["energy"]),
            "Queue_Delay_ms": float(acc["queue_delay_sum_ms"]) / delivered,
            "End_to_End_Delay_ms": float(acc["end_to_end_delay_sum_ms"]) / delivered,
            "Action_Duty_Cycle": int(acc["active_ticks"]) / max(int(acc["ticks"]), 1),
            "Relay_Travel_m_per_Simulated_Minute": float(acc["travel"]) / max(simulated_seconds / 60.0, 1e-12),
            "Simulated_Duration_s": simulated_seconds,
            "Generated_Packets": float(acc["generated"]),
        })
    return rows, action_rows


def run_seed(seed: int, config_path: Path, stride_override: int | None = None):
    started = time.perf_counter()
    config = load_config(config_path)
    closed = config.raw["closed_loop"]
    packet_parameters = config.raw["packet_simulation"]
    snapshots = build_dataset(config.sim, seed=int(seed))
    train, validation, test, stratify, overlap = _split(config, snapshots)
    densities = {int(value) for value in closed["topology_density_nodes"]}
    test = [snapshot for snapshot in test if int(snapshot.n_nodes) in densities]
    horizon, dt = int(config.sim["forecast_horizon_steps"]), float(config.sim["dt"])
    current = fit_current_state_extratrees(train, validation, horizon, dt, int(seed))
    kinetic = fit_kinetic_topoguard(train, validation, horizon, dt, seed=int(seed))
    source, source_artifact = _source_model(
        int(seed), train, validation, horizon, dt, config.training["source_gated"]
    )
    mappings = {}
    thresholds = {}
    for policy, result in (
        ("Current-state ExtraTrees intervention", current),
        ("Original Kinetic-TopoGuard intervention", kinetic),
        ("Source-Gated Kinetic-TopoGuard intervention", source),
    ):
        mappings[policy], thresholds[policy] = _score_map(result, test)
    mappings["No intervention"] = {}
    mappings["Reactive intervention"] = {}
    mappings["Persistence-triggered intervention"] = {}
    thresholds.update({policy: 0.5 for policy in POLICIES[:3]})
    stride = int(stride_override or closed["snapshot_stride"])
    action_config = RelayActionConfig(
        dt_s=dt * stride,
        max_speed_mps=float(closed["max_speed_mps"]),
        max_acceleration_mps2=float(closed["max_acceleration_mps2"]),
        reaction_delay_steps=int(closed["reaction_delay_steps"]),
        action_duration_steps=int(closed["action_duration_steps"]),
        cooldown_steps=int(closed["cooldown_steps"]),
        max_relays=int(closed["max_relays"]),
        radius_boost=float(closed["radius_boost"]),
        relay_link_budget_boost_db=float(closed["relay_link_budget_boost_db"]),
        min_trigger_separation_steps=int(closed["reactive_confirmation_steps"]),
        energy_speed_weight=float(closed["energy_speed_weight"]),
        energy_acceleration_weight=float(closed["energy_acceleration_weight"]),
    )
    runs: dict[str, list[Snapshot]] = {}
    for snapshot in test:
        runs.setdefault(snapshot.run_id, []).append(snapshot)
    metrics, logs = [], []
    for policy in POLICIES:
        for load in closed["traffic_loads"]:
            policy_metrics, policy_logs = _run_policy_load(
                int(seed), policy, int(load), runs, mappings[policy], thresholds[policy],
                action_config, config.sim, stride, horizon, packet_parameters,
            )
            metrics.extend(policy_metrics)
            logs.extend(policy_logs)
    validate_paired_initial_conditions(logs)
    max_speed = max((float(row["commanded_max_speed_mps"]) for row in logs), default=0.0)
    max_acceleration = max((float(row["commanded_max_acceleration_mps2"]) for row in logs), default=0.0)
    if max_speed > action_config.max_speed_mps + 1e-8 or max_acceleration > action_config.max_acceleration_mps2 + 1e-8:
        raise RuntimeError("relay motion constraint violation")
    if any(row["action_started"] or row["action_active"] for row in logs if row["policy"] == "No intervention"):
        raise RuntimeError("no-intervention policy emitted an action")
    metadata = {
        "seed": int(seed), "runtime_seconds": time.perf_counter() - started,
        "split_stratify_by": stratify, "run_id_intersections": overlap,
        "validation_selected_thresholds": thresholds,
        "source_gated_artifact": relative_repo_path(source_artifact, ROOT) if source_artifact else None,
        "test_used_for_policy_tuning": False,
        "future_information_read_by_policy": False,
        "paired_initial_conditions": True,
        "max_observed_speed_mps": max_speed,
        "max_observed_acceleration_mps2": max_acceleration,
    }
    return pd.DataFrame(metrics), pd.DataFrame(logs), metadata


def _bootstrap_ci(values: np.ndarray, rounds: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(int(rounds))]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["Wilcoxon_p_value"]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index]["Wilcoxon_p_value"]))
        running = max(running, adjusted)
        rows[index]["Holm_adjusted_p_value"] = running


def _summaries(metrics: pd.DataFrame, rounds: int) -> pd.DataFrame:
    columns = list(HIGHER_IS_BETTER) + [
        "Intervention_Count", "Unnecessary_Intervention_Count", "Missed_Intervention_Count",
        "Relay_Travel_Distance_m", "Action_Energy_Proxy", "Action_Duty_Cycle",
        "Relay_Travel_m_per_Simulated_Minute",
    ]
    rows = []
    for keys, group in metrics.groupby(["Policy", "Traffic_Load", "Topology_Density_Nodes"], sort=True):
        row = {"Policy": keys[0], "Traffic_Load": int(keys[1]), "Topology_Density_Nodes": int(keys[2]), "seed_count": int(group.seed.nunique())}
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values, rounds, _stable_seed(*keys, column, "summary"))
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_ci95_low"] = low
            row[f"{column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests(metrics: pd.DataFrame, rounds: int, cost_limits: dict) -> pd.DataFrame:
    rows: list[dict] = []
    reference = "No intervention"
    for (load, density), stratum in metrics.groupby(["Traffic_Load", "Topology_Density_Nodes"], sort=True):
        base = stratum[stratum.Policy == reference].set_index("seed")
        for policy in POLICIES[1:]:
            candidate = stratum[stratum.Policy == policy].set_index("seed")
            if set(base.index) != set(candidate.index) or base.index.has_duplicates or candidate.index.has_duplicates:
                raise RuntimeError(f"invalid seed pairing for {policy}, load={load}, density={density}")
            for metric, higher in HIGHER_IS_BETTER.items():
                raw_diff = candidate.loc[base.index, metric].to_numpy(float) - base[metric].to_numpy(float)
                benefit_diff = raw_diff if higher else -raw_diff
                low, high = _bootstrap_ci(benefit_diff, rounds, _stable_seed(policy, load, density, metric, "paired"))
                try:
                    pvalue = 1.0 if np.allclose(benefit_diff, 0.0) else float(wilcoxon(benefit_diff).pvalue)
                except ValueError:
                    pvalue = 1.0
                rows.append({
                    "Reference_Policy": reference, "Candidate_Policy": policy,
                    "Traffic_Load": int(load), "Topology_Density_Nodes": int(density),
                    "Metric": metric, "Positive_Difference_Means_Benefit": True,
                    "Paired_Seed_Count": len(benefit_diff),
                    "Paired_Mean_Benefit_Difference": float(np.mean(benefit_diff)),
                    "Paired_Median_Benefit_Difference": float(np.median(benefit_diff)),
                    "Bootstrap_CI95_Low": low, "Bootstrap_CI95_High": high,
                    "Wilcoxon_p_value": pvalue,
                })
    _holm(rows)
    for row in rows:
        subset = metrics[
            (metrics.Policy == row["Candidate_Policy"])
            & (metrics.Traffic_Load == row["Traffic_Load"])
            & (metrics.Topology_Density_Nodes == row["Topology_Density_Nodes"])
        ]
        acceptable = bool(
            subset["Relay_Travel_m_per_Simulated_Minute"].mean() <= float(cost_limits["max_travel_m_per_simulated_minute"])
            and subset["Action_Duty_Cycle"].mean() <= float(cost_limits["max_action_duty_cycle"])
        )
        row["Action_Cost_Acceptable"] = acceptable
        row["Engineering_Benefit_Supported"] = bool(
            row["Metric"] == "Packet_Delivery_Ratio"
            and float(row["Bootstrap_CI95_Low"]) > 0.0
            and float(row["Holm_adjusted_p_value"]) < 0.05
            and acceptable
        )
    return pd.DataFrame(rows)


def _plot(metrics: pd.DataFrame, output: Path) -> None:
    plot = metrics.groupby("Policy", as_index=False).agg(
        Packet_Delivery_Ratio=("Packet_Delivery_Ratio", "mean"),
        Relay_Travel=("Relay_Travel_m_per_Simulated_Minute", "mean"),
        Action_Duty=("Action_Duty_Cycle", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for _, row in plot.iterrows():
        axes[0].scatter(row.Relay_Travel, row.Packet_Delivery_Ratio, label=row.Policy, s=45)
        axes[1].scatter(row.Action_Duty, row.Packet_Delivery_Ratio, label=row.Policy, s=45)
    axes[0].set_xlabel("Relay travel (m per simulated minute)")
    axes[1].set_xlabel("Action duty cycle")
    for axis in axes:
        axis.set_ylabel("Packet delivery ratio")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(output / "closed_loop_tradeoff.pdf")
    plt.close(fig)


def _write_packet_table(metrics: pd.DataFrame, path: Path) -> None:
    grouped = metrics.groupby(["Policy", "Traffic_Load"], as_index=False).agg(
        seeds=("seed", "nunique"),
        pdr=("Packet_Delivery_Ratio", "mean"),
        deadline=("Packet_Deadline_Success", "mean"),
        p95=("Latency_P95_ms", "mean"),
        queue=("Mean_Queue_Occupancy", "mean"),
        no_route=("No_Route_Drop_Rate", "mean"),
        queue_drop=("Queue_Overflow_Drop_Rate", "mean"),
        deadline_drop=("Deadline_Miss_Drop_Rate", "mean"),
        link_drop=("Link_Failure_Drop_Rate", "mean"),
        transition_drop=("Intervention_Transition_Drop_Rate", "mean"),
    )
    labels = {
        "No intervention": "No intervention",
        "Reactive intervention": "Reactive",
        "Persistence-triggered intervention": "Persistence-triggered",
        "Current-state ExtraTrees intervention": "Current-state ExtraTrees",
        "Original Kinetic-TopoGuard intervention": "Kinetic-TopoGuard",
        "Source-Gated Kinetic-TopoGuard intervention": "Source-Gated Kinetic-TopoGuard",
    }
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Policy & Load & $n$ & PDR & Deadline & P95 (ms) & Queue & No route & Queue drop & Deadline drop & Link/transition \\",
        r"\midrule",
    ]
    for _, row in grouped.iterrows():
        label = labels[str(row["Policy"])].replace("-", r"\mbox{-}")
        combined_link = float(row["link_drop"]) + float(row["transition_drop"])
        lines.append(
            f"{label} & {int(row['Traffic_Load'])} & {int(row['seeds'])} & "
            f"{float(row['pdr']):.3f} & {float(row['deadline']):.3f} & {float(row['p95']):.2f} & "
            f"{float(row['queue']):.2f} & {float(row['no_route']):.3f} & {float(row['queue_drop']):.3f} & "
            f"{float(row['deadline_drop']):.3f} & {combined_link:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_packet_tradeoff(metrics: pd.DataFrame, output_path: Path) -> None:
    grouped = metrics.groupby(["Policy", "Traffic_Load"], as_index=False).agg(
        pdr=("Packet_Delivery_Ratio", "mean"),
        p95=("Latency_P95_ms", "mean"),
        travel=("Relay_Travel_m_per_Simulated_Minute", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for policy, group in grouped.groupby("Policy", sort=True):
        group = group.sort_values("Traffic_Load")
        axes[0].plot(group["Traffic_Load"], group["pdr"], marker="o", label=policy)
        axes[1].plot(group["travel"], group["pdr"], marker="o", label=policy)
    axes[0].set_xlabel("Packets per sampled controller tick")
    axes[0].set_ylabel("Packet delivery ratio")
    axes[1].set_xlabel("Relay travel (m per simulated minute)")
    axes[1].set_ylabel("Packet delivery ratio")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _versions() -> dict[str, str | None]:
    values = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy", "torch", "torch-geometric"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired causal closed-loop packet-controller validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--snapshot-stride", type=int)
    parser.add_argument("--bootstrap-rounds", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_dir.resolve()
    config = load_config(config_path)
    closed = config.raw["closed_loop"]
    seeds = list(args.seeds or closed["seeds"])
    rounds = int(args.bootstrap_rounds or closed["bootstrap_rounds"])
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seeds must be non-empty and unique")
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metric_frames, log_frames, metadata = [], [], []
    pending = []
    for seed in seeds:
        seed_dir = output / "per_seed" / f"seed_{seed}"
        files = [seed_dir / "metrics.csv", seed_dir / "action_log.csv", seed_dir / "complete.json"]
        if args.resume and all(path.is_file() for path in files):
            metric_frames.append(pd.read_csv(files[0]))
            log_frames.append(pd.read_csv(files[1]))
            metadata.append(json.loads(files[2].read_text(encoding="utf-8")))
        else:
            pending.append(seed)
    def save_seed(seed, result):
        metrics, logs, meta = result
        seed_dir = output / "per_seed" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(seed_dir / "metrics.csv", index=False)
        logs.to_csv(seed_dir / "action_log.csv", index=False)
        (seed_dir / "complete.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        metric_frames.append(metrics); log_frames.append(logs); metadata.append(meta)
    if int(args.workers) == 1:
        for seed in pending:
            print(f"[run] seed {seed}", flush=True)
            save_seed(seed, run_seed(seed, config_path, args.snapshot_stride))
    elif pending:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(run_seed, seed, config_path, args.snapshot_stride): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                save_seed(seed, future.result())
                print(f"[complete] seed {seed}", flush=True)
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["seed", "Policy", "Traffic_Load", "Topology_Density_Nodes"])
    logs = pd.concat(log_frames, ignore_index=True).sort_values(["seed", "policy", "traffic_load", "run_id", "time_index"])
    validate_paired_initial_conditions(logs.to_dict("records"))
    summary = _summaries(metrics, rounds)
    paired = _paired_tests(metrics, rounds, closed["acceptable_action_cost"])
    metrics.to_csv(output / "metrics_per_seed.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    paired.to_csv(output / "paired_tests.csv", index=False)
    logs.to_csv(output / "action_log.csv", index=False)
    _plot(metrics, output)
    packet_table = ROOT / "paper" / "tables" / "generated" / "closed_loop_packet_table.tex"
    packet_figure = ROOT / "paper" / "figures" / "generated" / "closed_loop_packet_tradeoff.pdf"
    _write_packet_table(metrics, packet_table)
    _plot_packet_tradeoff(metrics, output / "closed_loop_packet_tradeoff.pdf")
    _plot_packet_tradeoff(metrics, packet_figure)
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    source_paths = [config_path, ROOT / "scripts" / "run_closed_loop_controller_validation.py", ROOT / "fanet" / "closed_loop.py", ROOT / "fanet" / "packet_sim.py"]
    source_paths.extend(
        ROOT / item["source_gated_artifact"] for item in metadata if item.get("source_gated_artifact")
    )
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "config": relative_repo_path(config_path, ROOT),
        "seeds": seeds, "split_seed": int(config.sim["split_seed"]),
        "policies": list(POLICIES), "traffic_loads": closed["traffic_loads"],
        "topology_density_nodes": closed["topology_density_nodes"],
        "snapshot_stride": int(args.snapshot_stride or closed["snapshot_stride"]),
        "primary_engineering_metric": closed["primary_metric"],
        "physical_actions": ["bounded relay UAV repositioning", "relay-incident link-budget/radius adaptation"],
        "unsupported_actions_not_claimed": ["field UAV actuation", "real IP-PDR", "independent route-switching protocol"],
        "causality": "policy receives current/past controller state and current model risk score only; future labels are post-hoc metrics only",
        "threshold_selection": "model thresholds selected on validation split; reactive/persistence rules frozen a priori",
        "policy_parameter_selection": "physical/action limits pre-specified before test execution; test runs are not used for tuning",
        "paired_realizations": "mobility, radio snapshot, source-destination packet pairs, and packet RNG seed are identical across policies",
        "packet_model": {
            **config.raw["packet_simulation"],
            "traffic_loads_packets_per_sampled_tick": closed["traffic_loads"],
            "controller_tick_interval_s": float(config.sim["dt"]) * int(args.snapshot_stride or closed["snapshot_stride"]),
            "arrival_time_within_tick_s": 0.0,
            "link_capacity_model": "one shared SimPy channel at bitrate_mbps; serialization time is packet_bytes*8/bitrate",
            "deadline_success_denominator": "packets with a route at packet generation",
            "latency_quantiles": "computed over all delivered packet events, not per-snapshot means",
            "queue_occupancy": "event-sampled number of transmissions waiting for the shared channel",
            "drop_accounting": "delivered + no_route + queue_overflow + deadline_miss + link_failure + intervention_transition = generated",
            "drop_precedence": ["no route", "queue overflow", "deadline miss", "intervention transition failure", "link failure"],
        },
        "action_config": {key: value for key, value in closed.items() if key in RelayActionConfig.__dataclass_fields__},
        "effective_controller_step_seconds": float(config.sim["dt"]) * int(args.snapshot_stride or closed["snapshot_stride"]),
        "reactive_confirmation_steps": int(closed["reactive_confirmation_steps"]),
        "acceptable_action_cost": closed["acceptable_action_cost"],
        "engineering_benefit_rule": "PDR benefit CI excludes zero, Holm-adjusted Wilcoxon p<0.05, and both pre-specified action-cost limits pass",
        "kinetic_topoguard_pdr_conclusion": (
            "consistent PDR superiority supported"
            if paired[(paired["Candidate_Policy"] == "Original Kinetic-TopoGuard intervention") & (paired["Metric"] == "Packet_Delivery_Ratio")]["Engineering_Benefit_Supported"].astype(bool).all()
            else "no consistent CI-, multiplicity-, and action-cost-supported PDR superiority over no intervention"
        ),
        "bootstrap_rounds": rounds, "package_versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine()},
        "git_commit": git_commit, "runtime_seconds": time.perf_counter() - started,
        "per_seed_protocol": metadata,
        "source_files": build_file_manifest(source_paths, ROOT),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary[["Policy", "Traffic_Load", "Topology_Density_Nodes", "Packet_Delivery_Ratio_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
