from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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

from fanet.closed_loop import traffic_realization_seed
from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import controlled_adjacency_for_snapshot, predict_generic
from fanet.packet_sim import PacketSimulationConfig, generate_traffic_pairs, simulate_packet_tick
from fanet.provenance import build_file_manifest, relative_repo_path
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard, select_best_shallow


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "packet_level_controller_v2"
DROP_KEYS = (
    "no_route_drops", "queue_drops", "deadline_drops",
    "link_failure_drops", "intervention_transition_drops",
)
METRICS = (
    "PDR", "Deadline_Success", "Latency_P50_ms", "Latency_P95_ms", "Latency_P99_ms",
    "Mean_Queue_Occupancy", "Max_Queue_Occupancy", "No_Route_Drop_Rate",
    "Queue_Drop_Rate", "Contention_Deadline_Drop_Rate", "Link_Failure_Drop_Rate",
    "Intervention_Transition_Drop_Rate",
)


def _packet_config(sim: dict, load: int) -> PacketSimulationConfig:
    return PacketSimulationConfig(
        tick_duration_s=float(sim["dt"]),
        packet_deadline_s=float(sim["dt"]),
        packets_per_tick=int(load),
        packet_bytes=1200,
        bitrate_mbps=6.0,
        queue_limit=64,
        difs_us=34.0,
        slot_us=9.0,
        max_backoff_slots=15,
        propagation_us_per_hop=5.0,
        link_failure_probability_per_attempt=0.0,
        intervention_transition_failure_probability_per_attempt=0.0,
        max_retransmissions=0,
    )


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, load), group in frame.groupby(["Model", "Packets_per_tick"], sort=True):
        row = {"Model": model, "Packets_per_tick": int(load), "seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            spread = 0.0 if len(values) < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Packets_per_tick", "PDR_mean"], ascending=[True, False])


def _run_seed(seed: int, config_path: Path, snapshot_stride: int, packet_loads: list[int]) -> list[dict]:
    config = load_config(config_path)
    sim = config.sim
    snapshots = build_dataset(sim, seed=seed)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    models = [
        fit_kinetic_topoguard(train, validation, int(sim["forecast_horizon_steps"]), float(sim["dt"]), seed=seed),
        select_best_shallow(train, validation),
        fit_current_state_persistence(),
    ]
    model_scores = {}
    thresholds = {}
    model_names = {}
    for result in models:
        _, scores, _, aligned, threshold = predict_generic(result, test)
        name = result.model_name.split(" (")[0] if result.model_name.startswith("Shallow ML") else result.model_name
        model_names[result.model_name] = name
        model_scores[name] = {(snapshot.run_id, int(snapshot.time_index)): float(score) for snapshot, score in zip(aligned, scores)}
        thresholds[name] = float(threshold)
    sampled = sorted(test, key=lambda snapshot: (snapshot.run_id, snapshot.time_index))[:: max(int(snapshot_stride), 1)]
    rows = []
    for name in model_scores:
        for load in packet_loads:
            cfg = _packet_config(sim, int(load))
            totals = {key: 0.0 for key in ("generated", "delivered", "routable_packets", *DROP_KEYS)}
            delay_samples: list[float] = []
            occupancy_sum = 0.0
            occupancy_count = 0
            occupancy_max = 0.0
            trace_hashes = []
            for snapshot in sampled:
                score = model_scores[name].get((snapshot.run_id, int(snapshot.time_index)), 0.0)
                adjacency, relays, proactive = controlled_adjacency_for_snapshot(
                    snapshot,
                    risk=score,
                    boost=float(config.evaluation["network_radius_boost"]),
                    risk_threshold=thresholds[name],
                    sim_config=sim,
                    relay_max_speed_mps=float(config.evaluation.get("relay_max_speed_mps", 30.0)),
                    relay_max_acceleration_mps2=float(config.evaluation.get("relay_max_acceleration_mps2", 12.0)),
                    relay_link_budget_boost_db=float(config.evaluation.get("relay_link_budget_boost_db", 3.0)),
                )
                arrival_seed = traffic_realization_seed(seed, snapshot.run_id, int(snapshot.time_index), int(load))
                pairs = generate_traffic_pairs(snapshot.n_nodes, int(load), np.random.default_rng(arrival_seed))
                trace_hashes.append(f"{snapshot.run_id}:{snapshot.time_index}:{arrival_seed}")
                packet = simulate_packet_tick(
                    adjacency,
                    np.random.default_rng((arrival_seed + 1_000_003) % (2**63 - 1)),
                    cfg,
                    traffic_pairs=pairs,
                    transition_nodes=relays if proactive else [],
                )
                for key in totals:
                    totals[key] += float(packet[key])
                delay_samples.extend(float(value) for value in packet["delay_samples_ms"])
                count = int(packet["queue_occupancy_observations"])
                occupancy_sum += float(packet["mean_queue_occupancy"]) * count
                occupancy_count += count
                occupancy_max = max(occupancy_max, float(packet["max_queue_occupancy"]))
            generated = max(totals["generated"], 1.0)
            routable = max(totals["routable_packets"], 1.0)
            delay = np.asarray(delay_samples, dtype=float)
            accounting = totals["delivered"] + sum(totals[key] for key in DROP_KEYS)
            rows.append({
                "seed": int(seed), "Model": name, "Packets_per_tick": int(load),
                "sampled_snapshots": len(sampled), "generated_packets": totals["generated"],
                "arrival_trace_signature": hashlib.sha256("|".join(trace_hashes).encode("utf-8")).hexdigest(),
                "PDR": totals["delivered"] / generated,
                "Deadline_Success": totals["delivered"] / routable,
                "Mean_Delay_ms": float(np.mean(delay)) if delay.size else float("nan"),
                "Latency_P50_ms": float(np.quantile(delay, 0.50)) if delay.size else float("nan"),
                "Latency_P95_ms": float(np.quantile(delay, 0.95)) if delay.size else float("nan"),
                "Latency_P99_ms": float(np.quantile(delay, 0.99)) if delay.size else float("nan"),
                "P95_Delay_ms": float(np.quantile(delay, 0.95)) if delay.size else float("nan"),
                "Mean_Queue_Occupancy": occupancy_sum / max(occupancy_count, 1),
                "Max_Queue_Occupancy": occupancy_max,
                "No_Route_Drop_Rate": totals["no_route_drops"] / generated,
                "Queue_Drop_Rate": totals["queue_drops"] / generated,
                "Contention_Deadline_Drop_Rate": totals["deadline_drops"] / generated,
                "Link_Failure_Drop_Rate": totals["link_failure_drops"] / generated,
                "Intervention_Transition_Drop_Rate": totals["intervention_transition_drops"] / generated,
                "Drop_Accounting_Error": abs(totals["generated"] - accounting),
            })
    print(f"completed seed={seed}", flush=True)
    return rows


def _paired_tests(per_seed: pd.DataFrame) -> pd.DataFrame:
    reference_name = "Current-state persistence baseline"
    rows = []
    for load, load_frame in per_seed.groupby("Packets_per_tick", sort=True):
        reference = load_frame[load_frame.Model == reference_name].set_index("seed")
        for model in sorted(set(load_frame.Model) - {reference_name}):
            candidate = load_frame[load_frame.Model == model].set_index("seed")
            if set(reference.index) != set(candidate.index):
                raise RuntimeError(f"unpaired seeds for {model}, load={load}")
            for metric in ("PDR", "Deadline_Success", "Latency_P95_ms", "No_Route_Drop_Rate", "Contention_Deadline_Drop_Rate"):
                difference = candidate.loc[reference.index, metric].to_numpy(float) - reference[metric].to_numpy(float)
                if metric not in {"PDR", "Deadline_Success"}:
                    difference = -difference
                try:
                    pvalue = 1.0 if np.allclose(difference, 0.0) else float(wilcoxon(difference).pvalue)
                except ValueError:
                    pvalue = 1.0
                rows.append({
                    "Reference": reference_name, "Candidate": model, "Packets_per_tick": int(load),
                    "Metric": metric, "Paired_Seed_Count": len(difference),
                    "Paired_Mean_Benefit_Difference": float(np.mean(difference)),
                    "Paired_Median_Benefit_Difference": float(np.median(difference)),
                    "Wilcoxon_p_value": pvalue,
                })
    return pd.DataFrame(rows)


def _versions() -> dict[str, str | None]:
    values = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run auditable one-snapshot SimPy packet validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--snapshot-stride", type=int, default=20)
    parser.add_argument("--packets-per-tick", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    worker_count = max(1, min(int(args.workers), len(args.seeds)))
    if worker_count == 1:
        batches = [_run_seed(seed, args.config.resolve(), args.snapshot_stride, args.packets_per_tick) for seed in args.seeds]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            batches = list(executor.map(
                _run_seed, args.seeds, [args.config.resolve()] * len(args.seeds),
                [args.snapshot_stride] * len(args.seeds), [args.packets_per_tick] * len(args.seeds),
            ))
    per_seed = pd.DataFrame([row for batch in batches for row in batch]).sort_values(["seed", "Model", "Packets_per_tick"])
    if float(per_seed["Drop_Accounting_Error"].max()) > 1e-9:
        raise RuntimeError("packet drop accounting failed")
    trace_counts = per_seed.groupby(["seed", "Packets_per_tick"])["arrival_trace_signature"].nunique()
    if not trace_counts.eq(1).all():
        raise RuntimeError("models did not receive the same arrival trace")
    summary = _aggregate(per_seed)
    paired = _paired_tests(per_seed)
    per_seed.to_csv(output / "packet_metrics_per_seed.csv", index=False)
    summary.to_csv(output / "packet_metrics_summary.csv", index=False)
    paired.to_csv(output / "packet_paired_tests.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for model, group in summary.groupby("Model"):
        group = group.sort_values("Packets_per_tick")
        axes[0].plot(group["Packets_per_tick"], group["PDR_mean"], marker="o", label=model)
        axes[1].plot(group["Packets_per_tick"], group["Latency_P95_ms_mean"], marker="o", label=model)
    axes[0].set_ylabel("Packet delivery ratio")
    axes[1].set_ylabel("Delivered-packet P95 latency (ms)")
    for axis in axes:
        axis.set_xlabel("Packets per 0.1 s batch")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "packet_level_controller.png", dpi=220)
    fig.savefig(output / "packet_level_controller.pdf")
    plt.close(fig)
    config = load_config(args.config)
    packet_config = _packet_config(config.sim, 16)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative_repo_path(args.config.resolve(), ROOT), "seeds": args.seeds,
        "split_seed": int(config.sim["split_seed"]), "snapshot_stride": args.snapshot_stride,
        "packets_per_tick": args.packets_per_tick, "parallel_workers": worker_count,
        "engine": "SimPy 4 discrete-event simulation",
        "experiment_role": "one-snapshot packet-event sensitivity; not the stateful closed-loop policy experiment",
        "traffic_arrival_model": "fixed-size batch at time zero of each sampled tick; deterministic source-destination pairs shared across models",
        "queue_model": "finite FIFO waiting queue feeding one shared single-collision-domain SimPy Resource",
        "packet_size_bytes": packet_config.packet_bytes,
        "link_capacity_mbps": packet_config.bitrate_mbps,
        "deadline_seconds": packet_config.packet_deadline_s,
        "routing": "unweighted shortest path recomputed per packet on the controller-adjusted physical adjacency",
        "retransmission": {"max_retransmissions": packet_config.max_retransmissions, "enabled": packet_config.max_retransmissions > 0},
        "drop_causes": {
            "no_route": "no path on the physical adjacency at packet generation",
            "queue_overflow": f"shared waiting queue already contains {packet_config.queue_limit} transmissions",
            "deadline_miss": "packet remains unfinished or crosses its deadline",
            "link_failure": "within-tick per-attempt failure; disabled because physical link realization is already represented by adjacency",
            "intervention_transition": "moving-relay handover loss; separately counted but disabled without a measured parameter",
        },
        "packet_config": asdict(packet_config),
        "deadline_success_denominator": "packets having a route at generation",
        "latency_quantiles": "all delivered packet events",
        "queue_occupancy": "event-sampled waiting transmissions",
        "paired_comparison_unit": "seed",
        "scope": "PDR is produced by packet events; graph reachability proxies are not labelled PDR.",
        "runtime_seconds": time.perf_counter() - started,
        "package_versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "git_commit": commit,
        "source_files": build_file_manifest([args.config.resolve(), ROOT / "fanet/packet_sim.py", ROOT / "scripts/run_packet_level_controller_validation.py"], ROOT),
    }
    (output / "packet_level_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
