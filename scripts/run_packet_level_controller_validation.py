from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import controlled_adjacency_for_snapshot, predict_generic
from fanet.packet_sim import PacketSimulationConfig, simulate_packet_tick
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard, select_best_shallow


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "packet_level_controller"


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, load), group in frame.groupby(["Model", "Packets_per_tick"]):
        row = {
            "Model": model,
            "Packets_per_tick": int(load),
            "seeds": int(group["seed"].nunique()),
        }
        for metric in [
            "PDR",
            "Mean_Delay_ms",
            "P95_Delay_ms",
            "No_Route_Drop_Rate",
            "Queue_Drop_Rate",
            "Contention_Deadline_Drop_Rate",
        ]:
            values = group[metric].dropna().astype(float)
            mean = float(values.mean())
            spread = 0.0 if len(values) < 2 else 1.96 * float(values.std(ddof=1)) / len(values) ** 0.5
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows).sort_values("PDR_mean", ascending=False)


def _run_seed(
    seed: int,
    raw: dict,
    snapshot_stride: int,
    packet_loads: list[int],
) -> list[dict]:
    sim = raw["sim"]
    snapshots = build_dataset(sim, seed=seed)
    train, val, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    models = [
        fit_kinetic_topoguard(
            train,
            val,
            int(sim["forecast_horizon_steps"]),
            float(sim["dt"]),
            seed=seed,
        ),
        select_best_shallow(train, val),
        fit_current_state_persistence(),
    ]
    rows = []
    for result in models:
        _, scores, _, aligned, threshold = predict_generic(result, test)
        sampled = list(zip(aligned, scores))[:: max(int(snapshot_stride), 1)]
        for packet_load in packet_loads:
            packet_config = PacketSimulationConfig(
                tick_duration_s=float(sim["dt"]),
                packets_per_tick=int(packet_load),
            )
            totals = {
                "generated": 0.0,
                "delivered": 0.0,
                "no_route_drops": 0.0,
                "queue_drops": 0.0,
                "deadline_drops": 0.0,
            }
            delay_samples = []
            for sample_index, (snap, score) in enumerate(sampled):
                rng = np.random.default_rng(
                    seed * 1_000_003 + sample_index + int(packet_load) * 10_007
                )
                adjacency, _, _ = controlled_adjacency_for_snapshot(
                    snap,
                    risk=float(score),
                    boost=float(raw["evaluation"]["network_radius_boost"]),
                    risk_threshold=float(threshold),
                    sim_config=sim,
                    relay_max_speed_mps=float(raw["evaluation"].get("relay_max_speed_mps", 30.0)),
                    relay_max_acceleration_mps2=float(
                        raw["evaluation"].get("relay_max_acceleration_mps2", 12.0)
                    ),
                    relay_link_budget_boost_db=float(
                        raw["evaluation"].get("relay_link_budget_boost_db", 3.0)
                    ),
                )
                metrics = simulate_packet_tick(adjacency, rng, packet_config)
                for key in totals:
                    totals[key] += metrics[key]
                delay_samples.extend(metrics["delay_samples_ms"])
            generated = max(totals["generated"], 1.0)
            delay_values = np.asarray(delay_samples, dtype=float)
            rows.append(
                {
                    "seed": seed,
                    "Model": result.model_name.split(" (")[0] if result.model_name.startswith("Shallow ML") else result.model_name,
                    "Packets_per_tick": int(packet_load),
                    "sampled_snapshots": len(sampled),
                    "generated_packets": totals["generated"],
                    "PDR": totals["delivered"] / generated,
                    "Mean_Delay_ms": float(delay_values.mean()) if delay_values.size else float("nan"),
                    "P95_Delay_ms": float(np.quantile(delay_values, 0.95)) if delay_values.size else float("nan"),
                    "No_Route_Drop_Rate": totals["no_route_drops"] / generated,
                    "Queue_Drop_Rate": totals["queue_drops"] / generated,
                    "Contention_Deadline_Drop_Rate": totals["deadline_drops"] / generated,
                }
            )
    print(f"completed seed={seed}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SimPy packet-level controller validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--snapshot-stride", type=int, default=20)
    parser.add_argument("--packets-per-tick", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    sim = raw["sim"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(int(args.workers), len(args.seeds)))
    if worker_count == 1:
        batches = [
            _run_seed(seed, raw, args.snapshot_stride, args.packets_per_tick)
            for seed in args.seeds
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            batches = list(
                executor.map(
                    _run_seed,
                    args.seeds,
                    [raw] * len(args.seeds),
                    [args.snapshot_stride] * len(args.seeds),
                    [args.packets_per_tick] * len(args.seeds),
                )
            )
    rows = [row for batch in batches for row in batch]

    per_seed = pd.DataFrame(rows)
    summary = _aggregate(per_seed)
    per_seed.to_csv(args.output_dir / "packet_metrics_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "packet_metrics_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for model, group in summary.groupby("Model"):
        group = group.sort_values("Packets_per_tick")
        axes[0].plot(group["Packets_per_tick"], 100.0 * group["PDR_mean"], marker="o", label=model)
        axes[1].plot(group["Packets_per_tick"], group["Mean_Delay_ms_mean"], marker="o", label=model)
    axes[0].set_ylabel("Packet delivery ratio (%)")
    axes[1].set_ylabel("Mean delivered-packet delay (ms)")
    axes[0].set_xlabel("Offered packets per 0.1 s tick")
    axes[1].set_xlabel("Offered packets per 0.1 s tick")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "packet_level_controller.png", dpi=220)
    fig.savefig(args.output_dir / "packet_level_controller.pdf")
    plt.close(fig)

    protocol = {
        "config": str(args.config.relative_to(ROOT)),
        "seeds": args.seeds,
        "snapshot_stride": args.snapshot_stride,
        "packets_per_tick": args.packets_per_tick,
        "parallel_workers": worker_count,
        "engine": "SimPy 4 discrete-event simulation",
        "packet_config": PacketSimulationConfig(tick_duration_s=float(sim["dt"])).__dict__,
        "scope": (
            "Simplified packet-level queue, CSMA-style random backoff, route availability, single-collision-domain "
            "contention, and per-tick deadline evaluation over physical-radio snapshot graphs. Identical packet "
            "source-destination demand and random seeds are reused across models within each seed/snapshot."
        ),
    }
    (args.output_dir / "packet_level_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
