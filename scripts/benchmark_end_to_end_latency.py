from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, graph_stats, train_val_test_split
from fanet.geometry import pairwise_distances
from fanet.graph_utils import betti_zero
from fanet.radio import build_fixed_adaptive_adjacencies
from fanet.topology import persistence_image
from fanet.training import fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "end_to_end_latency"


def _scenario_config(sim: dict, name: str) -> dict:
    result = copy.deepcopy(sim)
    base = dict(result["physical_layer"])
    for scenario in result.get("radio_scenarios", []):
        if scenario.get("name") == name:
            base.update({key: value for key, value in scenario.items() if key != "name"})
            break
    result["physical_layer"] = base
    result.pop("radio_scenarios", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the complete host-side topology-to-warning path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    sim = raw["sim"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        snapshots = build_dataset(sim, seed=seed)
        train, val, test = train_val_test_split(
            snapshots,
            split_seed=int(sim["split_seed"]),
            stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
        )
        model = fit_kinetic_topoguard(
            train,
            val,
            int(sim["forecast_horizon_steps"]),
            float(sim["dt"]),
            seed=seed,
        ).model
        model.predict_one(test[1], test[0])
        for n_nodes in sorted({snap.n_nodes for snap in test}):
            candidates = []
            grouped = {}
            for snap in test:
                if snap.n_nodes == n_nodes:
                    grouped.setdefault(snap.run_id, []).append(snap)
            for sequence in grouped.values():
                ordered = sorted(sequence, key=lambda item: item.time_index)
                candidates.extend(zip(ordered[1:], ordered[:-1]))
            sample_count = min(int(args.samples_per_size), len(candidates))
            selected = np.linspace(0, len(candidates) - 1, sample_count, dtype=int)
            for sample_index in selected:
                snap, previous = candidates[int(sample_index)]
                radio_config = _scenario_config(sim, snap.radio_scenario)
                start_total = time.perf_counter_ns()
                start = start_total
                distances = pairwise_distances(snap.positions.astype(float))
                adjacency_fixed, adjacency_adaptive = build_fixed_adaptive_adjacencies(
                    distances,
                    float(snap.radius_fixed),
                    float(snap.radius_adaptive),
                    np.random.default_rng(seed * 1_000_003 + int(sample_index)),
                    radio_config,
                )
                if snap.graph_policy == "fixed":
                    adjacency = adjacency_fixed
                    beta_current = float(betti_zero(adjacency_fixed))
                    radius = float(snap.radius_fixed)
                else:
                    adjacency = adjacency_adaptive
                    beta_current = float(betti_zero(adjacency_adaptive))
                    radius = float(snap.radius_adaptive)
                stats = graph_stats(
                    snap.positions,
                    snap.velocities,
                    adjacency,
                    beta_current,
                )
                graph_ms = (time.perf_counter_ns() - start) / 1e6

                start = time.perf_counter_ns()
                pi = persistence_image(
                    snap.positions,
                    int(sim["pi_resolution"]),
                    float(sim["pi_sigma"]),
                    float(sim["pi_max_radius"]),
                ).reshape(-1)
                pi_ms = (time.perf_counter_ns() - start) / 1e6

                updated = replace(
                    snap,
                    adjacency=adjacency.astype(np.float32),
                    adjacency_fixed=adjacency_fixed.astype(np.float32),
                    adjacency_adaptive=adjacency_adaptive.astype(np.float32),
                    pi=pi.astype(np.float32),
                    stats=stats,
                    beta_current=beta_current,
                    beta_fixed=float(betti_zero(adjacency_fixed)),
                    beta_adaptive=float(betti_zero(adjacency_adaptive)),
                    radius=radius,
                    edge_count_fixed=int(adjacency_fixed.sum() / 2),
                    edge_count_adaptive=int(adjacency_adaptive.sum() / 2),
                    is_connected=int(beta_current == 1.0),
                )
                start = time.perf_counter_ns()
                model.predict_one(updated, previous)
                forecast_ms = (time.perf_counter_ns() - start) / 1e6
                total_ms = (time.perf_counter_ns() - start_total) / 1e6
                rows.append(
                    {
                        "seed": seed,
                        "n_nodes": n_nodes,
                        "mobility": snap.mobility,
                        "graph_policy": snap.graph_policy,
                        "radio_scenario": snap.radio_scenario,
                        "graph_and_stats_ms": graph_ms,
                        "persistence_image_ms": pi_ms,
                        "feature_and_forecast_ms": forecast_ms,
                        "total_host_loop_ms": total_ms,
                    }
                )

    samples = pd.DataFrame(rows)
    summary = samples.groupby("n_nodes", as_index=False).agg(
        samples=("total_host_loop_ms", "size"),
        total_mean_ms=("total_host_loop_ms", "mean"),
        total_p50_ms=("total_host_loop_ms", "median"),
        total_p95_ms=("total_host_loop_ms", lambda values: float(np.quantile(values, 0.95))),
        total_p99_ms=("total_host_loop_ms", lambda values: float(np.quantile(values, 0.99))),
        graph_and_stats_mean_ms=("graph_and_stats_ms", "mean"),
        persistence_mean_ms=("persistence_image_ms", "mean"),
        forecast_mean_ms=("feature_and_forecast_ms", "mean"),
    )
    samples.to_csv(args.output_dir / "latency_samples.csv", index=False)
    summary.to_csv(args.output_dir / "latency_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(6.8, 3.7))
    axis.plot(summary["n_nodes"], summary["total_mean_ms"], marker="o", label="Mean")
    axis.plot(summary["n_nodes"], summary["total_p95_ms"], marker="s", label="P95")
    axis.plot(summary["n_nodes"], summary["total_p99_ms"], marker="^", label="P99")
    axis.axhline(20.0, color="#a23b2a", linestyle="--", label="20 ms budget")
    axis.set(xlabel="Swarm size", ylabel="Host-loop latency (ms)", title="End-to-end host-side latency")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "end_to_end_latency.png", dpi=220)
    fig.savefig(args.output_dir / "end_to_end_latency.pdf")
    plt.close(fig)

    protocol = {
        "config": str(args.config.relative_to(ROOT)),
        "seeds": args.seeds,
        "samples_per_swarm_size_per_seed": args.samples_per_size,
        "timed_stages": [
            "pairwise distance, physical-radio fixed/adaptive adjacency, and graph-statistic construction",
            "H0 persistence-image construction",
            "Kinetic-TopoGuard feature assembly and risk/regression inference",
        ],
        "scope": "Host-side algorithmic loop from observed positions to warning score; sensor acquisition and flight-controller transport are excluded.",
        "platform": {
            "system": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    (args.output_dir / "latency_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
