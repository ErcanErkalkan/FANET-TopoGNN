from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, graph_stats, train_val_test_split
from fanet.geometry import normalize_positions, pairwise_distances
from fanet.graph_utils import betti_zero
from fanet.provenance import build_file_manifest, relative_repo_path
from fanet.radio import build_fixed_adaptive_adjacencies
from fanet.runtime_benchmark import (
    STAGES,
    TOTAL_STAGE,
    environment_manifest,
    summarize_latency_samples,
    thread_limit_context,
    validate_percentile_order,
    validate_stage_totals,
    write_json,
)
from fanet.source_gated import (
    FEATURE_GROUPS,
    SourceGatedKineticTopoGuard,
    _calibrated_predict,
    _positive_probability,
    fit_current_state_extratrees,
    source_feature_groups,
)
from fanet.topology import persistence_image
from fanet.training import (
    TrainResult,
    collate_snapshots,
    fit_kinetic_topoguard,
    kinetic_topoguard_feature_vector,
    train_torch_model,
)


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "edge_runtime_benchmark"
SOURCE_GATED_ARTIFACT = ROOT / "outputs/source_gated_development/per_seed/seed_7/source_gated_model.pkl"
MODEL_NAMES = (
    "Current-state ExtraTrees",
    "Kinetic-TopoGuard",
    "Source-Gated Kinetic-TopoGuard",
    "FANET-TopoGNN",
)


def _ns(callable_) -> tuple[object, float]:
    start = time.perf_counter_ns()
    value = callable_()
    return value, (time.perf_counter_ns() - start) / 1e6


def _scenario_config(sim: dict, name: str) -> dict:
    result = copy.deepcopy(sim)
    physical = dict(result["physical_layer"])
    for scenario in result.get("radio_scenarios", []):
        if scenario.get("name") == name:
            physical.update({key: value for key, value in scenario.items() if key != "name"})
            break
    result["physical_layer"] = physical
    result.pop("radio_scenarios", None)
    return result


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() or None


def _current_features(snapshot, previous) -> np.ndarray:
    return np.asarray(
        [[
            snapshot.beta_current,
            previous.beta_current if previous is not None else snapshot.beta_current,
            snapshot.beta_current - previous.beta_current if previous is not None else 0.0,
            snapshot.n_nodes / 30.0,
            snapshot.radius / 1000.0,
        ]],
        dtype=np.float32,
    )


def _torch_inputs(snapshot) -> dict:
    batch = collate_snapshots([snapshot])
    return {key: value for key, value in batch.items() if key != "y"}


def _fit_models(raw: dict, seed: int, training_steps: int, neural_epochs: int) -> dict[str, TrainResult]:
    sim = copy.deepcopy(raw["sim"])
    sim["time_steps"] = int(training_steps)
    sim["swarm_sizes"] = [10, 20, 30]
    snapshots = build_dataset(sim, seed=seed)
    train, validation, _ = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    horizon = int(sim["forecast_horizon_steps"])
    dt = float(sim["dt"])
    models = {
        "Current-state ExtraTrees": fit_current_state_extratrees(train, validation, horizon, dt, seed),
        "Kinetic-TopoGuard": fit_kinetic_topoguard(train, validation, horizon, dt, seed),
    }
    if not SOURCE_GATED_ARTIFACT.exists():
        raise FileNotFoundError(
            f"frozen Source-Gated model is missing: {relative_repo_path(SOURCE_GATED_ARTIFACT, ROOT)}"
        )
    source_model = SourceGatedKineticTopoGuard.load(SOURCE_GATED_ARTIFACT)
    models["Source-Gated Kinetic-TopoGuard"] = TrainResult(
        model=source_model, model_name="Source-Gated Kinetic-TopoGuard", inference_ms=0.0
    )
    torch_cfg = dict(raw["training"])
    torch_cfg["epochs"] = int(neural_epochs)
    torch_cfg["patience"] = min(int(torch_cfg.get("patience", neural_epochs)), int(neural_epochs))
    models["FANET-TopoGNN"] = train_torch_model(
        "FANET-TopoGNN", train, validation, torch_cfg, len(train[0].pi), seed
    )
    return models


def _serialize_models(models: dict[str, TrainResult], sample, output_dir: Path) -> pd.DataFrame:
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, result in models.items():
        model = result.model
        if name == "Source-Gated Kinetic-TopoGuard":
            path = artifact_dir / "source_gated_kinetic_topoguard.pkl"
            model.save(path)
            restored = SourceGatedKineticTopoGuard.load(path)
            before = model.predict_snapshots([sample])[0]
            after = restored.predict_snapshots([sample])[0]
            method = "pickle via typed SourceGatedKineticTopoGuard.save/load; trusted local artifacts only"
        elif name == "FANET-TopoGNN":
            import torch

            path = artifact_dir / "fanet_topognn_state_dict.pt"
            torch.save(model.state_dict(), path)
            restored = copy.deepcopy(model)
            state = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(state)
            model.eval()
            restored.eval()
            inputs = _torch_inputs(sample)
            with torch.no_grad():
                before = model(inputs["x"], inputs["adj"], inputs["pi"], inputs["mask"]).numpy()
                after = restored(inputs["x"], inputs["adj"], inputs["pi"], inputs["mask"]).numpy()
            method = "PyTorch state_dict; weights_only=True reload"
        else:
            slug = name.lower().replace("-", "_").replace(" ", "_")
            path = artifact_dir / f"{slug}.joblib"
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)
            before = model.predict_snapshots([sample])[0]
            after = restored.predict_snapshots([sample])[0]
            method = "joblib; trusted local artifacts only (joblib is not safe for untrusted files)"
        max_error = float(np.max(np.abs(np.asarray(before, dtype=float) - np.asarray(after, dtype=float))))
        if max_error > 1e-9:
            raise RuntimeError(f"serialization prediction mismatch for {name}: {max_error}")
        rows.append(
            {
                "model": name,
                "serialization_format": path.suffix.lstrip("."),
                "artifact_path": relative_repo_path(path, ROOT),
                "file_size_bytes": int(path.stat().st_size),
                "reload_max_abs_prediction_error": max_error,
                "prediction_consistent": True,
                "trust_boundary": method,
            }
        )
    return pd.DataFrame(rows)


def _benchmark_iteration(model_name, result, snap, previous, sim, seed, iteration) -> tuple[list[float], float, float]:
    values = []
    distances, elapsed = _ns(lambda: pairwise_distances(snap.positions.astype(float)))
    values.append(elapsed)
    radio_cfg = _scenario_config(sim, snap.radio_scenario)
    rng = np.random.default_rng(seed * 1_000_003 + snap.n_nodes * 10_007 + iteration)
    (fixed, adaptive), elapsed = _ns(
        lambda: build_fixed_adaptive_adjacencies(
            distances, float(snap.radius_fixed), float(snap.radius_adaptive), rng, radio_cfg
        )
    )
    values.append(elapsed)
    adjacency = fixed if snap.graph_policy == "fixed" else adaptive
    beta = float(betti_zero(adjacency))
    stats, elapsed = _ns(lambda: graph_stats(snap.positions, snap.velocities, adjacency, beta))
    values.append(elapsed)
    pi, elapsed = _ns(
        lambda: persistence_image(
            snap.positions, int(sim["pi_resolution"]), float(sim["pi_sigma"]), float(sim["pi_max_radius"])
        ).reshape(-1).astype(np.float32)
    )
    values.append(elapsed)
    updated = replace(
        snap,
        adjacency=adjacency.astype(np.float32),
        adjacency_fixed=fixed.astype(np.float32),
        adjacency_adaptive=adaptive.astype(np.float32),
        pi=pi,
        stats=stats,
        beta_current=beta,
        beta_fixed=float(betti_zero(fixed)),
        beta_adaptive=float(betti_zero(adaptive)),
        edge_count_fixed=int(fixed.sum() / 2),
        edge_count_adaptive=int(adaptive.sum() / 2),
        is_connected=int(beta == 1.0),
        node_features=np.concatenate(
            [normalize_positions(snap.positions), (snap.velocities / 30.0).astype(np.float32)], axis=1
        ).astype(np.float32),
    )
    model = result.model
    if model_name == "Current-state ExtraTrees":
        features, elapsed = _ns(lambda: _current_features(updated, previous))
        values.append(elapsed)
        residual, elapsed = _ns(lambda: model.regressor.predict(features))
        count = np.clip(updated.beta_current + model.residual_scale * residual, 1.0, None)
        values.append(elapsed)
        risk, elapsed = _ns(lambda: _positive_probability(model.classifier, features))
        values.append(elapsed)
        calibrated, elapsed = _ns(lambda: np.clip(risk, 0.0, 1.0))
        values.append(elapsed)
        threshold = model.risk_threshold
    elif model_name == "Kinetic-TopoGuard":
        (features, kinetic), elapsed = _ns(
            lambda: kinetic_topoguard_feature_vector(updated, previous, model.horizon_steps, model.dt)
        )
        matrix = features.reshape(1, -1)
        values.append(elapsed)
        count, elapsed = _ns(lambda: model._predict_from_matrix(matrix, [updated]))
        values.append(elapsed)
        risk, elapsed = _ns(
            lambda: model._risk_from_matrix(matrix, [updated], np.asarray([kinetic]), count)
        )
        values.append(elapsed)
        calibrated, elapsed = _ns(lambda: np.clip(risk, 0.0, 1.0))
        values.append(elapsed)
        threshold = model.risk_threshold
    elif model_name == "Source-Gated Kinetic-TopoGuard":
        groups, elapsed = _ns(
            lambda: source_feature_groups(updated, previous, model.horizon_steps, model.dt)
        )
        matrices = {name: vector.reshape(1, -1) for name, vector in groups.items()}
        values.append(elapsed)
        def count_call():
            base = np.column_stack([model.count_base_models[name].predict(matrices[name]) for name in FEATURE_GROUPS])
            return np.clip(updated.beta_current + model.count_meta_model.predict(base), 1.0, None)
        count, elapsed = _ns(count_call)
        values.append(elapsed)
        def risk_call():
            base = np.column_stack([_positive_probability(model.risk_base_models[name], matrices[name]) for name in FEATURE_GROUPS])
            return model.risk_meta_model.predict_proba(base)[:, 1]
        raw_risk, elapsed = _ns(risk_call)
        values.append(elapsed)
        calibrated, elapsed = _ns(lambda: np.clip(_calibrated_predict(model.calibrator, raw_risk), 0.0, 1.0))
        values.append(elapsed)
        threshold = model.risk_threshold
    else:
        import torch

        inputs, elapsed = _ns(lambda: _torch_inputs(updated))
        values.append(elapsed)
        model.eval()
        def torch_call():
            with torch.no_grad():
                return model(inputs["x"], inputs["adj"], inputs["pi"], inputs["mask"]).cpu().numpy()
        count, elapsed = _ns(torch_call)
        values.append(elapsed)
        risk, elapsed = _ns(lambda: (np.asarray(count) > 1.0).astype(float))
        values.append(elapsed)
        calibrated, elapsed = _ns(lambda: np.asarray(risk, dtype=float))
        values.append(elapsed)
        threshold = 0.5
    _, elapsed = _ns(lambda: bool(float(np.asarray(calibrated).reshape(-1)[0]) >= float(threshold)))
    values.append(elapsed)
    return values, float(np.asarray(count).reshape(-1)[0]), float(np.asarray(calibrated).reshape(-1)[0])


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _write_runtime_table(summary: pd.DataFrame, model_sizes: pd.DataFrame) -> Path:
    totals = summary.loc[(summary["stage"] == TOTAL_STAGE) & (summary["n_nodes"] == 30)].copy()
    totals = totals.merge(model_sizes[["model", "file_size_bytes"]], on="model", how="left")
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Threads & P50 & P95 & P99 (ms) & Size (MiB) \\", r"\midrule",
    ]
    for row in totals.sort_values(["model", "thread_mode"]).itertuples():
        label = str(row.model).replace("_", r"\_")
        lines.append(
            f"{label} & {row.thread_mode} & {row.p50_ms:.2f} & {row.p95_ms:.2f} & "
            f"{row.p99_ms:.2f} & {row.file_size_bytes / (1024 ** 2):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination = ROOT / "paper/tables/generated/runtime_table.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    started_at = datetime.now(timezone.utc)
    start_wall = time.perf_counter()
    parser = argparse.ArgumentParser(description="Benchmark measured host-side edge-runtime stages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-counts", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--timed-iterations", type=int, default=30)
    parser.add_argument("--training-steps", type=int, default=16)
    parser.add_argument("--neural-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing benchmark output set.")
    args = parser.parse_args()
    if args.warmup_iterations < 1 or args.timed_iterations < 1:
        raise ValueError("warm-up and timed iteration counts must be positive")
    scientific_outputs = [args.output_dir / name for name in ["latency_samples.csv", "latency_summary.csv", "protocol.json"]]
    existing = [path for path in scientific_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing benchmark; choose a versioned --output-dir or pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_manifest()
    write_json(args.output_dir / "environment.json", environment)

    models = _fit_models(raw, args.seed, args.training_steps, args.neural_epochs)
    benchmark_sim = copy.deepcopy(raw["sim"])
    benchmark_sim["time_steps"] = max(int(benchmark_sim["forecast_horizon_steps"]) + 2, 8)
    benchmark_sim["swarm_sizes"] = sorted(set(args.node_counts))
    benchmark_sim["runs_per_setting"] = 1
    snapshots = build_dataset(benchmark_sim, seed=args.seed + 900_001)
    by_size = {}
    for n_nodes in args.node_counts:
        candidates = sorted(
            [item for item in snapshots if item.n_nodes == n_nodes],
            key=lambda item: (item.run_id, item.time_index),
        )
        if len(candidates) < 2:
            raise RuntimeError(f"not enough benchmark snapshots for n_nodes={n_nodes}")
        by_size[n_nodes] = (candidates[1], candidates[0])

    model_sizes = _serialize_models(models, by_size[args.node_counts[0]][0], args.output_dir)
    model_sizes.to_csv(args.output_dir / "model_sizes.csv", index=False)
    rows = []
    memory_rows = []
    for thread_mode in ["single", "default"]:
        with thread_limit_context(thread_mode):
            for model_name in MODEL_NAMES:
                baseline_rss = _rss_bytes()
                peak_rss = baseline_rss
                for n_nodes in args.node_counts:
                    snap, previous = by_size[n_nodes]
                    total_iterations = args.warmup_iterations + args.timed_iterations
                    for iteration in range(total_iterations):
                        phase = "warmup" if iteration < args.warmup_iterations else "timed"
                        stage_values, _, _ = _benchmark_iteration(
                            model_name, models[model_name], snap, previous, benchmark_sim, args.seed, iteration
                        )
                        additive_total = float(sum(stage_values))
                        for stage, latency in zip(STAGES, stage_values):
                            rows.append({"model": model_name, "thread_mode": thread_mode, "n_nodes": n_nodes, "phase": phase, "iteration": iteration, "stage": stage, "latency_ms": latency})
                        rows.append({"model": model_name, "thread_mode": thread_mode, "n_nodes": n_nodes, "phase": phase, "iteration": iteration, "stage": TOTAL_STAGE, "latency_ms": additive_total})
                        current_rss = _rss_bytes()
                        if current_rss is not None:
                            peak_rss = max(peak_rss or current_rss, current_rss)
                memory_rows.append(
                    {
                        "model": model_name,
                        "thread_mode": thread_mode,
                        "baseline_rss_bytes": baseline_rss,
                        "peak_observed_rss_bytes": peak_rss,
                        "peak_increment_bytes": None if peak_rss is None or baseline_rss is None else max(peak_rss - baseline_rss, 0),
                        "measurement_scope": "host process RSS sampled after every complete host-loop iteration",
                    }
                )
    samples = pd.DataFrame(rows)
    validate_stage_totals(samples)
    summary = summarize_latency_samples(samples)
    validate_percentile_order(summary)
    samples.to_csv(args.output_dir / "latency_samples.csv", index=False)
    summary.to_csv(args.output_dir / "latency_summary.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(args.output_dir / "memory_summary.csv", index=False)

    totals = summary.loc[summary["stage"] == TOTAL_STAGE]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for axis, thread_mode in zip(axes, ["single", "default"]):
        subset = totals.loc[totals["thread_mode"] == thread_mode]
        for model_name, group in subset.groupby("model"):
            group = group.sort_values("n_nodes")
            axis.plot(group["n_nodes"], group["p50_ms"], marker="o", label=f"{model_name} P50")
            axis.plot(group["n_nodes"], group["p95_ms"], marker=".", linestyle="--", alpha=0.8)
        axis.set_title(f"{thread_mode.title()} threads")
        axis.set_xlabel("UAV count")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Measured additive host-loop latency (ms)")
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output_dir / "runtime_scaling.pdf")
    plt.close(figure)
    table = _write_runtime_table(summary, model_sizes)

    onnx_status = {
        "status": "unsupported",
        "reason": "onnx is not installed and the Python/NumPy radio, graph-statistic, persistence-image, and feature-assembly preprocessing is not represented by an ONNX graph",
        "onnxruntime_present": environment["packages"].get("onnxruntime") is not None,
    }
    protocol = {
        "benchmark_kind": "measured host-side algorithm loop",
        "scope": "Observed positions/velocities through preprocessing, model inference, calibration, and controller decision on this host.",
        "explicit_exclusions": ["sensor acquisition", "telemetry transport", "operating-system scheduling outside the process", "autopilot communication", "actuation", "network transfer"],
        "not_a_sensor_to_actuator_measurement": True,
        "config": relative_repo_path(args.config, ROOT),
        "seed": args.seed,
        "node_counts": args.node_counts,
        "higher_node_sensitivity": max(args.node_counts) if max(args.node_counts) > 30 else None,
        "models": list(MODEL_NAMES),
        "selected_neural_models": ["FANET-TopoGNN"],
        "neural_risk_semantics": "deterministic count>1 conversion used by the existing evaluation pipeline; no learned neural risk head or calibration is claimed",
        "stages": list(STAGES) + [TOTAL_STAGE],
        "total_definition": "sum of mutually exclusive measured stages; timing-call overhead is excluded",
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations": args.timed_iterations,
        "warmup_excluded_from_summary": True,
        "thread_modes": {"single": "threadpoolctl limit=1", "default": "process/library defaults"},
        "training_profile": {"simulation_steps": args.training_steps, "neural_epochs": args.neural_epochs, "source_gated_artifact": relative_repo_path(SOURCE_GATED_ARTIFACT, ROOT)},
        "serialization": {"sklearn": "joblib trusted-local-only", "source_gated": "typed pickle loader trusted-local-only", "pytorch": "state_dict and weights_only=True reload"},
        "onnx_export": onnx_status,
        "jetson": {"measured": False, "claim": "No Jetson hardware benchmark was performed; no Jetson latency or memory result is reported."},
        "gpu_measured": bool(environment["gpu"]["available"]),
        "git_commit": _git_commit(),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_runtime_seconds": float(time.perf_counter() - start_wall),
        "environment": environment,
        "source_files": build_file_manifest([args.config, Path(__file__), ROOT / "fanet/runtime_benchmark.py", ROOT / "fanet/training.py", ROOT / "fanet/source_gated.py", ROOT / "fanet/models.py"], ROOT),
        "outputs": [relative_repo_path(args.output_dir / name, ROOT) for name in ["latency_samples.csv", "latency_summary.csv", "memory_summary.csv", "model_sizes.csv", "environment.json", "protocol.json", "runtime_scaling.pdf"]],
        "paper_table": relative_repo_path(table, ROOT),
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print(totals.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
