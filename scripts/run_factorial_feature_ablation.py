from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import Snapshot, build_dataset, train_val_test_split
from fanet.evaluation import (
    alert_event_metrics,
    classification_metrics,
    risk_probability_metrics,
)
from fanet.geometry import pairwise_distances
from fanet.graph_utils import adjacency_from_radius, betti_zero
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file


DEFAULT_CONFIG = ROOT / "configs" / "paper_like_submission.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "factorial_feature_ablation_20seed"
SOURCES = ("graph", "topology", "kinematic")
CURRENT_ONLY_MODEL = "Current-state ExtraTrees"
THRESHOLD_GRID = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
RESIDUAL_SCALE_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
LEARNER_HYPERPARAMETERS = {
    "regressor": {
        "class": "sklearn.ensemble.ExtraTreesRegressor",
        "n_estimators": 128,
        "min_samples_leaf": 2,
        "n_jobs": 2,
        "random_state_offset": 101,
        "target": "beta_target minus current beta",
    },
    "classifier": {
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "n_estimators": 128,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 2,
        "random_state_offset": 211,
    },
}
FEATURE_GROUP_DEFINITIONS = {
    "current": "current and lagged beta0, beta0 change, node count, and active radius",
    "graph": "current physical-adjacency degree, component, clustering, path, and edge-density statistics",
    "topology": "H0 persistence image and persistence-image distribution summaries",
    "kinematic": "current/projected pair-distance, velocity, projected-component, and radius-crossing summaries",
}
PER_SEED_FILENAME = "factorial_ablation_per_seed.csv"
SEED_PROTOCOL_FILENAME = "seed_protocol.json"
SUCCESS_FILENAME = "_SUCCESS.json"
METRIC_COLUMNS = (
    "MAE",
    "R2",
    "Risk_Precision",
    "Risk_Recall",
    "Risk_F1",
    "Alert_Event_Precision",
    "Alert_Event_Recall",
    "Alert_Event_F1",
    "False_Alert_Events_per_minute",
    "Risk_Brier",
    "Risk_ECE",
    "selected_threshold",
    "residual_scale",
    "feature_count",
    "Training_Time_s",
    "Inference_Time_ms",
)


def feature_combinations() -> list[tuple[str, ...]]:
    return [
        combination
        for size in range(len(SOURCES) + 1)
        for combination in itertools.combinations(SOURCES, size)
    ]


def feature_label(selected_sources: tuple[str, ...]) -> str:
    return "current-only" if not selected_sources else "+".join(selected_sources)


def model_name(selected_sources: tuple[str, ...]) -> str:
    if not selected_sources:
        return CURRENT_ONLY_MODEL
    return "ExtraTrees (current + " + " + ".join(selected_sources) + ")"


def _summary(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(values.mean()) if values.size else 0.0,
            float(values.std()) if values.size else 0.0,
            float(np.quantile(values, 0.05)) if values.size else 0.0,
            float(np.quantile(values, 0.5)) if values.size else 0.0,
            float(np.quantile(values, 0.95)) if values.size else 0.0,
        ],
        dtype=np.float32,
    )


def _components(
    snapshot: Snapshot,
    previous: Snapshot | None,
    horizon_steps: int,
    dt: float,
) -> dict[str, np.ndarray]:
    current = np.asarray(
        [
            snapshot.beta_current,
            previous.beta_current if previous is not None else snapshot.beta_current,
            snapshot.beta_current - previous.beta_current if previous is not None else 0.0,
            snapshot.n_nodes / 30.0,
            snapshot.radius / 1000.0,
        ],
        dtype=np.float32,
    )
    graph = np.asarray(
        [
            snapshot.stats[4] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[5] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[6] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[7],
            snapshot.stats[8],
            snapshot.stats[9] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[16],
        ],
        dtype=np.float32,
    )
    topology = np.concatenate([snapshot.pi, _summary(snapshot.pi)]).astype(np.float32)
    tau = float(horizon_steps) * float(dt)
    current_dist = pairwise_distances(snapshot.positions.astype(float))
    projected = snapshot.positions.astype(float) + snapshot.velocities.astype(float) * tau
    projected_dist = pairwise_distances(projected)
    tri = np.triu_indices(snapshot.n_nodes, k=1)
    radius = max(float(snapshot.radius), 1e-6)
    current_pairs = current_dist[tri] / radius
    projected_pairs = projected_dist[tri] / radius
    delta = projected_pairs - current_pairs
    projected_adj = adjacency_from_radius(projected_dist, radius)
    speeds = np.linalg.norm(snapshot.velocities.astype(float), axis=1) / 30.0
    kinematic = np.concatenate(
        [
            _summary(current_pairs),
            _summary(projected_pairs),
            _summary(delta),
            _summary(speeds),
            np.asarray(
                [
                    float(betti_zero(projected_adj)),
                    float(np.mean((current_pairs <= 1.0) & (projected_pairs > 1.0))),
                    float(np.mean((current_pairs > 1.0) & (projected_pairs <= 1.0))),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return {"current": current, "graph": graph, "topology": topology, "kinematic": kinematic}


def _component_matrices(
    snapshots: list[Snapshot],
    horizon_steps: int,
    dt: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[Snapshot]]:
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("current", *SOURCES)}
    targets: list[float] = []
    risks: list[int] = []
    aligned: list[Snapshot] = []
    grouped: dict[str, list[Snapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.run_id, []).append(snapshot)
    for run_id in sorted(grouped):
        previous = None
        for snapshot in sorted(grouped[run_id], key=lambda item: item.time_index):
            parts = _components(snapshot, previous, horizon_steps, dt)
            for name in rows:
                rows[name].append(parts[name])
            targets.append(float(snapshot.beta_target))
            risks.append(int(snapshot.frag_at_horizon))
            aligned.append(snapshot)
            previous = snapshot
    return (
        {name: np.asarray(values) for name, values in rows.items()},
        np.asarray(targets),
        np.asarray(risks),
        aligned,
    )


def _select_matrix(parts: dict[str, np.ndarray], selected_sources: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([parts["current"], *(parts[name] for name in selected_sources)], axis=1)


def _positive_class_probability(classifier: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    classes = np.asarray(classifier.classes_)
    positive = np.flatnonzero(classes == 1)
    if positive.size == 0:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, int(positive[0])]


def select_threshold_on_validation(
    labels: np.ndarray,
    scores: np.ndarray,
    snapshots: list[Snapshot],
    dt: float,
    horizon_steps: int,
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID,
) -> float:
    """Select on validation only: event F1, sample F1, false-event rate, threshold."""
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in threshold_grid:
        binary = (np.asarray(scores) >= float(threshold)).astype(int)
        sample = classification_metrics(np.asarray(labels, dtype=int), binary)
        event = alert_event_metrics(
            snapshots,
            scores,
            threshold=float(threshold),
            dt=dt,
            horizon_steps=horizon_steps,
        )
        ranked.append(
            (
                float(event["Alert_Event_F1"]),
                float(sample["Risk_F1"]),
                -float(event["False_Alert_Events_per_minute"]),
                float(threshold),
            )
        )
    return max(ranked)[3]


def _select_residual_scale(
    y_validation: np.ndarray,
    current_beta: np.ndarray,
    residual_prediction: np.ndarray,
) -> float:
    ranked = []
    for scale in RESIDUAL_SCALE_GRID:
        prediction = np.clip(current_beta + scale * residual_prediction, 1.0, None)
        ranked.append((float(mean_absolute_error(y_validation, prediction)), float(scale)))
    return min(ranked, key=lambda item: (item[0], item[1]))[1]


def _run_ids(snapshots: list[Snapshot]) -> set[str]:
    return {snapshot.run_id for snapshot in snapshots}


def assert_disjoint_run_ids(
    train: list[Snapshot],
    validation: list[Snapshot],
    test: list[Snapshot],
) -> dict[str, list[str]]:
    train_ids = _run_ids(train)
    validation_ids = _run_ids(validation)
    test_ids = _run_ids(test)
    intersections = {
        "train_validation": sorted(train_ids & validation_ids),
        "train_test": sorted(train_ids & test_ids),
        "validation_test": sorted(validation_ids & test_ids),
    }
    if any(intersections.values()):
        raise RuntimeError(f"run_id leakage detected: {intersections}")
    return intersections


def _run_seed(seed: int, sim: dict, combinations: list[tuple[str, ...]]) -> tuple[list[dict], dict]:
    seed_start = time.perf_counter()
    horizon = int(sim["forecast_horizon_steps"])
    dt = float(sim["dt"])
    snapshots = build_dataset(sim, seed=seed)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    intersections = assert_disjoint_run_ids(train, validation, test)
    train_parts, y_train, risk_train, _ = _component_matrices(train, horizon, dt)
    validation_parts, y_validation, risk_validation, validation_aligned = _component_matrices(
        validation, horizon, dt
    )
    test_parts, y_test, risk_test, test_aligned = _component_matrices(test, horizon, dt)

    rows: list[dict] = []
    for selected_sources in combinations:
        x_train = _select_matrix(train_parts, selected_sources)
        x_validation = _select_matrix(validation_parts, selected_sources)
        x_test = _select_matrix(test_parts, selected_sources)
        residual_train = y_train - x_train[:, 0]

        regressor = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=2,
            random_state=seed + 101,
            n_jobs=2,
        )
        classifier = ExtraTreesClassifier(
            n_estimators=128,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed + 211,
            n_jobs=2,
        )
        training_start = time.perf_counter()
        regressor.fit(x_train, residual_train)
        classifier.fit(x_train, risk_train)
        training_time = time.perf_counter() - training_start

        validation_residual = regressor.predict(x_validation)
        residual_scale = _select_residual_scale(
            y_validation,
            x_validation[:, 0],
            validation_residual,
        )
        validation_score = _positive_class_probability(classifier, x_validation)
        threshold = select_threshold_on_validation(
            risk_validation,
            validation_score,
            validation_aligned,
            dt,
            horizon,
        )

        inference_start = time.perf_counter()
        test_residual = regressor.predict(x_test)
        test_score = _positive_class_probability(classifier, x_test)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0 / max(len(x_test), 1)
        test_prediction = np.clip(x_test[:, 0] + residual_scale * test_residual, 1.0, None)
        binary_test = (test_score >= threshold).astype(int)
        sample_metrics = classification_metrics(risk_test, binary_test)
        probability_metrics = risk_probability_metrics(risk_test, test_score)
        event_metrics = alert_event_metrics(
            test_aligned,
            test_score,
            threshold=threshold,
            dt=dt,
            horizon_steps=horizon,
        )
        rows.append(
            {
                "seed": int(seed),
                "Model": model_name(selected_sources),
                "feature_sources": feature_label(selected_sources),
                "feature_count": int(x_train.shape[1]),
                "residual_scale": float(residual_scale),
                "selected_threshold": float(threshold),
                "MAE": float(mean_absolute_error(y_test, test_prediction)),
                "R2": float(r2_score(y_test, test_prediction)),
                "Risk_Precision": float(sample_metrics["Risk_Precision"]),
                "Risk_Recall": float(sample_metrics["Risk_Recall"]),
                "Risk_F1": float(sample_metrics["Risk_F1"]),
                "Alert_Event_Precision": float(event_metrics["Alert_Event_Precision"]),
                "Alert_Event_Recall": float(event_metrics["Alert_Event_Recall"]),
                "Alert_Event_F1": float(event_metrics["Alert_Event_F1"]),
                "False_Alert_Events_per_minute": float(
                    event_metrics["False_Alert_Events_per_minute"]
                ),
                "Risk_Brier": float(probability_metrics["Risk_Brier"]),
                "Risk_ECE": float(probability_metrics["Risk_ECE"]),
                "Training_Time_s": float(training_time),
                "Inference_Time_ms": float(inference_time_ms),
            }
        )

    split_protocol = {
        "split_seed": int(sim["split_seed"]),
        "stratify_by": list(sim.get("split_stratify_by", ["mobility"])),
        "run_ids": {
            "train": sorted(_run_ids(train)),
            "validation": sorted(_run_ids(validation)),
            "test": sorted(_run_ids(test)),
        },
        "run_id_intersections": intersections,
        "snapshot_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "runtime_seconds": float(time.perf_counter() - seed_start),
    }
    return rows, split_protocol


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_directory(output_dir: Path, seed: int) -> Path:
    return output_dir / "per_seed" / f"seed_{seed}"


def _expected_labels(combinations: list[tuple[str, ...]]) -> set[str]:
    return {feature_label(combination) for combination in combinations}


def is_seed_complete(seed_dir: Path, seed: int, combinations: list[tuple[str, ...]]) -> bool:
    csv_path = seed_dir / PER_SEED_FILENAME
    success_path = seed_dir / SUCCESS_FILENAME
    if not csv_path.is_file() or not success_path.is_file():
        return False
    try:
        frame = pd.read_csv(csv_path)
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(frame) == len(combinations)
        and set(frame.get("feature_sources", [])) == _expected_labels(combinations)
        and set(frame.get("seed", [])) == {int(seed)}
        and success.get("status") == "complete"
        and success.get("rows") == len(combinations)
        and success.get("csv_sha256") == sha256_file(csv_path)
    )


def _archive_partial_seed(seed_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = seed_dir.with_name(seed_dir.name + f"_incomplete_{stamp}")
    counter = 1
    while archive.exists():
        archive = seed_dir.with_name(seed_dir.name + f"_incomplete_{stamp}_{counter}")
        counter += 1
    seed_dir.rename(archive)
    return archive


def prepare_seed_schedule(
    seeds: list[int],
    output_dir: Path,
    combinations: list[tuple[str, ...]],
    resume: bool,
) -> tuple[list[int], list[int], list[str]]:
    pending: list[int] = []
    skipped: list[int] = []
    archived: list[str] = []
    for seed in seeds:
        seed_dir = _seed_directory(output_dir, seed)
        if is_seed_complete(seed_dir, seed, combinations):
            if not resume:
                raise FileExistsError(f"completed seed output already exists: {seed_dir}")
            skipped.append(seed)
            continue
        if seed_dir.exists():
            if not resume:
                raise FileExistsError(f"partial seed output already exists: {seed_dir}; use --resume")
            archived.append(relative_repo_path(_archive_partial_seed(seed_dir), ROOT))
        pending.append(seed)
    return pending, skipped, archived


def _persist_seed(
    seed: int,
    sim: dict,
    combinations: list[tuple[str, ...]],
    output_dir: Path,
) -> int:
    rows, split_protocol = _run_seed(seed, sim, combinations)
    seed_dir = _seed_directory(output_dir, seed)
    temporary = seed_dir.with_name(seed_dir.name + f"_writing_{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    csv_path = temporary / PER_SEED_FILENAME
    frame.to_csv(csv_path, index=False)
    _atomic_write_json(
        temporary / SEED_PROTOCOL_FILENAME,
        {
            "schema_version": 1,
            "seed": int(seed),
            "status": "complete",
            "feature_combinations": [feature_label(item) for item in combinations],
            "split": split_protocol,
        },
    )
    _atomic_write_json(
        temporary / SUCCESS_FILENAME,
        {
            "status": "complete",
            "seed": int(seed),
            "rows": len(frame),
            "csv_sha256": sha256_file(csv_path),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    temporary.replace(seed_dir)
    print(f"completed seed={seed}", flush=True)
    return seed


def execute_seeds(
    pending: list[int],
    sim: dict,
    combinations: list[tuple[str, ...]],
    output_dir: Path,
    workers: int,
    runner: Callable[[int, dict, list[tuple[str, ...]], Path], int] = _persist_seed,
) -> list[int]:
    if not pending:
        return []
    worker_count = max(1, min(int(workers), len(pending)))
    if worker_count == 1:
        return [runner(seed, sim, combinations, output_dir) for seed in pending]
    completed: list[int] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(runner, seed, sim, combinations, output_dir): seed for seed in pending
        }
        for future in as_completed(futures):
            completed.append(int(future.result()))
    return sorted(completed)


def collect_per_seed(
    output_dir: Path,
    seeds: list[int],
    combinations: list[tuple[str, ...]],
) -> pd.DataFrame:
    missing = [
        seed for seed in seeds if not is_seed_complete(_seed_directory(output_dir, seed), seed, combinations)
    ]
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete seeds: {missing}")
    frames = [
        pd.read_csv(_seed_directory(output_dir, seed) / PER_SEED_FILENAME) for seed in seeds
    ]
    combined = pd.concat(frames, ignore_index=True)
    expected_rows = len(seeds) * len(combinations)
    if len(combined) != expected_rows:
        raise RuntimeError(f"per-seed row count mismatch: {len(combined)} != {expected_rows}")
    return combined.sort_values(["seed", "feature_sources"]).reset_index(drop=True)


def summarise_results(
    frame: pd.DataFrame,
    bootstrap_rounds: int,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    unique_seeds = np.asarray(sorted(frame["seed"].astype(int).unique()), dtype=int)
    if unique_seeds.size == 0:
        raise ValueError("cannot summarise an empty per-seed table")
    rng = np.random.default_rng(bootstrap_seed)
    sampled_seed_indices = rng.integers(
        0,
        len(unique_seeds),
        size=(max(int(bootstrap_rounds), 1), len(unique_seeds)),
    )
    rows: list[dict] = []
    for (model, sources), group in frame.groupby(["Model", "feature_sources"], sort=False):
        group = group.set_index("seed").loc[unique_seeds]
        row: dict[str, object] = {
            "Model": model,
            "feature_sources": sources,
            "seeds": int(len(unique_seeds)),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            spread = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            bootstrap_means = values[sampled_seed_indices].mean(axis=1)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
            row[f"{metric}_seed_bootstrap_ci95_low"] = float(
                np.quantile(bootstrap_means, 0.025)
            )
            row[f"{metric}_seed_bootstrap_ci95_high"] = float(
                np.quantile(bootstrap_means, 0.975)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def _save_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    plot = summary.sort_values("MAE_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = plot["feature_sources"].replace({"current-only": CURRENT_ONLY_MODEL})
    axes[0].barh(labels, plot["MAE_mean"], color="#2f6db3")
    axes[1].barh(labels, plot["Alert_Event_F1_mean"], color="#3b7a57")
    axes[0].set_xlabel("MAE")
    axes[1].set_xlabel("Fragmentation-event F1")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "factorial_feature_ablation.png", dpi=220)
    fig.savefig(output_dir / "factorial_feature_ablation.pdf")
    plt.close(fig)


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "simpy",
        "torch",
        "torch-geometric",
    )
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _source_manifest(config_path: Path) -> list[dict]:
    paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "fanet" / "dataset.py",
        ROOT / "fanet" / "evaluation.py",
        ROOT / "fanet" / "training.py",
        ROOT / "fanet" / "reporting.py",
    ]
    return build_file_manifest(paths, ROOT)


def _base_protocol(
    config_path: Path,
    seeds: list[int],
    sim: dict,
    workers: int,
    bootstrap_rounds: int,
) -> dict:
    combinations = feature_combinations()
    return {
        "schema_version": 2,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative_repo_path(config_path, ROOT),
        "config_sha256": sha256_file(config_path),
        "seed_list": seeds,
        "seed_count": len(seeds),
        "split_seed": int(sim["split_seed"]),
        "split_stratify_by": list(sim.get("split_stratify_by", ["mobility"])),
        "feature_groups": FEATURE_GROUP_DEFINITIONS,
        "feature_combinations": [feature_label(item) for item in combinations],
        "threshold_grid": list(THRESHOLD_GRID),
        "residual_scale_grid": list(RESIDUAL_SCALE_GRID),
        "threshold_selection": {
            "partition": "validation only",
            "primary": "maximum fragmentation-event F1",
            "tie_break_order": [
                "maximum sample-level F1",
                "minimum false alert events per minute",
                "maximum threshold",
            ],
            "test_partition_role": "locked final evaluation only",
        },
        "learner_hyperparameters": LEARNER_HYPERPARAMETERS,
        "bootstrap": {
            "rounds": int(bootstrap_rounds),
            "cluster": "simulation seed",
            "random_seed": 20260714,
        },
        "parallel_workers": int(workers),
        "package_versions": _package_versions(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "git": _git_state(),
        "source_files": _source_manifest(config_path),
    }


def _validate_resume_protocol(protocol_path: Path, config_path: Path, seeds: list[int]) -> dict:
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    if existing.get("seed_list") != seeds:
        raise ValueError("resume seed list differs from the existing protocol")
    if existing.get("config_sha256") != sha256_file(config_path):
        raise ValueError("resume config hash differs from the existing protocol")
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 2^3 equal-learner feature-source factorial.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", help="Override training.seed_list")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-rounds", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    sim = raw["sim"]
    configured_seeds = [int(seed) for seed in raw["training"]["seed_list"]]
    seeds = [int(seed) for seed in (args.seeds if args.seeds is not None else configured_seeds)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed list must be non-empty and contain unique values")
    if args.seeds is None and len(seeds) != 20:
        raise ValueError(f"default submission configuration must contain 20 unique seeds, found {len(seeds)}")
    bootstrap_rounds = int(
        args.bootstrap_rounds
        if args.bootstrap_rounds is not None
        else raw.get("evaluation", {}).get("bootstrap_rounds", 1000)
    )
    if bootstrap_rounds < 1:
        raise ValueError("bootstrap rounds must be positive")

    combinations = feature_combinations()
    protocol_path = output_dir / "factorial_ablation_protocol.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --resume or a new path")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_seed").mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    if args.resume and protocol_path.is_file():
        protocol = _validate_resume_protocol(protocol_path, config_path, seeds)
        protocol["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol["status"] = "running"
    else:
        protocol = _base_protocol(config_path, seeds, sim, args.workers, bootstrap_rounds)
    pending, skipped, archived = prepare_seed_schedule(
        seeds, output_dir, combinations, resume=args.resume
    )
    protocol["resume"] = {
        "enabled": bool(args.resume),
        "skipped_complete_seeds": skipped,
        "pending_seeds_at_start": pending,
        "archived_partial_directories": archived,
    }
    _atomic_write_json(protocol_path, protocol)

    execute_seeds(pending, sim, combinations, output_dir, args.workers)
    per_seed = collect_per_seed(output_dir, seeds, combinations)
    summary = summarise_results(per_seed, bootstrap_rounds=bootstrap_rounds)
    per_seed.to_csv(output_dir / "factorial_ablation_per_seed.csv", index=False)
    summary.to_csv(output_dir / "factorial_ablation_summary.csv", index=False)
    _save_plot(summary, output_dir)

    protocol["status"] = "complete"
    protocol["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["runtime_seconds"] = float(time.perf_counter() - run_start)
    protocol["completed_seeds"] = seeds
    protocol["artifacts"] = build_file_manifest(
        [
            output_dir / "factorial_ablation_per_seed.csv",
            output_dir / "factorial_ablation_summary.csv",
            output_dir / "factorial_feature_ablation.pdf",
            output_dir / "factorial_feature_ablation.png",
        ],
        ROOT,
    )
    _atomic_write_json(protocol_path, protocol)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
