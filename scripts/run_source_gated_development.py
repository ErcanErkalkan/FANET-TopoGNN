from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file
from fanet.source_gated import (
    FEATURE_GROUPS,
    fit_current_state_extratrees,
    fit_source_gated_kinetic_topoguard,
)
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs" / "source_gated_development.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "source_gated_development"


def _split(config, snapshots):
    stratify = list(config.sim.get("split_stratify_by", ["mobility"]))
    for field in ("graph_policy", "radio_scenario"):
        if len({getattr(snapshot, field) for snapshot in snapshots}) > 1 and field not in stratify:
            stratify.append(field)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(config.sim["split_seed"]),
        stratify_by=tuple(stratify),
    )
    sets = [{snapshot.run_id for snapshot in part} for part in (train, validation, test)]
    intersections = {
        "train_validation": sorted(sets[0] & sets[1]),
        "train_test": sorted(sets[0] & sets[2]),
        "validation_test": sorted(sets[1] & sets[2]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"run_id leakage: {intersections}")
    return train, validation, test, stratify, intersections


def run_seed(seed: int, config_path: Path, output_dir: Path, bootstrap_rounds: int):
    started = time.perf_counter()
    config = load_config(config_path)
    snapshots = build_dataset(config.sim, seed=seed)
    train, validation, test, stratify, intersections = _split(config, snapshots)
    horizon = int(config.sim["forecast_horizon_steps"])
    dt = float(config.sim["dt"])
    models = [
        fit_current_state_persistence(),
        fit_current_state_extratrees(train, validation, horizon, dt, seed),
        fit_kinetic_topoguard(train, validation, horizon, dt, seed),
        fit_source_gated_kinetic_topoguard(
            train,
            validation,
            horizon,
            dt,
            seed,
            parameters=config.training["source_gated"],
        ),
    ]
    metric_rows = []
    source_gated = models[-1].model
    for result in models:
        predictions, scores, inference, aligned, threshold = predict_generic(result, test)
        metrics, _, _ = evaluate_predictions(
            result.model_name,
            aligned,
            predictions,
            scores,
            inference,
            dt=dt,
            bootstrap_rounds=bootstrap_rounds,
            horizon_steps=horizon,
            risk_threshold=threshold,
        )
        metrics.update(
            {
                "seed": int(seed),
                "evaluation_split": "test_locked_once",
                "selected_threshold": float(threshold),
            }
        )
        metric_rows.append(metrics)
    coefficients = []
    metadata = source_gated.artifact_metadata()
    for outcome, values in (
        ("count_residual", metadata["count_meta_coefficients"]),
        ("fragmentation_risk", metadata["risk_meta_coefficients"]),
    ):
        denominator = sum(abs(float(value)) for value in values.values())
        for source in FEATURE_GROUPS:
            coefficient = float(values[source])
            coefficients.append(
                {
                    "seed": int(seed),
                    "outcome": outcome,
                    "feature_group": source,
                    "coefficient": coefficient,
                    "coefficient_sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                    "absolute_coefficient_share": abs(coefficient) / denominator if denominator else 0.0,
                }
            )
    selected = metadata["selected_parameters"]
    calibration = []
    for row in source_gated.validation_candidates:
        record = {"seed": int(seed), "selection_split": "validation", **row}
        record["selected"] = bool(
            int(row["min_samples_leaf"]) == int(selected["risk_min_samples_leaf"])
            and float(row["logistic_c"]) == float(selected["logistic_c"])
            and str(row["calibration_type"]) == str(selected["calibration_type"])
        )
        calibration.append(record)
    seed_dir = output_dir / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(seed_dir / "metrics.csv", index=False)
    pd.DataFrame(coefficients).to_csv(seed_dir / "group_coefficients.csv", index=False)
    pd.DataFrame(calibration).to_csv(seed_dir / "calibration_metrics.csv", index=False)
    source_gated.save(seed_dir / "source_gated_model.pkl")
    (seed_dir / "source_gated_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    completion = {
        "seed": int(seed),
        "status": "complete",
        "runtime_seconds": time.perf_counter() - started,
        "split_seed": int(config.sim["split_seed"]),
        "split_stratify_by": stratify,
        "run_id_intersections": intersections,
        "test_evaluation_count_per_model": 1,
        "model_selection_uses_test": False,
        "selected_parameters": selected,
    }
    (seed_dir / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficients), pd.DataFrame(calibration)


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "MAE",
        "R2",
        "Risk_F1",
        "Risk_Brier",
        "Risk_ECE",
        "Alert_Event_F1",
        "Alert_Event_Precision",
        "Alert_Event_Recall",
        "False_Alert_Events_per_minute",
    ]
    rows = []
    for model, group in metrics.groupby("Model", sort=True):
        row = {"Model": model, "seed_count": int(group["seed"].nunique())}
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            spread = 1.96 * row[f"{column}_std"] / np.sqrt(max(len(values), 1))
            row[f"{column}_ci95_low"] = row[f"{column}_mean"] - spread
            row[f"{column}_ci95_high"] = row[f"{column}_mean"] + spread
        rows.append(row)
    return pd.DataFrame(rows)


def _versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy", "torch", "torch-geometric"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _git() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run leakage-free Source-Gated Kinetic-TopoGuard development.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    seeds = list(args.seeds if args.seeds is not None else config.training["seed_list"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if args.workers < 1 or args.bootstrap_rounds < 1:
        raise ValueError("workers and bootstrap rounds must be positive")
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or another directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metric_frames, coefficient_frames, calibration_frames = [], [], []
    pending = []
    for seed in seeds:
        seed_dir = output_dir / "per_seed" / f"seed_{seed}"
        paths = [seed_dir / name for name in ("metrics.csv", "group_coefficients.csv", "calibration_metrics.csv", "complete.json")]
        if args.resume and all(path.is_file() for path in paths):
            print(f"[resume] seed {seed}")
            metric_frames.append(pd.read_csv(paths[0]))
            coefficient_frames.append(pd.read_csv(paths[1]))
            calibration_frames.append(pd.read_csv(paths[2]))
        else:
            pending.append(seed)
    if args.workers == 1:
        for seed in pending:
            print(f"[run] seed {seed}")
            metric, coefficient, calibration = run_seed(seed, config_path, output_dir, args.bootstrap_rounds)
            metric_frames.append(metric)
            coefficient_frames.append(coefficient)
            calibration_frames.append(calibration)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_seed, seed, config_path, output_dir, args.bootstrap_rounds): seed
                for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                metric, coefficient, calibration = future.result()
                metric_frames.append(metric)
                coefficient_frames.append(coefficient)
                calibration_frames.append(calibration)
                print(f"[complete] seed {seed}")
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["seed", "Model"])
    coefficients = pd.concat(coefficient_frames, ignore_index=True).sort_values(["seed", "outcome", "feature_group"])
    calibration = pd.concat(calibration_frames, ignore_index=True).sort_values(["seed", "min_samples_leaf", "logistic_c", "calibration_type"])
    expected_models = {
        "Current-state persistence baseline",
        "Current-state ExtraTrees",
        "Kinetic-TopoGuard",
        "Source-Gated Kinetic-TopoGuard",
    }
    if set(metrics["Model"]) != expected_models or not (metrics.groupby("Model")["seed"].nunique() == len(seeds)).all():
        raise RuntimeError("model/seed coverage is incomplete")
    summary = _aggregate(metrics)
    metrics.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "group_coefficients.csv", index=False)
    calibration.to_csv(output_dir / "calibration_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    source_files = build_file_manifest(
        [config_path, ROOT / "configs" / "paper_like_submission.json", Path(__file__).resolve(), ROOT / "fanet" / "source_gated.py", ROOT / "fanet" / "dataset.py", ROOT / "fanet" / "evaluation.py"],
        ROOT,
    )
    protocol = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "config": relative_repo_path(config_path, ROOT),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config.raw,
        "seed_list": seeds,
        "split_seed": int(config.sim["split_seed"]),
        "feature_groups": list(FEATURE_GROUPS),
        "cross_fitting": {"group": "split_group_id", "predictions": ["count residual", "fragmentation risk"], "test_role": "locked evaluation once per model"},
        "selection": {"base_and_meta_hyperparameters": "validation only", "calibration": "validation only", "threshold": "validation event F1 with fixed tie-break"},
        "package_versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "parallel_workers": args.workers,
        "bootstrap_rounds_per_test_evaluation": args.bootstrap_rounds,
        "git": _git(),
        "source_files": source_files,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    sg = summary[summary["Model"] == "Source-Gated Kinetic-TopoGuard"].iloc[0]
    original = summary[summary["Model"] == "Kinetic-TopoGuard"].iloc[0]
    decision = {
        "development_only": True,
        "test_used_for_model_selection": False,
        "source_gated_mae_mean": float(sg["MAE_mean"]),
        "original_mae_mean": float(original["MAE_mean"]),
        "source_gated_event_f1_mean": float(sg["Alert_Event_F1_mean"]),
        "original_event_f1_mean": float(original["Alert_Event_F1_mean"]),
        "superiority_claimed": False,
        "interpretation": "Development results are reported descriptively; no architecture or hyperparameter was changed after test evaluation and no superiority claim is made.",
    }
    (output_dir / "development_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
