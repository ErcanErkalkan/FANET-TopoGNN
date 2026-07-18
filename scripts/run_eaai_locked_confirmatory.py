from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
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
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file, verify_manifest
from fanet.seed_audit import CONFIRMATORY_DOMAIN, deterministic_confirmatory_seeds, discover_prior_seeds
from fanet.source_gated import fit_current_state_extratrees, fit_source_gated_kinetic_topoguard
from fanet.training import (
    fit_current_state_persistence,
    fit_kinetic_topoguard,
    select_best_shallow,
)


CONFIG_PATH = ROOT / "configs" / "eaai_locked_confirmatory.json"
LOCK_PATH = ROOT / "configs" / "eaai_locked_confirmatory.lock.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "eaai_locked_confirmatory"
DRY_OUTPUT = ROOT / "outputs" / "eaai_locked_confirmatory_dry_run"
DRY_RUN_SEED = 7
PRIMARY_MODEL = "Source-Gated Kinetic-TopoGuard"
PRIMARY_COMPARATOR = "Current-state ExtraTrees"
METRICS = {
    "Alert_Event_F1": True,
    "Alert_Event_Precision": True,
    "Alert_Event_Recall": True,
    "False_Alert_Events_per_minute": False,
    "MAE": False,
    "R2": True,
    "Risk_Brier": False,
    "Risk_ECE": False,
    "Inference_ms": False,
}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _selection_sources() -> list[Path]:
    return [
        ROOT / "configs" / "source_gated_development.json",
        ROOT / "outputs" / "source_gated_development" / "calibration_metrics.csv",
        *sorted((ROOT / "outputs" / "source_gated_development" / "per_seed").glob("seed_*/source_gated_model_metadata.json")),
    ]


def create_lock() -> dict:
    if LOCK_PATH.exists():
        raise FileExistsError(f"lock already exists and will not be overwritten: {LOCK_PATH}")
    config = load_config(CONFIG_PATH)
    excluded_paths = {
        relative_repo_path(CONFIG_PATH, ROOT),
        relative_repo_path(LOCK_PATH, ROOT),
        "outputs/eaai_locked_confirmatory",
        "outputs/eaai_locked_confirmatory_dry_run",
    }
    excluded, records = discover_prior_seeds(ROOT, excluded_paths=excluded_paths)
    selected = deterministic_confirmatory_seeds(excluded, count=20)
    configured = [int(value) for value in config.training["seed_list"]]
    if configured != selected:
        raise RuntimeError(
            f"configured seed list does not match deterministic selection; expected {selected}, got {configured}"
        )
    if set(configured) & excluded:
        raise RuntimeError("confirmatory seeds overlap prior seed inventory")
    selection_manifest = build_file_manifest(_selection_sources(), ROOT)
    code_manifest = build_file_manifest(
        [
            ROOT / "fanet" / "source_gated.py",
            ROOT / "fanet" / "seed_audit.py",
            Path(__file__).resolve(),
            ROOT / "configs" / "paper_like_submission.json",
        ],
        ROOT,
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative_repo_path(CONFIG_PATH, ROOT),
        "config_sha256": sha256_file(CONFIG_PATH),
        "resolved_config_sha256": _canonical_hash(config.raw),
        "git": _git_state(),
        "excluded_prior_seeds": sorted(int(value) for value in excluded),
        "excluded_seed_source_count": len(records),
        "seed_selection": {
            "domain": CONFIRMATORY_DOMAIN,
            "method": "SHA-256 counter stream mapped to positive 31-bit-range integers; prior and duplicate seeds discarded",
            "selected_seeds": configured,
        },
        "model_selection_source_artifacts": selection_manifest,
        "locked_code_and_parent_config": code_manifest,
        "frozen_primary_model": PRIMARY_MODEL,
        "primary_comparator": PRIMARY_COMPARATOR,
        "primary_metric": "Alert_Event_F1",
        "primary_hypothesis": config.raw["confirmatory"]["hypothesis"],
        "primary_decision_rule": config.raw["confirmatory"]["primary_decision_rule"],
        "frozen_package_derivation": {
            "source": "modal validation-selected values across five development seeds; development test metrics were not used",
            "count_min_samples_leaf": 2,
            "risk_min_samples_leaf": 5,
            "ridge_alpha": 1.0,
            "logistic_c": 0.1,
            "calibration_type": "isotonic",
            "threshold": "selected per seed using the frozen validation-only rule and grid",
        },
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_lock() -> dict[str, object]:
    errors = []
    if not LOCK_PATH.is_file():
        return {"valid": False, "errors": [f"missing lock: {LOCK_PATH}"]}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config(CONFIG_PATH)
    actual_config_hash = sha256_file(CONFIG_PATH)
    actual_resolved_hash = _canonical_hash(config.raw)
    if actual_config_hash != lock.get("config_sha256"):
        errors.append(f"config SHA-256 mismatch: expected {lock.get('config_sha256')}, got {actual_config_hash}")
    if actual_resolved_hash != lock.get("resolved_config_sha256"):
        errors.append(
            f"resolved config SHA-256 mismatch: expected {lock.get('resolved_config_sha256')}, got {actual_resolved_hash}"
        )
    selected = [int(value) for value in config.training["seed_list"]]
    if selected != lock.get("seed_selection", {}).get("selected_seeds"):
        errors.append("configured seed order differs from locked selected seeds")
    overlap = sorted(set(selected) & set(lock.get("excluded_prior_seeds", [])))
    if overlap:
        errors.append(f"locked seeds overlap excluded prior seeds: {overlap}")
    for key in ("model_selection_source_artifacts", "locked_code_and_parent_config"):
        verification = verify_manifest(lock.get(key, []), ROOT)
        errors.extend(f"{key}: {error}" for error in verification["errors"])
    return {
        "valid": not errors,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": actual_config_hash,
        "resolved_config_sha256": actual_resolved_hash,
        "lock_sha256": sha256_file(LOCK_PATH),
        "errors": errors,
    }


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
    run_sets = [{snapshot.run_id for snapshot in part} for part in (train, validation, test)]
    intersections = {
        "train_validation": sorted(run_sets[0] & run_sets[1]),
        "train_test": sorted(run_sets[0] & run_sets[2]),
        "validation_test": sorted(run_sets[1] & run_sets[2]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"run_id leakage: {intersections}")
    return train, validation, test, stratify, intersections


def run_seed(seed: int, output_dir: Path, bootstrap_rounds: int, lock_sha256: str):
    started = time.perf_counter()
    config = load_config(CONFIG_PATH)
    snapshots = build_dataset(config.sim, seed=seed)
    train, validation, test, stratify, intersections = _split(config, snapshots)
    horizon = int(config.sim["forecast_horizon_steps"])
    dt = float(config.sim["dt"])
    shallow = select_best_shallow(train, validation)
    selected_shallow_learner = shallow.model_name
    shallow.model_name = "Shallow ML"
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
        shallow,
    ]
    rows = []
    source_gated = next(result.model for result in models if result.model_name == PRIMARY_MODEL)
    for result in models:
        prediction, risk, latency, aligned, threshold = predict_generic(result, test)
        metrics, _, _ = evaluate_predictions(
            result.model_name,
            aligned,
            prediction,
            risk,
            latency,
            dt=dt,
            bootstrap_rounds=bootstrap_rounds,
            horizon_steps=horizon,
            risk_threshold=threshold,
        )
        rows.append(
            {
                "seed": int(seed),
                "Model": result.model_name,
                "evaluation_split": "test_locked_once",
                "selected_threshold": float(threshold),
                "selected_base_learner": (
                    selected_shallow_learner if result.model_name == "Shallow ML" else result.model_name
                ),
                **{metric: metrics[metric] for metric in METRICS},
            }
        )
    seed_dir = output_dir / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(seed_dir / "metrics.csv", index=False)
    source_gated.save(seed_dir / "source_gated_model.pkl")
    metadata = source_gated.artifact_metadata()
    (seed_dir / "source_gated_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    completion = {
        "seed": int(seed),
        "status": "complete",
        "runtime_seconds": time.perf_counter() - started,
        "config_sha256": sha256_file(CONFIG_PATH),
        "lock_sha256": lock_sha256,
        "split_seed": int(config.sim["split_seed"]),
        "split_stratify_by": stratify,
        "run_id_intersections": intersections,
        "shared_realization_across_models": True,
        "test_evaluation_count_per_model": 1,
        "model_selection_uses_test": False,
        "source_gated_selected_parameters": metadata["selected_parameters"],
    }
    (seed_dir / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in metrics.groupby("Model", sort=True):
        row = {"Model": model, "seed_count": int(group["seed"].nunique())}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            spread = 1.96 * std / np.sqrt(max(len(values), 1))
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_ci95_low": mean - spread,
                    f"{metric}_ci95_high": mean + spread,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, rounds: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(rounds, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _permutation_pvalue(values: np.ndarray, rounds: int, seed: int) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(values)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(rounds, len(values)))
    permuted = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.sum(permuted >= observed)) / (rounds + 1))


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_tests(metrics: pd.DataFrame, bootstrap_rounds: int = 20_000, permutation_rounds: int = 100_000) -> pd.DataFrame:
    candidate = metrics[metrics["Model"] == PRIMARY_MODEL].sort_values("seed")
    reference = metrics[metrics["Model"] == PRIMARY_COMPARATOR].sort_values("seed")
    if candidate["seed"].duplicated().any() or reference["seed"].duplicated().any():
        raise ValueError("duplicate seed in primary comparison")
    if candidate["seed"].tolist() != reference["seed"].tolist():
        raise ValueError("primary comparison seed pairing mismatch")
    rows = []
    for index, (metric, higher_is_better) in enumerate(METRICS.items()):
        raw = candidate[metric].to_numpy(dtype=float) - reference[metric].to_numpy(dtype=float)
        benefit = raw if higher_is_better else -raw
        low, high = _bootstrap_ci(raw, bootstrap_rounds, 20260715 + index)
        benefit_low, benefit_high = _bootstrap_ci(benefit, bootstrap_rounds, 20261715 + index)
        try:
            wilcoxon_p = 1.0 if np.allclose(raw, 0.0) else float(wilcoxon(raw, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "candidate": PRIMARY_MODEL,
                "reference": PRIMARY_COMPARATOR,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "seed_pairs": len(raw),
                "paired_mean_difference": float(np.mean(raw)),
                "paired_median_difference": float(np.median(raw)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "benefit_mean": float(np.mean(benefit)),
                "benefit_bootstrap_ci95_low": benefit_low,
                "benefit_bootstrap_ci95_high": benefit_high,
                "wilcoxon_pvalue": wilcoxon_p,
                "paired_permutation_pvalue": _permutation_pvalue(raw, permutation_rounds, 20262715 + index),
            }
        )
    frame = pd.DataFrame(rows)
    frame["paired_permutation_holm_pvalue"] = _holm(frame["paired_permutation_pvalue"].tolist())
    return frame


def _versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy", "torch", "torch-geometric"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _protocol(output_dir: Path, seeds: list[int], status: str, started: float, workers: int, dry_run: bool) -> dict:
    config = load_config(CONFIG_PATH)
    return {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "dry_run": dry_run,
        "config": relative_repo_path(CONFIG_PATH, ROOT),
        "config_sha256": sha256_file(CONFIG_PATH),
        "lock": relative_repo_path(LOCK_PATH, ROOT),
        "lock_sha256": sha256_file(LOCK_PATH),
        "hypothesis_registered_before_test": config.raw["confirmatory"]["hypothesis"],
        "primary_metric": "Alert_Event_F1",
        "primary_model": PRIMARY_MODEL,
        "primary_comparator": PRIMARY_COMPARATOR,
        "evaluation_models": config.raw["confirmatory"]["evaluation_models"],
        "seed_list": seeds,
        "shared_simulation_realization_within_seed": True,
        "test_role": "locked evaluation exactly once per seed/model",
        "selection_partitions": {"base/meta": "training OOF", "hyperparameters": "frozen", "calibration": "frozen isotonic", "threshold": "validation only"},
        "parallel_workers": workers,
        "package_versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "git": _git_state(),
        "output_dir": relative_repo_path(output_dir, ROOT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hash-locked EAAI confirmatory experiment.")
    parser.add_argument("--create-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.create_lock:
        lock = create_lock()
        print(json.dumps({"lock": relative_repo_path(LOCK_PATH, ROOT), "config_sha256": lock["config_sha256"], "selected_seeds": lock["seed_selection"]["selected_seeds"]}, indent=2))
        return 0
    verification = verify_lock()
    if not verification["valid"]:
        raise RuntimeError("lock verification failed:\n" + "\n".join(verification["errors"]))
    config = load_config(CONFIG_PATH)
    full_seeds = [int(value) for value in config.training["seed_list"]]
    seeds = [DRY_RUN_SEED] if args.dry_run else full_seeds
    output_dir = (args.output_dir or (DRY_OUTPUT if args.dry_run else DEFAULT_OUTPUT)).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or a new output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lock_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    started = time.perf_counter()
    running_protocol = _protocol(output_dir, seeds, "running", started, args.workers, args.dry_run)
    (output_dir / "protocol.json").write_text(json.dumps(running_protocol, indent=2), encoding="utf-8")
    lock_sha = str(verification["lock_sha256"])
    frames = []
    pending = []
    for seed in seeds:
        seed_dir = output_dir / "per_seed" / f"seed_{seed}"
        metrics_path = seed_dir / "metrics.csv"
        complete_path = seed_dir / "complete.json"
        if args.resume and metrics_path.is_file() and complete_path.is_file():
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if completion.get("config_sha256") != verification["config_sha256"] or completion.get("lock_sha256") != lock_sha:
                raise RuntimeError(f"seed {seed} completion marker belongs to another config/lock")
            frames.append(pd.read_csv(metrics_path))
            print(f"[resume] seed {seed}")
        else:
            pending.append(seed)
    if args.workers == 1:
        for seed in pending:
            print(f"[run] seed {seed}")
            frames.append(run_seed(seed, output_dir, args.bootstrap_rounds, lock_sha))
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_seed, seed, output_dir, args.bootstrap_rounds, lock_sha): seed
                for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                frames.append(future.result())
                print(f"[complete] seed {seed}")
    metrics = pd.concat(frames, ignore_index=True).sort_values(["seed", "Model"])
    expected_models = set(config.raw["confirmatory"]["evaluation_models"])
    if set(metrics["Model"]) != expected_models or not (metrics.groupby("Model")["seed"].nunique() == len(seeds)).all():
        raise RuntimeError("model/seed coverage is incomplete")
    aggregate = _aggregate(metrics)
    paired = _paired_tests(metrics)
    metrics.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_tests.csv", index=False)
    primary = paired[paired["metric"] == "Alert_Event_F1"].iloc[0]
    supported = bool(
        primary["benefit_mean"] > 0.0
        and primary["benefit_bootstrap_ci95_low"] > 0.0
        and primary["paired_permutation_pvalue"] < 0.05
    )
    decision = {
        "dry_run": args.dry_run,
        "full_confirmatory_complete": bool(not args.dry_run and len(seeds) == 20),
        "primary_hypothesis": config.raw["confirmatory"]["hypothesis"],
        "primary_metric": "Alert_Event_F1",
        "seed_pairs": int(primary["seed_pairs"]),
        "paired_mean_difference": float(primary["paired_mean_difference"]),
        "bootstrap_ci95": [float(primary["bootstrap_ci95_low"]), float(primary["bootstrap_ci95_high"])],
        "paired_permutation_pvalue": float(primary["paired_permutation_pvalue"]),
        "statistically_supported_superiority": supported,
        "decision": "statistically_supported_superiority" if supported else "no_supported_superiority",
        "model_or_hyperparameter_changes_after_lock": False,
    }
    (output_dir / "confirmatory_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    complete_protocol = _protocol(output_dir, seeds, "complete", started, args.workers, args.dry_run)
    complete_protocol["artifacts"] = build_file_manifest(
        [
            output_dir / "per_seed_metrics.csv",
            output_dir / "aggregate_metrics.csv",
            output_dir / "paired_tests.csv",
            output_dir / "lock_verification.json",
            output_dir / "confirmatory_decision.json",
        ],
        ROOT,
    )
    (output_dir / "protocol.json").write_text(json.dumps(complete_protocol, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
