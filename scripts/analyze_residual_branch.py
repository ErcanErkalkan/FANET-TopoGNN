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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file
from fanet.training import RESIDUAL_ALPHA_GRID, fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs" / "paper_like_submission.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "residual_branch_audit"
MODEL_NAME = "Kinetic-TopoGuard"


def _mae(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float))))


def make_diagnostic_row(
    *,
    seed: int,
    diagnostics: dict[str, object],
    y_test: np.ndarray,
    persistence_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    scope_type: str = "overall",
    scope_value: str = "all",
) -> dict[str, object]:
    test_mae_zero = _mae(y_test, persistence_prediction)
    test_mae_selected = _mae(y_test, selected_prediction)
    percentage = (
        100.0 * (test_mae_zero - test_mae_selected) / test_mae_zero
        if test_mae_zero > 0.0
        else (0.0 if test_mae_selected == 0.0 else float("nan"))
    )
    return {
        "seed": int(seed),
        "Model": MODEL_NAME,
        "scope_type": scope_type,
        "scope_value": scope_value,
        "selected_alpha": float(diagnostics["selected_alpha"]),
        "validation_mae_alpha_0": float(diagnostics["validation_mae_alpha_0"]),
        "validation_mae_selected_alpha": float(diagnostics["validation_mae_selected_alpha"]),
        "test_mae_alpha_0": test_mae_zero,
        "test_mae_selected_alpha": test_mae_selected,
        "paired_mae_difference_selected_minus_persistence": test_mae_selected - test_mae_zero,
        "percentage_improvement": float(percentage),
        "alpha_zero": bool(diagnostics["alpha_zero"]),
        "selection_split": str(diagnostics.get("selection_split", "validation")),
        "selection_metric": str(diagnostics.get("selection_metric", "MAE")),
    }


def _assert_disjoint(train: list, validation: list, test: list) -> dict[str, list[str]]:
    groups = [{snapshot.run_id for snapshot in split} for split in (train, validation, test)]
    intersections = {
        "train_validation": sorted(groups[0] & groups[1]),
        "train_test": sorted(groups[0] & groups[2]),
        "validation_test": sorted(groups[1] & groups[2]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"run_id leakage detected: {intersections}")
    return intersections


def run_seed(seed: int, config_path: Path, output_dir: Path) -> pd.DataFrame:
    config = load_config(config_path)
    start = time.perf_counter()
    snapshots = build_dataset(config.sim, seed=seed)
    split_stratify_by = list(config.sim.get("split_stratify_by", ["mobility"]))
    for field in ("graph_policy", "radio_scenario"):
        if len({getattr(snapshot, field) for snapshot in snapshots}) > 1 and field not in split_stratify_by:
            split_stratify_by.append(field)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(config.sim.get("split_seed", seed)),
        stratify_by=tuple(split_stratify_by),
    )
    intersections = _assert_disjoint(train, validation, test)
    result = fit_kinetic_topoguard(
        train,
        validation,
        horizon_steps=int(config.sim["forecast_horizon_steps"]),
        dt=float(config.sim["dt"]),
        seed=seed,
    )
    selected_prediction, _, aligned_test = result.model.predict_snapshots(test)
    diagnostics = dict(result.model.residual_diagnostics)
    y_test = np.asarray([snapshot.beta_target for snapshot in aligned_test], dtype=float)
    persistence = np.clip(
        np.asarray([snapshot.beta_current for snapshot in aligned_test], dtype=float), 1.0, None
    )
    rows = [
        make_diagnostic_row(
            seed=seed,
            diagnostics=diagnostics,
            y_test=y_test,
            persistence_prediction=persistence,
            selected_prediction=selected_prediction,
        )
    ]
    scenario_fields = ("mobility", "graph_policy", "radio_scenario")
    for field in scenario_fields:
        labels = np.asarray([str(getattr(snapshot, field)) for snapshot in aligned_test])
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            rows.append(
                make_diagnostic_row(
                    seed=seed,
                    diagnostics=diagnostics,
                    y_test=y_test[mask],
                    persistence_prediction=persistence[mask],
                    selected_prediction=np.asarray(selected_prediction)[mask],
                    scope_type=field,
                    scope_value=label,
                )
            )
    frame = pd.DataFrame(rows)
    seed_dir = output_dir / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(seed_dir / "residual_diagnostics.csv", index=False)
    metadata = {
        "seed": int(seed),
        "status": "complete",
        "runtime_seconds": time.perf_counter() - start,
        "split_seed": int(config.sim.get("split_seed", seed)),
        "split_stratify_by": split_stratify_by,
        "run_id_intersections": intersections,
        "selected_regressor_candidate_index": int(diagnostics["candidate_index"]),
        "alpha_grid": diagnostics["alpha_grid"],
        "selection_split": diagnostics["selection_split"],
        "selection_metric": diagnostics["selection_metric"],
    }
    (seed_dir / "complete.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def _bootstrap_mean_ci(values: np.ndarray, rounds: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(rounds, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def summarize_diagnostics(
    per_seed: pd.DataFrame,
    bootstrap_rounds: int = 20_000,
    random_seed: int = 20260714,
) -> pd.DataFrame:
    rows = []
    for offset, ((scope_type, scope_value), group) in enumerate(
        per_seed.groupby(["scope_type", "scope_value"], sort=True)
    ):
        if group["seed"].duplicated().any():
            raise ValueError(f"duplicate seed in {scope_type}={scope_value}")
        differences = group["paired_mae_difference_selected_minus_persistence"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_ci(differences, bootstrap_rounds, random_seed + offset)
        if np.allclose(differences, 0.0):
            pvalue = 1.0
        else:
            try:
                pvalue = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                pvalue = 1.0
        supported = bool(len(group) >= 2 and float(np.mean(differences)) < 0.0 and high < 0.0 and pvalue < 0.05)
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "seed_count": int(group["seed"].nunique()),
                "alpha_zero_seed_count": int(group["alpha_zero"].sum()),
                "alpha_zero_seed_ratio": float(group["alpha_zero"].mean()),
                "selected_alpha_mean": float(group["selected_alpha"].mean()),
                "validation_mae_alpha_0_mean": float(group["validation_mae_alpha_0"].mean()),
                "validation_mae_selected_alpha_mean": float(group["validation_mae_selected_alpha"].mean()),
                "test_mae_alpha_0_mean": float(group["test_mae_alpha_0"].mean()),
                "test_mae_selected_alpha_mean": float(group["test_mae_selected_alpha"].mean()),
                "paired_mae_difference_mean": float(np.mean(differences)),
                "paired_mae_difference_median": float(np.median(differences)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wilcoxon_pvalue": pvalue,
                "improved_seed_count": int(np.sum(differences < 0.0)),
                "unchanged_seed_count": int(np.sum(np.isclose(differences, 0.0))),
                "worsened_seed_count": int(np.sum(differences > 0.0)),
                "count_branch_supported": supported,
            }
        )
    return pd.DataFrame(rows)


def _plot_alpha(per_seed: pd.DataFrame, output_path: Path) -> None:
    overall = per_seed[per_seed["scope_type"] == "overall"].sort_values("seed")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(overall["seed"].astype(str), overall["selected_alpha"], color="#2f6db3")
    ax.set_xlabel("Simulation seed")
    ax.set_ylabel("Validation-selected residual alpha")
    ax.set_ylim(0.0, max(1.0, float(overall["selected_alpha"].max()) * 1.1))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy", "torch", "torch-geometric")
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _write_protocol(
    config_path: Path,
    output_dir: Path,
    seeds: list[int],
    started: float,
    status: str,
    bootstrap_rounds: int,
    workers: int,
) -> None:
    sources = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "fanet" / "training.py",
        ROOT / "fanet" / "pipeline.py",
        ROOT / "fanet" / "dataset.py",
    ]
    payload = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "config": relative_repo_path(config_path, ROOT),
        "config_sha256": sha256_file(config_path),
        "seed_list": seeds,
        "seed_count": len(seeds),
        "model": MODEL_NAME,
        "split": {"selection": "validation", "test_role": "locked final evaluation only"},
        "alpha_grid": list(RESIDUAL_ALPHA_GRID),
        "selection_metric": "validation MAE",
        "selection_minimum_improvement": 1e-5,
        "regressor_candidates": [
            "ExtraTreesRegressor(n_estimators=64,min_samples_leaf=2)",
            "GradientBoostingRegressor(n_estimators=96,learning_rate=0.05,max_depth=3)",
            "RandomForestRegressor(n_estimators=64,min_samples_leaf=2)",
        ],
        "bootstrap": {"rounds": int(bootstrap_rounds), "cluster": "simulation seed", "random_seed": 20260714},
        "parallel_workers": int(workers),
        "package_versions": _package_versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "git": _git_state(),
        "source_files": build_file_manifest(sources, ROOT),
    }
    (output_dir / "residual_branch_protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the Kinetic-TopoGuard residual component-count branch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-rounds", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    seeds = list(args.seeds if args.seeds is not None else config.training["seed_list"])
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seed list must be non-empty and unique")
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists; use --resume or a new path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.workers < 1 or args.bootstrap_rounds < 1:
        raise ValueError("workers and bootstrap rounds must be positive")
    frames = []
    pending = []
    for seed in seeds:
        seed_file = output_dir / "per_seed" / f"seed_{seed}" / "residual_diagnostics.csv"
        complete_file = seed_file.with_name("complete.json")
        if args.resume and seed_file.is_file() and complete_file.is_file():
            print(f"[resume] seed {seed}")
            frames.append(pd.read_csv(seed_file))
        else:
            pending.append(seed)
    if args.workers == 1:
        for seed in pending:
            print(f"[run] seed {seed}")
            frames.append(run_seed(seed, config_path, output_dir))
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_seed, seed, config_path, output_dir): seed for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                frames.append(future.result())
                print(f"[complete] seed {seed}")
    per_seed = pd.concat(frames, ignore_index=True)
    overall_seeds = per_seed.loc[per_seed["scope_type"] == "overall", "seed"]
    if sorted(overall_seeds.astype(int).tolist()) != sorted(seeds):
        raise RuntimeError("aggregate output does not contain exactly one overall row per requested seed")
    summary = summarize_diagnostics(per_seed, args.bootstrap_rounds)
    per_seed.to_csv(output_dir / "per_seed.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _plot_alpha(per_seed, output_dir / "alpha_distribution.pdf")
    overall = summary[(summary["scope_type"] == "overall") & (summary["scope_value"] == "all")].iloc[0]
    supported_scenarios = summary[
        (summary["scope_type"] != "overall") & summary["count_branch_supported"]
    ][["scope_type", "scope_value"]].to_dict("records")
    decision = {
        "count_branch_supported": bool(overall["count_branch_supported"]),
        "decision_rule": "supported only when mean selected-minus-persistence MAE is negative, its seed-bootstrap 95% CI excludes zero, and two-sided Wilcoxon p < 0.05",
        "seed_count": int(overall["seed_count"]),
        "alpha_zero_seed_ratio": float(overall["alpha_zero_seed_ratio"]),
        "paired_mae_difference_mean": float(overall["paired_mae_difference_mean"]),
        "bootstrap_ci95": [float(overall["bootstrap_ci95_low"]), float(overall["bootstrap_ci95_high"])],
        "wilcoxon_pvalue": float(overall["wilcoxon_pvalue"]),
        "improved_seed_count": int(overall["improved_seed_count"]),
        "unchanged_seed_count": int(overall["unchanged_seed_count"]),
        "worsened_seed_count": int(overall["worsened_seed_count"]),
        "scenario_stratified_supported_subgroups": supported_scenarios,
        "interpretation": (
            "The residual count branch is supported over persistence across seeds."
            if bool(overall["count_branch_supported"])
            else "No statistically supported overall MAE benefit over persistence was found; this is not an equivalence claim."
        ),
        "source_csv": relative_repo_path(output_dir / "per_seed.csv", ROOT),
    }
    (output_dir / "residual_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _write_protocol(
        config_path,
        output_dir,
        seeds,
        started,
        "complete",
        args.bootstrap_rounds,
        args.workers,
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
