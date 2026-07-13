from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import platform
import re
import time
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dataset import build_dataset, to_frame, train_val_test_split
from .evaluation import (
    benjamini_hochberg,
    evaluate_predictions,
    paired_statistics,
    predict_generic,
    run_network_controller,
    save_summary_json,
    summarise_leads,
    threshold_sensitivity_rows,
)
from .manuscript import export_manuscript_summary, export_manuscript_tables
from .reporting import (
    plot_ablation_summary,
    plot_dataset_overview,
    plot_latency,
    plot_lead_cdf,
    plot_lead_time_summary,
    plot_metric_bars,
    plot_network_extended_metrics,
    plot_network_metrics,
    plot_publication_performance,
    plot_radio_policy_sensitivity,
    plot_residual_box,
    plot_scatter,
    plot_statistical_summary,
    save_table,
    write_artifact_manifest,
    write_claims_summary,
    write_markdown_report,
)
from .training import TORCH_AVAILABLE, fit_current_state_persistence, fit_heuristic, fit_kinetic_topoguard, select_best_shallow, train_torch_model


NUMERIC_METRICS = [
    "MAE",
    "MSE",
    "R2",
    "Inference_ms",
    "Risk_Precision",
    "Risk_Recall",
    "Risk_F1",
    "Risk_Specificity",
    "Risk_Accuracy",
    "Risk_PR_AUC",
    "Risk_ROC_AUC",
    "Risk_Brier",
    "Risk_ECE",
    "False_Alarms",
    "False_Alarms_per_minute",
    "Alert_Events",
    "True_Alert_Events",
    "False_Alert_Events",
    "Ground_Truth_Fragmentation_Events",
    "Missed_Fragmentation_Events",
    "Alert_Event_Precision",
    "Alert_Event_Recall",
    "Alert_Event_F1",
    "False_Alert_Events_per_minute",
]

METRIC_BOUNDS = {
    "MAE": (0.0, None),
    "MSE": (0.0, None),
    "R2": (None, 1.0),
    "Inference_ms": (0.0, None),
    "Risk_Precision": (0.0, 1.0),
    "Risk_Recall": (0.0, 1.0),
    "Risk_F1": (0.0, 1.0),
    "Risk_Specificity": (0.0, 1.0),
    "Risk_Accuracy": (0.0, 1.0),
    "Risk_PR_AUC": (0.0, 1.0),
    "Risk_ROC_AUC": (0.0, 1.0),
    "Risk_Brier": (0.0, 1.0),
    "Risk_ECE": (0.0, 1.0),
    "False_Alarms": (0.0, None),
    "False_Alarms_per_minute": (0.0, None),
    "Alert_Events": (0.0, None),
    "True_Alert_Events": (0.0, None),
    "False_Alert_Events": (0.0, None),
    "Ground_Truth_Fragmentation_Events": (0.0, None),
    "Missed_Fragmentation_Events": (0.0, None),
    "Alert_Event_Precision": (0.0, 1.0),
    "Alert_Event_Recall": (0.0, 1.0),
    "Alert_Event_F1": (0.0, 1.0),
    "False_Alert_Events_per_minute": (0.0, None),
    "Lead_5th_ms": (0.0, None),
    "Lead_median_ms": (0.0, None),
    "Lead_IQR_ms": (0.0, None),
    "Lead_95th_ms": (0.0, None),
    "Lead_mean_ms": (0.0, None),
    "Lead_norm_mean": (0.0, None),
    "Connectivity ratio": (0.0, 1.0),
    "Reachability-delivery proxy (%)": (0.0, 100.0),
    "Proxy delay (ms)": (0.0, None),
    "Proactive reroute (%)": (0.0, 100.0),
    "DTN buffered (%)": (0.0, 100.0),
    "Relay actions": (0.0, None),
}

SEED_REQUIRED_FILES = [
    "dataset_summary.csv",
    "metrics_overall.csv",
    "lead_time_summary.csv",
    "network_metrics.csv",
    "risk_metrics.csv",
    "split_assignments.csv",
    "validation_threshold_sensitivity.csv",
    "operating_point_metrics.csv",
]

CACHE_VERSION = "kinetic_topoguard_v6_fragmentation_events_correlated_radio"


def _mean_ci(values: pd.Series, bounds: tuple[float | None, float | None] | None = None) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        low = high = mean
    else:
        spread = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))
        low = mean - spread
        high = mean + spread
    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            low = max(low, lower)
            high = max(high, lower)
        if upper is not None:
            low = min(low, upper)
            high = min(high, upper)
    return pd.Series({"mean": mean, "ci95_low": low, "ci95_high": high})


def _aggregate_metrics(per_seed_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in per_seed_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            stats = _mean_ci(group[metric], METRIC_BOUNDS.get(metric))
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_ci95_low"] = stats["ci95_low"]
            row[f"{metric}_ci95_high"] = stats["ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def _cuda_available() -> bool | None:
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _backend_metadata(tasks: list[str]) -> dict:
    torch_version = None
    device = None
    if TORCH_AVAILABLE:
        import torch

        torch_version = str(torch.__version__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    neural_tasks = [task for task in tasks if task not in {"heuristic", "current_state_persistence", "shallow", "kinetic_topoguard"}]
    backends = {}
    for task in tasks:
        if task == "current_state_persistence":
            backends[task] = "deterministic_persistence_baseline"
        elif task == "heuristic":
            backends[task] = "deterministic_heuristic"
        elif task in {"shallow", "kinetic_topoguard"}:
            backends[task] = "scikit_learn"
        else:
            backends[task] = "pytorch" if TORCH_AVAILABLE else "scikit_learn_surrogate"
    return {
        "model_backend": backends,
        "torch_available": bool(TORCH_AVAILABLE),
        "torch_version": torch_version,
        "device": device,
        "surrogate_used": bool(neural_tasks and not TORCH_AVAILABLE),
    }
def _write_runtime_profile(out_path: Path, summary: dict, runtime_seconds: float) -> None:
    author_env = {
        "purpose": "Environment used to record this run's wall-clock runtime.",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": summary.get("cuda_available"),
        "scope": "Local host used for the executed run; this is not asserted to be identical to environment.yml.",
    }
    reproducibility_env = {
        "purpose": "Reference environment specification distributed for rerunning the package.",
        "environment_file": "environment.yml",
        "python_version": "3.12",
        "scope": (
            "Base scientific environment without PyTorch; use requirements-deep.txt "
            "or pip install -e .[deep] for actual PyTorch neural GNN and temporal baselines."
        ),
    }
    payload = {
        "experiment_name": summary["experiment_name"],
        "runtime_seconds": round(float(runtime_seconds), 3),
        "runtime_minutes": round(float(runtime_seconds) / 60.0, 3),
        "n_seeds": summary["n_seeds"],
        "total_snapshots": summary["total_snapshots"],
        "forecast_horizon_steps": summary["forecast_horizon_steps"],
        "graph_policies": summary["graph_policies"],
        "radio_scenarios": summary["radio_scenarios"],
        "python_version": author_env["python_version"],
        "platform": author_env["platform"],
        "cuda_available": author_env["cuda_available"],
        "model_backend": summary.get("model_backend", {}),
        "torch_available": summary.get("torch_available"),
        "torch_version": summary.get("torch_version"),
        "device": summary.get("device"),
        "surrogate_used": summary.get("surrogate_used"),
        "runtime_context": {
            "author_recorded_runtime_environment": author_env,
            "reproducibility_environment": reproducibility_env,
        },
        "scope": (
            "End-to-end pipeline wall-clock time for the configured benchmark on the local host; "
            "reproducibility environment metadata is recorded separately below."
        ),
        "note": (
            "Wall-clock runtime is environment-specific. The author-recorded runtime environment and "
            "the distributed reproducibility environment are intentionally separated; per-snapshot "
            "inference latency is reported in metrics_overall.csv."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _canonical_model_name(name: object) -> str:
    text = str(name)
    if text.startswith("Shallow ML"):
        return "Shallow ML"
    match = re.fullmatch(r"(tgcn|stgcn|tgn):(\d+)", text)
    if match:
        label_map = {"tgcn": "T-GCN", "stgcn": "STGCN", "tgn": "TGN"}
        return f"{label_map[match.group(1)]} (w={match.group(2)})"
    return text


def _canonicalise_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Model" not in df.columns:
        return df
    df = df.copy()
    df["Model"] = df["Model"].map(_canonical_model_name)
    return df


def _training_tasks(config: ExperimentConfig) -> list[str]:
    tasks = [
        "heuristic",
        "current_state_persistence",
        "shallow",
        "kinetic_topoguard",
        "GCN",
        "GAT",
        "GraphSAGE",
        "PI+MLP",
        "FANET-TopoGNN",
        "FANET-TopoGNN (concat)",
    ]
    for window in config.training["temporal_windows"]:
        for prefix in ["tgcn", "stgcn", "tgn"]:
            tasks.append(f"{prefix}:{window}")
    selected = config.training.get("selected_models")
    if not selected:
        return tasks
    unknown = [name for name in selected if name not in tasks]
    if unknown:
        raise ValueError(f"Unknown selected_models entries: {unknown}")
    return [name for name in selected if name in tasks]


def _train_task(task_name: str, train_data, val_data, config: ExperimentConfig, pi_dim: int, seed: int):
    if task_name == "heuristic":
        return fit_heuristic(train_data)
    if task_name == "current_state_persistence":
        return fit_current_state_persistence()
    if task_name == "shallow":
        return select_best_shallow(train_data, val_data)
    if task_name == "kinetic_topoguard":
        return fit_kinetic_topoguard(
            train_data,
            val_data,
            horizon_steps=int(config.sim.get("forecast_horizon_steps", config.evaluation["warning_horizon_steps"])),
            dt=float(config.sim["dt"]),
            seed=seed,
        )
    return train_torch_model(task_name, train_data, val_data, config.training, pi_dim, seed=seed)


def _cache_stem(task_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_name)


def _cache_paths(seed_dir: Path, task_name: str) -> tuple[Path, Path]:
    cache_dir = seed_dir / ".resume"
    stem = _cache_stem(task_name)
    return cache_dir / f"{stem}.json", cache_dir / f"{stem}.npz"


def _config_signature(config: ExperimentConfig) -> str:
    payload = json.dumps({"cache_version": CACHE_VERSION, "config": config.raw}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_model_cache(
    seed_dir: Path,
    task_name: str,
    summary: dict,
    lead_summary: dict,
    network_metrics: dict | None,
    risk_row: dict,
    residuals: np.ndarray,
    leads: list[float],
    normalised_leads: list[float],
    threshold_rows: list[dict],
    validation_threshold_rows: list[dict],
    operating_point_rows: list[dict],
    config_signature: str,
) -> None:
    meta_path, array_path = _cache_paths(seed_dir, task_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": task_name,
        "summary": summary,
        "lead_summary": lead_summary,
        "network_metrics": network_metrics,
        "risk_row": risk_row,
        "threshold_rows": threshold_rows,
        "validation_threshold_rows": validation_threshold_rows,
        "operating_point_rows": operating_point_rows,
        "config_signature": config_signature,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        array_path,
        residuals=np.asarray(residuals, dtype=float),
        leads=np.asarray(leads, dtype=float),
        normalised_leads=np.asarray(normalised_leads, dtype=float),
    )


def _load_model_cache(seed_dir: Path, task_name: str, config_signature: str | None = None) -> dict | None:
    meta_path, array_path = _cache_paths(seed_dir, task_name)
    if not meta_path.exists():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if config_signature is not None and payload.get("config_signature") != config_signature:
        return None
    residuals = np.asarray([], dtype=float)
    leads = []
    normalised_leads = []
    if array_path.exists():
        with np.load(array_path) as data:
            residuals = np.asarray(data["residuals"], dtype=float)
            leads = np.asarray(data["leads"], dtype=float).tolist()
            if "normalised_leads" in data:
                normalised_leads = np.asarray(data["normalised_leads"], dtype=float).tolist()
    payload["residuals"] = residuals
    payload["leads"] = leads
    payload["normalised_leads"] = normalised_leads
    payload.setdefault("threshold_rows", [])
    payload.setdefault("validation_threshold_rows", [])
    payload.setdefault("operating_point_rows", [])
    return payload


def _seed_complete(seed_dir: Path, config: ExperimentConfig) -> bool:
    if not all((seed_dir / name).exists() for name in SEED_REQUIRED_FILES):
        return False
    signature = _config_signature(config)
    for task_name in _training_tasks(config):
        if _load_model_cache(seed_dir, task_name, signature) is None:
            return False
    return True


def _load_seed_tables(seed_dir: Path) -> tuple[pd.DataFrame, ...]:
    threshold_path = seed_dir / "risk_threshold_sensitivity.csv"
    validation_threshold_path = seed_dir / "validation_threshold_sensitivity.csv"
    operating_point_path = seed_dir / "operating_point_metrics.csv"
    return (
        pd.read_csv(seed_dir / "metrics_overall.csv"),
        pd.read_csv(seed_dir / "lead_time_summary.csv"),
        pd.read_csv(seed_dir / "network_metrics.csv"),
        pd.read_csv(seed_dir / "risk_metrics.csv"),
        pd.read_csv(seed_dir / "dataset_summary.csv"),
        pd.read_csv(threshold_path) if threshold_path.exists() else pd.DataFrame(),
        pd.read_csv(validation_threshold_path) if validation_threshold_path.exists() else pd.DataFrame(),
        pd.read_csv(operating_point_path) if operating_point_path.exists() else pd.DataFrame(),
    )


def _load_cached_seed_maps(seed_dir: Path, config: ExperimentConfig) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    residual_map: dict[str, np.ndarray] = {}
    lead_map: dict[str, list[float]] = {}
    signature = _config_signature(config)
    for task_name in _training_tasks(config):
        cached = _load_model_cache(seed_dir, task_name, signature)
        if cached is None:
            continue
        model_name = cached["summary"]["Model"]
        residual_map[model_name] = np.asarray(cached["residuals"], dtype=float)
        lead_map[model_name] = list(cached["leads"])
    return residual_map, lead_map


def _run_seed(seed: int, config: ExperimentConfig, seed_dir: Path, resume: bool = False) -> tuple:
    seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[seed {seed}] building dataset")
    dataset_start = time.perf_counter()
    snapshots = build_dataset(config.sim, seed=seed)
    print(f"[seed {seed}] dataset ready in {time.perf_counter() - dataset_start:.1f}s with {len(snapshots):,} snapshots")
    seed_frame = to_frame(snapshots)
    dataset_summary = (
        seed_frame.groupby(["mobility", "n_nodes", "link_model", "graph_policy", "radio_scenario"], as_index=False)
        .agg(
            snapshots=("run_id", "count"),
            beta_fixed_mean=("beta_fixed", "mean"),
            beta_adaptive_mean=("beta_adaptive", "mean"),
            beta_current_mean=("beta_current", "mean"),
            connected_ratio=("is_connected", "mean"),
            frag_rate=("frag_at_horizon", "mean"),
            edge_count_mean=("edge_count", "mean"),
            edge_count_fixed_mean=("edge_count_fixed", "mean"),
            edge_count_adaptive_mean=("edge_count_adaptive", "mean"),
            radius_mean=("radius", "mean"),
            radius_adaptive_mean=("radius_adaptive", "mean"),
        )
    )
    save_table(dataset_summary, seed_dir, "dataset_summary")
    split_stratify_by = list(config.sim.get("split_stratify_by", ["mobility"]))
    for column in ["graph_policy", "radio_scenario"]:
        if column in seed_frame.columns and seed_frame[column].nunique() > 1 and column not in split_stratify_by:
            split_stratify_by.append(column)
    train_data, val_data, test_data, split_frame = train_val_test_split(
        snapshots,
        split_seed=int(config.sim.get("split_seed", seed)),
        stratify_by=tuple(split_stratify_by),
        return_mapping=True,
    )
    save_table(split_frame, seed_dir, "split_assignments")
    pi_dim = train_data[0].pi.shape[0]
    signature = _config_signature(config)

    metrics_rows = []
    lead_rows = []
    network_rows = []
    residual_map: dict[str, np.ndarray] = {}
    lead_map: dict[str, list[float]] = {}
    risk_rows = []
    threshold_rows = []
    validation_threshold_rows = []
    operating_point_rows = []

    for task_name in _training_tasks(config):
        cached = _load_model_cache(seed_dir, task_name, signature) if resume else None
        if cached is not None:
            model_name = cached["summary"]["Model"]
            print(f"[seed {seed}] resume cache hit for {model_name}")
            metrics_rows.append(cached["summary"])
            lead_rows.append(cached["lead_summary"])
            lead_map[model_name] = list(cached["leads"])
            residual_map[model_name] = np.asarray(cached["residuals"], dtype=float)
            risk_rows.append(cached["risk_row"])
            threshold_rows.extend(cached.get("threshold_rows", []))
            validation_threshold_rows.extend(cached.get("validation_threshold_rows", []))
            operating_point_rows.extend(cached.get("operating_point_rows", []))
            if cached["network_metrics"] is not None:
                network_rows.append(cached["network_metrics"])
            continue

        print(f"[seed {seed}] training {task_name}")
        model_start = time.perf_counter()
        result = _train_task(task_name, train_data, val_data, config, pi_dim, seed)
        _, val_risk_scores, _, aligned_val, _ = predict_generic(result, val_data)
        preds, risk_scores, inference_ms, aligned_test, risk_threshold = predict_generic(result, test_data)
        summary, leads, normalised_leads = evaluate_predictions(
            result.model_name,
            aligned_test,
            preds,
            risk_scores,
            inference_ms,
            dt=config.sim["dt"],
            bootstrap_rounds=config.evaluation["bootstrap_rounds"],
            horizon_steps=config.sim.get("forecast_horizon_steps", config.evaluation["warning_horizon_steps"]),
            risk_threshold=risk_threshold,
        )
        summary["seed"] = seed
        summary["Model_Backend"] = _backend_metadata([task_name])["model_backend"][task_name]
        metrics_rows.append(summary)
        lead_summary = summarise_leads(result.model_name, leads, normalised_leads)
        lead_summary["seed"] = seed
        lead_rows.append(lead_summary)
        lead_map[result.model_name] = leads
        y_true = np.asarray([snap.beta_target for snap in aligned_test], dtype=float)
        residual_map[result.model_name] = preds - y_true
        risk_row = {
            "seed": seed,
            "Model": result.model_name,
            "Risk_Precision": summary["Risk_Precision"],
            "Risk_Recall": summary["Risk_Recall"],
            "Risk_F1": summary["Risk_F1"],
            "Risk_Specificity": summary["Risk_Specificity"],
            "Risk_Accuracy": summary["Risk_Accuracy"],
            "Risk_PR_AUC": summary["Risk_PR_AUC"],
            "Risk_ROC_AUC": summary["Risk_ROC_AUC"],
            "Risk_Brier": summary["Risk_Brier"],
            "Risk_ECE": summary["Risk_ECE"],
            "False_Alarms": summary["False_Alarms"],
            "False_Alarms_per_minute": summary["False_Alarms_per_minute"],
            "Alert_Events": summary["Alert_Events"],
            "True_Alert_Events": summary["True_Alert_Events"],
            "False_Alert_Events": summary["False_Alert_Events"],
            "Ground_Truth_Fragmentation_Events": summary["Ground_Truth_Fragmentation_Events"],
            "Missed_Fragmentation_Events": summary["Missed_Fragmentation_Events"],
            "Alert_Event_Precision": summary["Alert_Event_Precision"],
            "Alert_Event_Recall": summary["Alert_Event_Recall"],
            "Alert_Event_F1": summary["Alert_Event_F1"],
            "False_Alert_Events_per_minute": summary["False_Alert_Events_per_minute"],
            "Model_Backend": summary["Model_Backend"],
        }
        risk_rows.append(risk_row)
        y_risk_true = np.asarray([snap.frag_at_horizon for snap in aligned_test], dtype=int)
        horizon_steps = int(config.sim.get("forecast_horizon_steps", config.evaluation["warning_horizon_steps"]))
        model_threshold_rows = threshold_sensitivity_rows(
            y_risk_true,
            risk_scores,
            dt=float(config.sim["dt"]),
            snapshots=aligned_test,
            horizon_steps=horizon_steps,
        )
        for row in model_threshold_rows:
            row.update(
                {
                    "seed": seed,
                    "Model": result.model_name,
                    "Model_Backend": summary["Model_Backend"],
                    "Split": "test_sensitivity_only",
                }
            )
        threshold_rows.extend(model_threshold_rows)
        y_val_risk = np.asarray([snap.frag_at_horizon for snap in aligned_val], dtype=int)
        model_validation_rows = threshold_sensitivity_rows(
            y_val_risk,
            val_risk_scores,
            dt=float(config.sim["dt"]),
            snapshots=aligned_val,
            horizon_steps=horizon_steps,
            thresholds=tuple(np.round(np.linspace(0.05, 0.95, 19), 2)),
        )
        for row in model_validation_rows:
            row.update(
                {
                    "seed": seed,
                    "Model": result.model_name,
                    "Model_Backend": summary["Model_Backend"],
                    "Split": "validation_selection",
                }
            )
        validation_threshold_rows.extend(model_validation_rows)

        if result.model_name == "Kinetic-TopoGuard":
            val_frame = pd.DataFrame(model_validation_rows)
            budgets = config.evaluation.get("false_alert_event_budgets_per_minute", [2.0, 1.0])
            for policy, budget in zip(["deployable", "strict"], budgets):
                feasible = val_frame[val_frame["False_Alert_Events_per_minute"] <= float(budget)].copy()
                if feasible.empty:
                    selected = val_frame.sort_values(
                        ["False_Alert_Events_per_minute", "Risk_F1"],
                        ascending=[True, False],
                    ).iloc[0]
                    constraint_met = False
                else:
                    selected = feasible.sort_values(
                        ["Alert_Event_F1", "Alert_Event_Recall", "Risk_F1", "Risk_Recall"],
                        ascending=[False, False, False, False],
                    ).iloc[0]
                    constraint_met = True
                selected_threshold = float(selected["Threshold"])
                test_summary, _, _ = evaluate_predictions(
                    result.model_name,
                    aligned_test,
                    preds,
                    risk_scores,
                    inference_ms,
                    dt=config.sim["dt"],
                    bootstrap_rounds=config.evaluation["bootstrap_rounds"],
                    horizon_steps=horizon_steps,
                    risk_threshold=selected_threshold,
                )
                operating_point_rows.append(
                    {
                        "seed": seed,
                        "Model": result.model_name,
                        "Policy": policy,
                        "Validation_False_Alert_Budget_per_minute": float(budget),
                        "Validation_Constraint_Met": bool(constraint_met),
                        "Selected_Threshold": selected_threshold,
                        "Validation_Risk_F1": float(selected["Risk_F1"]),
                        "Validation_False_Alert_Events_per_minute": float(
                            selected["False_Alert_Events_per_minute"]
                        ),
                        **{f"Test_{key}": value for key, value in test_summary.items() if key != "Model"},
                    }
                )
        network_metrics = None
        if result.model_name == "FANET-TopoGNN":
            plot_scatter(y_true, preds, seed_dir / "fanet_topognn_scatter.png")
        if result.model_name in {"GCN", "GAT", "GraphSAGE", "Kinetic-TopoGuard", "FANET-TopoGNN", "FANET-TopoGNN (concat)"}:
            network_metrics = run_network_controller(
                aligned_test,
                preds,
                risk_scores,
                boost=config.evaluation["network_radius_boost"],
                risk_threshold=risk_threshold,
                sim_config=config.sim,
                relay_max_speed_mps=float(config.evaluation.get("relay_max_speed_mps", 30.0)),
                relay_max_acceleration_mps2=float(
                    config.evaluation.get("relay_max_acceleration_mps2", 12.0)
                ),
                relay_link_budget_boost_db=float(
                    config.evaluation.get("relay_link_budget_boost_db", 3.0)
                ),
            )
            network_metrics["Model"] = result.model_name
            network_metrics["seed"] = seed
            network_rows.append(network_metrics)
        _save_model_cache(
            seed_dir,
            task_name,
            summary=summary,
            lead_summary=lead_summary,
            network_metrics=network_metrics,
            risk_row=risk_row,
            residuals=residual_map[result.model_name],
            leads=leads,
            normalised_leads=normalised_leads,
            threshold_rows=model_threshold_rows,
            validation_threshold_rows=model_validation_rows,
            operating_point_rows=[
                row
                for row in operating_point_rows
                if row["seed"] == seed and row["Model"] == result.model_name
            ],
            config_signature=signature,
        )
        print(f"[seed {seed}] finished {result.model_name} in {time.perf_counter() - model_start:.1f}s")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE").reset_index(drop=True)
    leads_df = pd.DataFrame(lead_rows).sort_values("Lead_median_ms", ascending=False).reset_index(drop=True)
    network_df = pd.DataFrame(network_rows)
    risk_df = pd.DataFrame(risk_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    validation_threshold_df = pd.DataFrame(validation_threshold_rows)
    operating_point_df = pd.DataFrame(operating_point_rows)
    save_table(metrics_df, seed_dir, "metrics_overall")
    save_table(leads_df, seed_dir, "lead_time_summary")
    save_table(network_df, seed_dir, "network_metrics")
    save_table(risk_df, seed_dir, "risk_metrics")
    if not threshold_df.empty:
        save_table(threshold_df, seed_dir, "risk_threshold_sensitivity")
    if not validation_threshold_df.empty:
        save_table(validation_threshold_df, seed_dir, "validation_threshold_sensitivity")
    if not operating_point_df.empty:
        save_table(operating_point_df, seed_dir, "operating_point_metrics")
    plot_lead_cdf({name: vals for name, vals in lead_map.items() if name in {"Kinetic-TopoGuard", "GraphSAGE", "FANET-TopoGNN"}}, seed_dir / "lead_cdf.png")
    return (
        metrics_df,
        leads_df,
        network_df,
        residual_map,
        lead_map,
        risk_df,
        threshold_df,
        validation_threshold_df,
        operating_point_df,
    )


def _build_stats_table(metrics_seed_df: pd.DataFrame, leads_seed_df: pd.DataFrame) -> pd.DataFrame:
    reference = "Kinetic-TopoGuard" if "Kinetic-TopoGuard" in set(metrics_seed_df["Model"].astype(str)) else "FANET-TopoGNN"
    candidates = [name for name in sorted(metrics_seed_df["Model"].unique()) if name != reference]
    rows = []
    ref_mae = metrics_seed_df[metrics_seed_df["Model"] == reference].sort_values("seed")["MAE"].to_numpy(dtype=float)
    ref_lead = leads_seed_df[leads_seed_df["Model"] == reference].sort_values("seed")["Lead_median_ms"].to_numpy(dtype=float)
    ref_risk_f1 = metrics_seed_df[metrics_seed_df["Model"] == reference].sort_values("seed")["Risk_F1"].to_numpy(dtype=float)
    for candidate in candidates:
        cand_mae = metrics_seed_df[metrics_seed_df["Model"] == candidate].sort_values("seed")["MAE"].to_numpy(dtype=float)
        cand_lead = leads_seed_df[leads_seed_df["Model"] == candidate].sort_values("seed")["Lead_median_ms"].to_numpy(dtype=float)
        cand_risk_f1 = metrics_seed_df[metrics_seed_df["Model"] == candidate].sort_values("seed")["Risk_F1"].to_numpy(dtype=float)
        row = paired_statistics(reference, ref_mae, candidate, cand_mae)
        lead_stats = paired_statistics(reference, ref_lead, candidate, cand_lead)
        risk_stats = paired_statistics(reference, ref_risk_f1, candidate, cand_risk_f1)
        row["lead_paired_t_pvalue"] = lead_stats["paired_t_pvalue"]
        row["lead_wilcoxon_pvalue"] = lead_stats["wilcoxon_pvalue"]
        row["lead_cohens_d"] = lead_stats["cohens_d"]
        row["risk_f1_paired_t_pvalue"] = risk_stats["paired_t_pvalue"]
        row["risk_f1_wilcoxon_pvalue"] = risk_stats["wilcoxon_pvalue"]
        row["risk_f1_cohens_d"] = risk_stats["cohens_d"]
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["mae_fdr_pvalue"] = benjamini_hochberg(stats_df["paired_t_pvalue"].tolist())
        stats_df["lead_fdr_pvalue"] = benjamini_hochberg(stats_df["lead_paired_t_pvalue"].tolist())
        stats_df["risk_f1_fdr_pvalue"] = benjamini_hochberg(stats_df["risk_f1_paired_t_pvalue"].tolist())
        stats_df = stats_df.sort_values(["risk_f1_fdr_pvalue", "mae_fdr_pvalue", "lead_fdr_pvalue"]).reset_index(drop=True)
    return stats_df


def _run_or_load_seed(seed: int, config: ExperimentConfig, seeds_root: Path, resume: bool) -> tuple:
    seed_dir = seeds_root / f"seed_{seed}"
    if resume and _seed_complete(seed_dir, config):
        print(f"[resume] skipping completed seed {seed}")
        (
            metrics_df,
            leads_df,
            network_df,
            risk_df,
            dataset_df,
            threshold_df,
            validation_threshold_df,
            operating_point_df,
        ) = _load_seed_tables(seed_dir)
        residual_map, lead_map = _load_cached_seed_maps(seed_dir, config)
    else:
        (
            metrics_df,
            leads_df,
            network_df,
            residual_map,
            lead_map,
            risk_df,
            threshold_df,
            validation_threshold_df,
            operating_point_df,
        ) = _run_seed(seed, config, seed_dir, resume=resume)
        dataset_df = pd.read_csv(seed_dir / "dataset_summary.csv")
    return (
        seed,
        metrics_df,
        leads_df,
        network_df,
        residual_map,
        lead_map,
        risk_df,
        threshold_df,
        validation_threshold_df,
        operating_point_df,
        dataset_df,
    )


def run_experiment(config: ExperimentConfig, resume: bool = False, seed_workers: int = 1) -> dict:
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    seeds_root = out_dir / "per_seed"
    seeds_root.mkdir(exist_ok=True)

    metrics_seed_frames = []
    leads_seed_frames = []
    network_seed_frames = []
    risk_seed_frames = []
    threshold_seed_frames = []
    validation_threshold_seed_frames = []
    operating_point_seed_frames = []
    dataset_frames = []
    residual_map_first: dict[str, np.ndarray] = {}
    lead_map_first: dict[str, list[float]] = {}

    run_start = time.perf_counter()
    seeds = list(config.training["seed_list"])
    worker_count = max(1, min(int(seed_workers), len(seeds)))
    if worker_count == 1:
        seed_results = [_run_or_load_seed(seed, config, seeds_root, resume) for seed in seeds]
    else:
        print(f"[run] executing {len(seeds)} seeds with {worker_count} parallel workers")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            seed_results = list(
                executor.map(
                    _run_or_load_seed,
                    seeds,
                    [config] * len(seeds),
                    [seeds_root] * len(seeds),
                    [resume] * len(seeds),
                )
            )

    for idx, result in enumerate(seed_results):
        (
            seed,
            metrics_df,
            leads_df,
            network_df,
            residual_map,
            lead_map,
            risk_df,
            threshold_df,
            validation_threshold_df,
            operating_point_df,
            dataset_df,
        ) = result
        metrics_seed_frames.append(metrics_df)
        leads_seed_frames.append(leads_df)
        network_seed_frames.append(network_df)
        risk_seed_frames.append(risk_df)
        if not threshold_df.empty:
            threshold_seed_frames.append(threshold_df)
        if not validation_threshold_df.empty:
            validation_threshold_seed_frames.append(validation_threshold_df)
        if not operating_point_df.empty:
            operating_point_seed_frames.append(operating_point_df)
        dataset_frames.append(dataset_df)
        if idx == 0:
            residual_map_first = {_canonical_model_name(name): values for name, values in residual_map.items()}
            lead_map_first = {_canonical_model_name(name): values for name, values in lead_map.items()}

    metrics_seed_df = pd.concat(metrics_seed_frames, ignore_index=True)
    leads_seed_df = pd.concat(leads_seed_frames, ignore_index=True)
    network_seed_df = pd.concat(network_seed_frames, ignore_index=True)
    risk_seed_df = pd.concat(risk_seed_frames, ignore_index=True)
    threshold_seed_df = pd.concat(threshold_seed_frames, ignore_index=True) if threshold_seed_frames else pd.DataFrame()
    validation_threshold_seed_df = (
        pd.concat(validation_threshold_seed_frames, ignore_index=True)
        if validation_threshold_seed_frames
        else pd.DataFrame()
    )
    operating_point_seed_df = (
        pd.concat(operating_point_seed_frames, ignore_index=True)
        if operating_point_seed_frames
        else pd.DataFrame()
    )
    metrics_seed_df = _canonicalise_model_labels(metrics_seed_df)
    leads_seed_df = _canonicalise_model_labels(leads_seed_df)
    network_seed_df = _canonicalise_model_labels(network_seed_df)
    risk_seed_df = _canonicalise_model_labels(risk_seed_df)
    dataset_seed_df = pd.concat(dataset_frames, ignore_index=True)
    dataset_group_cols = [col for col in ["mobility", "n_nodes", "link_model", "graph_policy", "radio_scenario"] if col in dataset_seed_df.columns]
    dataset_numeric_cols = [col for col in dataset_seed_df.columns if col not in dataset_group_cols]
    dataset_summary_df = dataset_seed_df.groupby(dataset_group_cols, as_index=False)[dataset_numeric_cols].mean()

    metrics_df = _aggregate_metrics(metrics_seed_df, ["Model"], NUMERIC_METRICS).sort_values("MAE_mean").reset_index(drop=True)
    leads_df = _aggregate_metrics(leads_seed_df, ["Model"], ["Lead_5th_ms", "Lead_median_ms", "Lead_IQR_ms", "Lead_95th_ms", "Lead_mean_ms", "Lead_norm_mean"]).sort_values("Lead_median_ms_mean", ascending=False).reset_index(drop=True)
    network_metric_cols = [
        "Connectivity ratio",
        "Reachability-delivery proxy (%)",
        "Proxy delay (ms)",
        "Proactive reroute (%)",
        "DTN buffered (%)",
        "Relay actions",
    ]
    network_metric_cols = [col for col in network_metric_cols if col in network_seed_df.columns]
    network_df = _aggregate_metrics(network_seed_df, ["Model"], network_metric_cols)
    if not network_df.empty:
        network_df = network_df.sort_values("Connectivity ratio_mean", ascending=False).reset_index(drop=True)
    risk_metric_cols = [
        "Risk_Precision",
        "Risk_Recall",
        "Risk_F1",
        "Risk_Specificity",
        "Risk_Accuracy",
        "Risk_PR_AUC",
        "Risk_ROC_AUC",
        "Risk_Brier",
        "Risk_ECE",
        "False_Alarms",
        "False_Alarms_per_minute",
        "Alert_Events",
        "True_Alert_Events",
        "False_Alert_Events",
        "Ground_Truth_Fragmentation_Events",
        "Missed_Fragmentation_Events",
        "Alert_Event_Precision",
        "Alert_Event_Recall",
        "Alert_Event_F1",
        "False_Alert_Events_per_minute",
    ]
    risk_df = _aggregate_metrics(risk_seed_df, ["Model"], risk_metric_cols).sort_values("Alert_Event_F1_mean", ascending=False).reset_index(drop=True)
    threshold_df = pd.DataFrame()
    if not threshold_seed_df.empty:
        threshold_metric_cols = [
            "Risk_Precision",
            "Risk_Recall",
            "Risk_F1",
            "Risk_Specificity",
            "Risk_Accuracy",
            "False_Alarms",
            "False_Alarms_per_minute",
            "Alert_Events",
            "True_Alert_Events",
            "False_Alert_Events",
            "Ground_Truth_Fragmentation_Events",
            "Missed_Fragmentation_Events",
            "Alert_Event_Precision",
            "Alert_Event_Recall",
            "Alert_Event_F1",
            "False_Alert_Events_per_minute",
        ]
        threshold_df = _aggregate_metrics(threshold_seed_df, ["Model", "Threshold"], threshold_metric_cols)

    validation_threshold_df = pd.DataFrame()
    if not validation_threshold_seed_df.empty:
        validation_threshold_df = _aggregate_metrics(
            validation_threshold_seed_df,
            ["Model", "Threshold"],
            [
                "Risk_Precision",
                "Risk_Recall",
                "Risk_F1",
                "False_Alarms_per_minute",
                "Alert_Event_Precision",
                "Alert_Event_Recall",
                "Alert_Event_F1",
                "False_Alert_Events_per_minute",
            ],
        )

    operating_point_df = pd.DataFrame()
    if not operating_point_seed_df.empty:
        operating_point_seed_df["Validation_Constraint_Met"] = (
            operating_point_seed_df["Validation_Constraint_Met"].astype(float)
        )
        numeric_operating_cols = [
            column
            for column in operating_point_seed_df.columns
            if column not in {"seed", "Model", "Policy"}
            and pd.api.types.is_numeric_dtype(operating_point_seed_df[column])
        ]
        operating_point_df = _aggregate_metrics(
            operating_point_seed_df,
            ["Model", "Policy"],
            numeric_operating_cols,
        )

    shallow_name = next((name for name in metrics_df["Model"] if name.startswith("Shallow ML")), None)
    static_models = [
        "Density + min-distance heuristics",
        "Current-state persistence baseline",
        "Kinetic-TopoGuard",
        "GCN",
        "GAT",
        "GraphSAGE",
        "PI+MLP",
        "FANET-TopoGNN",
        "FANET-TopoGNN (concat)",
    ]
    if shallow_name is not None:
        static_models.insert(2, shallow_name)
    static_df = metrics_df[metrics_df["Model"].isin(static_models)][["Model", "MAE_mean", "R2_mean"]].merge(leads_df[["Model", "Lead_median_ms_mean"]], on="Model", how="left")
    temporal_df = metrics_df[metrics_df["Model"].str.contains(r"\(w=\d+\)", regex=True)][["Model", "MAE_mean", "R2_mean"]].merge(leads_df[["Model", "Lead_median_ms_mean"]], on="Model", how="left")
    stats_df = _build_stats_table(metrics_seed_df, leads_seed_df)

    save_table(dataset_summary_df, out_dir, "dataset_summary")
    save_table(metrics_seed_df, out_dir, "per_seed_metrics")
    save_table(leads_seed_df, out_dir, "per_seed_leads")
    save_table(network_seed_df, out_dir, "per_seed_network_metrics")
    save_table(risk_seed_df, out_dir, "per_seed_risk_metrics")
    if not validation_threshold_seed_df.empty:
        save_table(validation_threshold_seed_df, out_dir, "per_seed_validation_threshold_sensitivity")
    if not operating_point_seed_df.empty:
        save_table(operating_point_seed_df, out_dir, "per_seed_operating_point_metrics")
    save_table(metrics_df, out_dir, "metrics_overall")
    save_table(leads_df, out_dir, "lead_time_summary")
    save_table(static_df, out_dir, "ablation_static")
    save_table(temporal_df, out_dir, "ablation_temporal")
    save_table(network_df, out_dir, "network_metrics")
    save_table(risk_df, out_dir, "risk_metrics")
    if not threshold_df.empty:
        save_table(threshold_df, out_dir, "risk_threshold_sensitivity")
    if not validation_threshold_df.empty:
        save_table(validation_threshold_df, out_dir, "validation_threshold_sensitivity")
    if not operating_point_df.empty:
        save_table(operating_point_df, out_dir, "operating_point_metrics")
    save_table(stats_df, out_dir, "stats_tests")

    lead_plot_map = {name: vals for name, vals in lead_map_first.items() if name in {"Kinetic-TopoGuard", "FANET-TopoGNN", "PI+MLP", "FANET-TopoGNN (concat)", "GraphSAGE"}}
    if lead_plot_map:
        plot_lead_cdf(lead_plot_map, figures_dir / "lead_cdf.png")
    plot_metric_bars(metrics_df.rename(columns={"MAE_mean": "MAE"}), "MAE", figures_dir / "mae_comparison.png")
    plot_metric_bars(metrics_df.rename(columns={"R2_mean": "R2"}), "R2", figures_dir / "r2_comparison.png")
    plot_metric_bars(risk_df.rename(columns={"Risk_F1_mean": "Risk_F1"}), "Risk_F1", figures_dir / "risk_f1_comparison.png")
    plot_latency(metrics_df.rename(columns={"Inference_ms_mean": "Inference_ms"}), figures_dir / "latency_budget.png")
    if not network_df.empty:
        plot_network_metrics(
            network_df.rename(
                columns={
                    "Connectivity ratio_mean": "Connectivity ratio",
                    "Reachability-delivery proxy (%)_mean": "Reachability-delivery proxy (%)",
                    "Proxy delay (ms)_mean": "Proxy delay (ms)",
                }
            ),
            figures_dir / "network_metrics.png",
        )
    plot_publication_performance(metrics_df, figures_dir / "publication_performance_ci.png")
    plot_lead_time_summary(leads_df, figures_dir / "publication_lead_time_ci.png")
    plot_dataset_overview(dataset_summary_df, figures_dir / "dataset_overview.png")
    plot_radio_policy_sensitivity(dataset_summary_df, figures_dir / "radio_policy_sensitivity.png")
    if not network_df.empty:
        plot_network_extended_metrics(network_df, figures_dir / "network_extended_metrics.png")
    plot_ablation_summary(metrics_df, figures_dir / "fusion_ablation_summary.png")
    plot_statistical_summary(stats_df, figures_dir / "statistical_evidence.png")
    if residual_map_first:
        plot_residual_box(residual_map_first, figures_dir / "residual_boxplot.png")

    total_snapshots = int(dataset_seed_df["snapshots"].sum()) if "snapshots" in dataset_seed_df.columns else 0
    lead_models: list[str] = []
    if not leads_df.empty and float(leads_df.iloc[0]["Lead_median_ms_mean"]) > 0.0:
        best_lead = float(leads_df["Lead_median_ms_mean"].max())
        lead_models = leads_df.loc[
            np.isclose(leads_df["Lead_median_ms_mean"].to_numpy(dtype=float), best_lead, atol=1e-6),
            "Model",
        ].astype(str).tolist()
    summary = {
        "experiment_name": config.experiment_name,
        "n_seeds": len(config.training["seed_list"]),
        "total_snapshots": total_snapshots,
        "graph_policies": sorted(dataset_seed_df["graph_policy"].dropna().unique().tolist()) if "graph_policy" in dataset_seed_df.columns else [],
        "radio_scenarios": sorted(dataset_seed_df["radio_scenario"].dropna().unique().tolist()) if "radio_scenario" in dataset_seed_df.columns else [],
        "forecast_horizon_steps": int(config.sim.get("forecast_horizon_steps", config.evaluation["warning_horizon_steps"])),
        "best_model_by_mae": metrics_df.iloc[0]["Model"] if not metrics_df.empty else None,
        "best_model_by_lead": ", ".join(lead_models) if lead_models else None,
        "best_models_by_lead": lead_models,
        "output_dir": str(out_dir),
        "cache_version": CACHE_VERSION,
    }
    summary.update(_backend_metadata(_training_tasks(config)))
    runtime_seconds = time.perf_counter() - run_start
    summary["runtime_seconds"] = float(runtime_seconds)
    summary["runtime_minutes"] = float(runtime_seconds) / 60.0
    summary["cuda_available"] = _cuda_available()
    save_summary_json(out_dir / "summary.json", summary)
    write_markdown_report(out_dir / "report.md", summary, metrics_df, leads_df, network_df, risk_df)
    write_claims_summary(out_dir / "claims_summary.md", summary, metrics_df, leads_df, network_df, risk_df, stats_df, dataset_summary_df)
    export_manuscript_tables(out_dir / "manuscript_tables.tex", metrics_df, leads_df, network_df, risk_df, summary)
    export_manuscript_summary(out_dir / "manuscript_summary.json", summary, metrics_df, leads_df)
    _write_runtime_profile(out_dir / "runtime_profile.json", summary, runtime_seconds)
    write_artifact_manifest(out_dir / "artifact_manifest.txt", out_dir)
    print(f"[run] completed in {runtime_seconds:.1f}s")
    return summary
