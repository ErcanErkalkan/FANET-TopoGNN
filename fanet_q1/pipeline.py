from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dataset import build_dataset, to_frame, train_val_test_split
from .evaluation import benjamini_hochberg, evaluate_predictions, paired_statistics, predict_generic, run_network_controller, save_summary_json, summarise_leads
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
from .training import fit_heuristic, fit_kinetic_topoguard, fit_union_find_oracle, select_best_shallow, train_torch_model


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
    "Lead_5th_ms": (0.0, None),
    "Lead_median_ms": (0.0, None),
    "Lead_IQR_ms": (0.0, None),
    "Lead_95th_ms": (0.0, None),
    "Lead_mean_ms": (0.0, None),
    "Lead_norm_mean": (0.0, None),
    "Connectivity ratio": (0.0, 1.0),
    "PDR (%)": (0.0, 100.0),
    "Avg. end-to-end delay (ms)": (0.0, None),
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
]

CACHE_VERSION = "kinetic_topoguard_v1"


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
    rows = []
    for keys, group in per_seed_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for metric in metric_cols:
            stats = _mean_ci(group[metric], METRIC_BOUNDS.get(metric))
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_ci95_low"] = stats["ci95_low"]
            row[f"{metric}_ci95_high"] = stats["ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows)


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
        "union_find",
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
    if task_name == "union_find":
        return fit_union_find_oracle()
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
    return payload


def _seed_complete(seed_dir: Path, config: ExperimentConfig) -> bool:
    if not all((seed_dir / name).exists() for name in SEED_REQUIRED_FILES):
        return False
    signature = _config_signature(config)
    for task_name in _training_tasks(config):
        if _load_model_cache(seed_dir, task_name, signature) is None:
            return False
    return True


def _load_seed_tables(seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(seed_dir / "metrics_overall.csv"),
        pd.read_csv(seed_dir / "lead_time_summary.csv"),
        pd.read_csv(seed_dir / "network_metrics.csv"),
        pd.read_csv(seed_dir / "risk_metrics.csv"),
        pd.read_csv(seed_dir / "dataset_summary.csv"),
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


def _run_seed(seed: int, config: ExperimentConfig, seed_dir: Path, resume: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, list[float]], pd.DataFrame]:
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
            frag_rate=("frag_within_horizon", "mean"),
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
            if cached["network_metrics"] is not None:
                network_rows.append(cached["network_metrics"])
            continue

        print(f"[seed {seed}] training {task_name}")
        model_start = time.perf_counter()
        result = _train_task(task_name, train_data, val_data, config, pi_dim, seed)
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
        }
        risk_rows.append(risk_row)
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
            config_signature=signature,
        )
        print(f"[seed {seed}] finished {result.model_name} in {time.perf_counter() - model_start:.1f}s")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE").reset_index(drop=True)
    leads_df = pd.DataFrame(lead_rows).sort_values("Lead_median_ms", ascending=False).reset_index(drop=True)
    network_df = pd.DataFrame(network_rows)
    risk_df = pd.DataFrame(risk_rows)
    save_table(metrics_df, seed_dir, "metrics_overall")
    save_table(leads_df, seed_dir, "lead_time_summary")
    save_table(network_df, seed_dir, "network_metrics")
    save_table(risk_df, seed_dir, "risk_metrics")
    plot_lead_cdf({name: vals for name, vals in lead_map.items() if name in {"Kinetic-TopoGuard", "GraphSAGE", "FANET-TopoGNN"}}, seed_dir / "lead_cdf.png")
    return metrics_df, leads_df, network_df, residual_map, lead_map, risk_df


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


def run_experiment(config: ExperimentConfig, resume: bool = False) -> dict:
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
    dataset_frames = []
    residual_map_first: dict[str, np.ndarray] = {}
    lead_map_first: dict[str, list[float]] = {}

    run_start = time.perf_counter()
    for idx, seed in enumerate(config.training["seed_list"]):
        seed_dir = seeds_root / f"seed_{seed}"
        if resume and _seed_complete(seed_dir, config):
            print(f"[resume] skipping completed seed {seed}")
            metrics_df, leads_df, network_df, risk_df, dataset_df = _load_seed_tables(seed_dir)
            residual_map, lead_map = _load_cached_seed_maps(seed_dir, config)
        else:
            metrics_df, leads_df, network_df, residual_map, lead_map, risk_df = _run_seed(seed, config, seed_dir, resume=resume)
            dataset_df = pd.read_csv(seed_dir / "dataset_summary.csv")
        metrics_seed_frames.append(metrics_df)
        leads_seed_frames.append(leads_df)
        network_seed_frames.append(network_df)
        risk_seed_frames.append(risk_df)
        dataset_frames.append(dataset_df)
        if idx == 0:
            residual_map_first = {_canonical_model_name(name): values for name, values in residual_map.items()}
            lead_map_first = {_canonical_model_name(name): values for name, values in lead_map.items()}

    metrics_seed_df = pd.concat(metrics_seed_frames, ignore_index=True)
    leads_seed_df = pd.concat(leads_seed_frames, ignore_index=True)
    network_seed_df = pd.concat(network_seed_frames, ignore_index=True)
    risk_seed_df = pd.concat(risk_seed_frames, ignore_index=True)
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
        "PDR (%)",
        "Avg. end-to-end delay (ms)",
        "Proactive reroute (%)",
        "DTN buffered (%)",
        "Relay actions",
    ]
    network_metric_cols = [col for col in network_metric_cols if col in network_seed_df.columns]
    network_df = _aggregate_metrics(network_seed_df, ["Model"], network_metric_cols).sort_values("Connectivity ratio_mean", ascending=False).reset_index(drop=True)
    risk_df = _aggregate_metrics(risk_seed_df, ["Model"], ["Risk_Precision", "Risk_Recall", "Risk_F1", "Risk_Specificity", "Risk_Accuracy"]).sort_values("Risk_F1_mean", ascending=False).reset_index(drop=True)

    shallow_name = next((name for name in metrics_df["Model"] if name.startswith("Shallow ML")), None)
    static_models = [
        "Density + min-distance heuristics",
        "Union-Find detection oracle",
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
    save_table(metrics_df, out_dir, "metrics_overall")
    save_table(leads_df, out_dir, "lead_time_summary")
    save_table(static_df, out_dir, "ablation_static")
    save_table(temporal_df, out_dir, "ablation_temporal")
    save_table(network_df, out_dir, "network_metrics")
    save_table(risk_df, out_dir, "risk_metrics")
    save_table(stats_df, out_dir, "stats_tests")

    lead_plot_map = {name: vals for name, vals in lead_map_first.items() if name in {"Kinetic-TopoGuard", "FANET-TopoGNN", "PI+MLP", "FANET-TopoGNN (concat)", "GraphSAGE"}}
    if lead_plot_map:
        plot_lead_cdf(lead_plot_map, figures_dir / "lead_cdf.png")
    plot_metric_bars(metrics_df.rename(columns={"MAE_mean": "MAE"}), "MAE", figures_dir / "mae_comparison.png")
    plot_metric_bars(metrics_df.rename(columns={"R2_mean": "R2"}), "R2", figures_dir / "r2_comparison.png")
    plot_metric_bars(risk_df.rename(columns={"Risk_F1_mean": "Risk_F1"}), "Risk_F1", figures_dir / "risk_f1_comparison.png")
    plot_latency(metrics_df.rename(columns={"Inference_ms_mean": "Inference_ms"}), figures_dir / "latency_budget.png")
    plot_network_metrics(network_df.rename(columns={"Connectivity ratio_mean": "Connectivity ratio", "PDR (%)_mean": "PDR (%)", "Avg. end-to-end delay (ms)_mean": "Avg. end-to-end delay (ms)"}), figures_dir / "network_metrics.png")
    plot_publication_performance(metrics_df, figures_dir / "publication_performance_ci.png")
    plot_lead_time_summary(leads_df, figures_dir / "publication_lead_time_ci.png")
    plot_dataset_overview(dataset_summary_df, figures_dir / "dataset_overview.png")
    plot_radio_policy_sensitivity(dataset_summary_df, figures_dir / "radio_policy_sensitivity.png")
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
    }
    save_summary_json(out_dir / "summary.json", summary)
    write_markdown_report(out_dir / "report.md", summary, metrics_df, leads_df, network_df, risk_df)
    write_claims_summary(out_dir / "claims_summary.md", summary, metrics_df, leads_df, network_df, risk_df, stats_df, dataset_summary_df)
    export_manuscript_tables(out_dir / "manuscript_tables.tex", metrics_df, leads_df, network_df, risk_df, summary)
    export_manuscript_summary(out_dir / "manuscript_summary.json", summary, metrics_df, leads_df)
    write_artifact_manifest(out_dir / "artifact_manifest.txt", out_dir)
    print(f"[run] completed in {time.perf_counter() - run_start:.1f}s")
    return summary
