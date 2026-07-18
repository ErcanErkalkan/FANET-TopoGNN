from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import Snapshot, build_dataset, train_val_test_split
from fanet.evaluation import alert_event_metrics
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file
from fanet.source_gated import (
    FEATURE_GROUPS,
    SourceGatedKineticTopoGuard,
    _calibrated_predict,
    _positive_probability,
    build_source_matrices,
)
from fanet.training import _build_kinetic_topoguard_matrix, fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs/source_gated_development.json"
DEFAULT_OUTPUT = ROOT / "outputs/explainability"
PAPER_TABLE = ROOT / "paper/tables/generated/explainability_table.tex"
PAPER_FIGURE = ROOT / "paper/figures/generated/explainability.pdf"
MODELS = ("Kinetic-TopoGuard", "Source-Gated Kinetic-TopoGuard")
GROUPS = (
    "current state",
    "graph summaries",
    "persistence image bins",
    "persistence-image summaries",
    "current pair distances",
    "projected pair distances",
    "pair-distance deltas",
    "speed summaries",
    "projected fragmentation statistics",
)
OUTCOMES = ("event_f1", "count_mae")
LOCAL_CATEGORIES = ("true_positive_event", "false_positive_event", "false_negative_event", "true_negative_interval")


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "shap"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def split_without_leakage(config, snapshots: list[Snapshot]):
    stratify = list(config.sim.get("split_stratify_by", ["mobility"]))
    for field in ("graph_policy", "radio_scenario"):
        if len({getattr(snapshot, field) for snapshot in snapshots}) > 1 and field not in stratify:
            stratify.append(field)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(config.sim["split_seed"]),
        stratify_by=tuple(stratify),
    )
    intersections = validate_partition_run_ids(train, validation, test)
    return train, validation, test, stratify, intersections


def validate_partition_run_ids(train, validation, test) -> dict[str, list[str]]:
    run_sets = [{snapshot.run_id for snapshot in split} for split in (train, validation, test)]
    intersections = {
        "train_validation": sorted(run_sets[0] & run_sets[1]),
        "train_test": sorted(run_sets[0] & run_sets[2]),
        "validation_test": sorted(run_sets[1] & run_sets[2]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"run-level leakage: {intersections}")
    return intersections


def block_permutation_indices(run_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map complete run blocks to donor blocks of equal length."""
    run_ids = np.asarray(run_ids, dtype=object)
    result = np.arange(len(run_ids), dtype=int)
    runs = sorted(set(run_ids.astype(str)))
    by_length: dict[int, list[str]] = {}
    run_indices: dict[str, np.ndarray] = {}
    for run_id in runs:
        indices = np.flatnonzero(run_ids.astype(str) == run_id)
        run_indices[run_id] = indices
        by_length.setdefault(len(indices), []).append(run_id)
    for length, same_length_runs in sorted(by_length.items()):
        donors = list(same_length_runs)
        if len(donors) > 1:
            donors = list(np.asarray(donors, dtype=object)[rng.permutation(len(donors))])
            if all(source == donor for source, donor in zip(same_length_runs, donors)):
                donors = donors[1:] + donors[:1]
        for source, donor in zip(same_length_runs, donors):
            source_indices = run_indices[source]
            donor_indices = run_indices[str(donor)]
            if len(source_indices) != length or len(donor_indices) != length:
                raise RuntimeError("run-block lengths changed during permutation")
            result[source_indices] = donor_indices
    return result


def permute_columns_by_run_block(
    matrix: np.ndarray,
    run_ids: np.ndarray,
    columns: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    donor_indices = block_permutation_indices(run_ids, rng)
    result = np.asarray(matrix, dtype=float).copy()
    result[:, columns] = np.asarray(matrix, dtype=float)[donor_indices][:, columns]
    return result, donor_indices


def kinetic_group_indices(pi_bins: int = 256) -> dict[str, np.ndarray]:
    stats = np.arange(0, 17)
    degree = np.arange(17, 22)
    distances = np.arange(22, 26)
    pi_summary = np.arange(26, 31)
    pi = np.arange(31, 31 + pi_bins)
    shallow = np.arange(31 + pi_bins, 37 + pi_bins)
    kinetic = np.arange(37 + pi_bins, 83 + pi_bins)
    lag = np.arange(83 + pi_bins, 89 + pi_bins)
    groups = {
        "current state": np.concatenate([stats[[10]], kinetic[[0, 1, 2, 3, 38, 39, 40, 41, 42, 43, 44, 45]], lag]),
        "graph summaries": np.concatenate([stats[[0, 4, 5, 6, 7, 8, 9, 16]], degree, shallow[[0, 2, 3, 4, 5]]]),
        "persistence image bins": pi,
        "persistence-image summaries": pi_summary,
        "current pair distances": np.concatenate([stats[[1, 2, 3, 13]], distances, shallow[[1]], kinetic[np.arange(15, 21)]]),
        "projected pair distances": kinetic[np.arange(21, 27)],
        "pair-distance deltas": kinetic[np.arange(27, 33)],
        "speed summaries": np.concatenate([stats[[11, 12, 14, 15]], kinetic[np.arange(35, 38)]]),
        "projected fragmentation statistics": kinetic[np.concatenate([np.arange(4, 15), np.arange(33, 35)])],
    }
    flattened = np.concatenate(list(groups.values()))
    expected = np.arange(89 + pi_bins)
    if len(flattened) != len(expected) or not np.array_equal(np.sort(flattened), expected):
        raise RuntimeError("Kinetic-TopoGuard grouped feature layout is not exhaustive and disjoint")
    return {name: np.asarray(columns, dtype=int) for name, columns in groups.items()}


def source_group_layout(matrices: dict[str, np.ndarray]) -> dict[str, tuple[str, np.ndarray]]:
    return {
        "current state": ("current", np.arange(matrices["current"].shape[1])),
        "graph summaries": ("graph", np.arange(matrices["graph"].shape[1])),
        "persistence image bins": ("topology", np.arange(0, matrices["topology"].shape[1] - 5)),
        "persistence-image summaries": ("topology", np.arange(matrices["topology"].shape[1] - 5, matrices["topology"].shape[1])),
        "current pair distances": ("kinematic", np.arange(0, 5)),
        "projected pair distances": ("kinematic", np.arange(5, 10)),
        "pair-distance deltas": ("kinematic", np.arange(10, 15)),
        "speed summaries": ("kinematic", np.arange(15, 20)),
        "projected fragmentation statistics": ("kinematic", np.arange(20, 23)),
    }


def _metrics(aligned: list[Snapshot], count: np.ndarray, risk: np.ndarray, threshold: float, dt: float, horizon: int) -> dict[str, float]:
    event = alert_event_metrics(aligned, risk, threshold, dt, horizon)
    target = np.asarray([snapshot.beta_target for snapshot in aligned], dtype=float)
    return {"event_f1": float(event["Alert_Event_F1"]), "count_mae": float(np.mean(np.abs(target - count)))}


def _importance(baseline: dict[str, float], permuted: dict[str, float], outcome: str) -> float:
    return baseline[outcome] - permuted[outcome] if outcome == "event_f1" else permuted[outcome] - baseline[outcome]


def grouped_permutation_rows(
    *,
    seed: int,
    model_name: str,
    aligned: list[Snapshot],
    baseline_count: np.ndarray,
    baseline_risk: np.ndarray,
    threshold: float,
    dt: float,
    horizon: int,
    repeats: int,
    permuted_predictor: Callable[[str, np.random.Generator], tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    baseline = _metrics(aligned, baseline_count, baseline_risk, threshold, dt, horizon)
    values: dict[tuple[str, str], list[float]] = {(group, outcome): [] for group in GROUPS for outcome in OUTCOMES}
    for repeat in range(repeats):
        for group_index, group in enumerate(GROUPS):
            rng = np.random.default_rng(seed * 100_003 + repeat * 101 + group_index)
            count, risk = permuted_predictor(group, rng)
            metrics = _metrics(aligned, count, risk, threshold, dt, horizon)
            for outcome in OUTCOMES:
                values[(group, outcome)].append(_importance(baseline, metrics, outcome))
    rows = []
    for group in GROUPS:
        for outcome in OUTCOMES:
            current = np.asarray(values[(group, outcome)], dtype=float)
            rows.append({
                "seed": seed,
                "Model": model_name,
                "method": "grouped_permutation",
                "outcome": outcome,
                "feature_group": group,
                "importance_mean": float(current.mean()),
                "importance_median": float(np.median(current)),
                "importance_std": float(current.std(ddof=1)) if len(current) > 1 else 0.0,
                "repeats": repeats,
                "baseline_metric": baseline[outcome],
                "permutation_unit": "complete run block; donor run has equal length",
            })
    return rows


def _normalise_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"unexpected SHAP value shape: {array.shape}")
    return array


def tree_shap_rows(
    seed: int,
    kinetic_model,
    kinetic_x: np.ndarray,
    source_model: SourceGatedKineticTopoGuard,
    source_x: dict[str, np.ndarray],
    max_samples: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        import shap
    except Exception as exc:
        return [], {"available": False, "status": "not_installed_or_import_failed", "reason": repr(exc)}
    try:
        rows: list[dict[str, object]] = []
        sample_count = min(max_samples, len(kinetic_x))
        sample_indices = np.linspace(0, len(kinetic_x) - 1, sample_count, dtype=int)
        if not kinetic_model.classifiers:
            raise RuntimeError("Kinetic-TopoGuard has no fitted risk tree")
        kinetic_values = _normalise_shap_values(
            shap.TreeExplainer(kinetic_model.classifiers[0]).shap_values(kinetic_x[sample_indices])
        )
        for group, columns in kinetic_group_indices().items():
            rows.append({
                "seed": seed, "Model": "Kinetic-TopoGuard", "method": "TreeSHAP",
                "outcome": "selected_risk_tree_score", "feature_group": group,
                "importance_mean": float(np.mean(np.sum(np.abs(kinetic_values[:, columns]), axis=1))),
                "importance_median": float(np.median(np.sum(np.abs(kinetic_values[:, columns]), axis=1))),
                "importance_std": float(np.std(np.sum(np.abs(kinetic_values[:, columns]), axis=1), ddof=1)) if sample_count > 1 else 0.0,
                "repeats": 1, "baseline_metric": math.nan,
                "permutation_unit": "not applicable; TreeSHAP on deterministic sample",
            })

        layout = source_group_layout(source_x)
        source_values: dict[str, np.ndarray] = {}
        source_indices = np.linspace(0, len(next(iter(source_x.values()))) - 1, sample_count, dtype=int)
        for source in FEATURE_GROUPS:
            source_values[source] = _normalise_shap_values(
                shap.TreeExplainer(source_model.risk_base_models[source]).shap_values(source_x[source][source_indices])
            )
        coefficients = dict(zip(FEATURE_GROUPS, source_model.risk_meta_model.coef_[0]))
        for group, (source, columns) in layout.items():
            magnitude = np.sum(np.abs(source_values[source][:, columns]), axis=1) * abs(float(coefficients[source]))
            rows.append({
                "seed": seed, "Model": "Source-Gated Kinetic-TopoGuard", "method": "TreeSHAP",
                "outcome": "meta_weighted_base_risk_tree_score", "feature_group": group,
                "importance_mean": float(np.mean(magnitude)), "importance_median": float(np.median(magnitude)),
                "importance_std": float(np.std(magnitude, ddof=1)) if sample_count > 1 else 0.0,
                "repeats": 1, "baseline_metric": math.nan,
                "permutation_unit": "not applicable; TreeSHAP on deterministic sample",
            })
        return rows, {
            "available": True,
            "status": "completed",
            "version": importlib.metadata.version("shap"),
            "max_samples": sample_count,
            "scope": "selected Kinetic risk tree and meta-weighted Source-Gated base risk trees; grouped permutation remains the whole-model explanation",
        }
    except Exception as exc:
        return [], {"available": True, "status": "failed_without_aborting_grouped_permutation", "reason": repr(exc)}


def select_local_examples(aligned: list[Snapshot], scores: np.ndarray, threshold: float, horizon: int) -> dict[str, int | None]:
    selected: dict[str, int | None] = {category: None for category in LOCAL_CATEGORIES}
    run_pairs: dict[str, list[int]] = {}
    for index, snapshot in enumerate(aligned):
        run_pairs.setdefault(snapshot.run_id, []).append(index)
    for run_id in sorted(run_pairs):
        indices = sorted(run_pairs[run_id], key=lambda index: aligned[index].time_index)
        events = [pos for pos in range(1, len(indices)) if aligned[indices[pos - 1]].beta_current <= 1 and aligned[indices[pos]].beta_current > 1]
        alerts, last_alert, was_positive = [], -max(horizon, 1), False
        for pos, global_index in enumerate(indices):
            if aligned[global_index].beta_current > 1:
                was_positive = False
                continue
            positive = bool(scores[global_index] >= threshold)
            if positive and not was_positive and pos - last_alert >= max(horizon, 1):
                alerts.append(pos)
                last_alert = pos
            was_positive = positive
        matched: set[int] = set()
        for alert in alerts:
            match = next((event for event in events if event not in matched and alert < event <= alert + horizon), None)
            if match is None and selected["false_positive_event"] is None:
                selected["false_positive_event"] = indices[alert]
            elif match is not None:
                matched.add(match)
                if selected["true_positive_event"] is None:
                    selected["true_positive_event"] = indices[alert]
        for event in events:
            if event not in matched and selected["false_negative_event"] is None:
                selected["false_negative_event"] = indices[event]
        if selected["true_negative_interval"] is None:
            event_set = set(events)
            alert_set = set(alerts)
            for pos, global_index in enumerate(indices):
                future_event = any(event in event_set for event in range(pos + 1, min(pos + horizon + 1, len(indices))))
                if aligned[global_index].beta_current <= 1 and pos not in alert_set and not future_event and scores[global_index] < threshold:
                    selected["true_negative_interval"] = global_index
                    break
        if all(value is not None for value in selected.values()):
            break
    return selected


def _source_predict(model: SourceGatedKineticTopoGuard, matrices: dict[str, np.ndarray], aligned: list[Snapshot]):
    count_base, risk_base = model._base_predictions(matrices, model.count_base_models, model.risk_base_models)
    persistence = np.asarray([snapshot.beta_current for snapshot in aligned], dtype=float)
    count = np.clip(persistence + model.count_meta_model.predict(count_base), 1.0, None)
    raw = model.risk_meta_model.predict_proba(risk_base)[:, 1]
    probability = np.clip(_calibrated_predict(model.calibrator, raw), 0.0, 1.0)
    return count, probability, count_base, risk_base, raw


def local_explanation_rows(
    seed: int,
    model_name: str,
    aligned: list[Snapshot],
    baseline_risk: np.ndarray,
    threshold: float,
    horizon: int,
    local_predictor: Callable[[int, str], tuple[float, dict[str, object]]],
) -> list[dict[str, object]]:
    examples = select_local_examples(aligned, baseline_risk, threshold, horizon=horizon)
    rows = []
    for category in LOCAL_CATEGORIES:
        index = examples[category]
        if index is None:
            rows.append({"seed": seed, "Model": model_name, "example_type": category, "status": "unavailable", "feature_group": ""})
            continue
        snapshot = aligned[index]
        for group in GROUPS:
            perturbed_probability, details = local_predictor(index, group)
            rows.append({
                "seed": seed, "Model": model_name, "example_type": category, "status": "selected",
                "selection_rule": "lexicographically first run_id, then earliest eligible time index",
                "run_id": snapshot.run_id, "time_index": snapshot.time_index,
                "feature_group": group, "true_fragmentation_label": int(snapshot.frag_at_horizon),
                "selected_threshold": threshold, "calibrated_probability": float(baseline_risk[index]),
                "probability_after_group_reference": float(perturbed_probability),
                "local_probability_contribution": float(baseline_risk[index] - perturbed_probability),
                **details,
            })
    return rows


def run_seed(seed: int, config_path: Path, output_dir: Path, repeats: int, shap_max_samples: int):
    started = time.perf_counter()
    config = load_config(config_path)
    snapshots = build_dataset(config.sim, seed)
    train, validation, test, stratify, intersections = split_without_leakage(config, snapshots)
    horizon, dt = int(config.sim["forecast_horizon_steps"]), float(config.sim["dt"])

    kinetic = fit_kinetic_topoguard(train, validation, horizon, dt, seed).model
    source_path = ROOT / f"outputs/source_gated_development/per_seed/seed_{seed}/source_gated_model.pkl"
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen Source-Gated model missing: {source_path}")
    source = SourceGatedKineticTopoGuard.load(source_path)

    kinetic_x, kinetic_aligned, kinetic_scores = _build_kinetic_topoguard_matrix(test, horizon, dt)
    kinetic_count = kinetic._predict_from_matrix(kinetic_x, kinetic_aligned)
    kinetic_risk = kinetic._risk_from_matrix(kinetic_x, kinetic_aligned, kinetic_scores, kinetic_count)
    kinetic_before = (kinetic_count.copy(), kinetic_risk.copy())
    kinetic_run_ids = np.asarray([snapshot.run_id for snapshot in kinetic_aligned], dtype=object)
    kinetic_layout = kinetic_group_indices(pi_bins=int(config.sim["pi_resolution"]) ** 2)

    def permute_kinetic(group: str, rng: np.random.Generator):
        permuted_x, donors = permute_columns_by_run_block(kinetic_x, kinetic_run_ids, kinetic_layout[group], rng)
        permuted_kinetic = kinetic_scores[donors] if group == "projected fragmentation statistics" else kinetic_scores
        count = kinetic._predict_from_matrix(permuted_x, kinetic_aligned)
        risk = kinetic._risk_from_matrix(permuted_x, kinetic_aligned, permuted_kinetic, count)
        return count, risk

    importance = grouped_permutation_rows(
        seed=seed, model_name="Kinetic-TopoGuard", aligned=kinetic_aligned,
        baseline_count=kinetic_count, baseline_risk=kinetic_risk, threshold=kinetic.risk_threshold,
        dt=dt, horizon=horizon, repeats=repeats, permuted_predictor=permute_kinetic,
    )

    source_x, _, _, _, source_aligned = build_source_matrices(test, horizon, dt)
    source_count, source_risk, count_base, risk_base, raw_risk = _source_predict(source, source_x, source_aligned)
    source_before = (source_count.copy(), source_risk.copy())
    source_run_ids = np.asarray([snapshot.run_id for snapshot in source_aligned], dtype=object)
    source_layout = source_group_layout(source_x)

    def permute_source(group: str, rng: np.random.Generator):
        source_name, columns = source_layout[group]
        changed, _ = permute_columns_by_run_block(source_x[source_name], source_run_ids, columns, rng)
        matrices = {name: values if name != source_name else changed for name, values in source_x.items()}
        count, risk, _, _, _ = _source_predict(source, matrices, source_aligned)
        return count, risk

    importance.extend(grouped_permutation_rows(
        seed=seed, model_name="Source-Gated Kinetic-TopoGuard", aligned=source_aligned,
        baseline_count=source_count, baseline_risk=source_risk, threshold=source.risk_threshold,
        dt=dt, horizon=horizon, repeats=repeats, permuted_predictor=permute_source,
    ))
    shap_rows, shap_status = tree_shap_rows(seed, kinetic, kinetic_x, source, source_x, shap_max_samples)
    importance.extend(shap_rows)

    kinetic_reference = np.median(kinetic_x, axis=0)
    kinetic_score_reference = float(np.median(kinetic_scores))
    def local_kinetic(index: int, group: str):
        row = kinetic_x[index:index + 1].copy()
        row[:, kinetic_layout[group]] = kinetic_reference[kinetic_layout[group]]
        score = np.asarray([kinetic_score_reference if group == "projected fragmentation statistics" else kinetic_scores[index]])
        count = kinetic._predict_from_matrix(row, [kinetic_aligned[index]])
        risk = kinetic._risk_from_matrix(row, [kinetic_aligned[index]], score, count)
        return float(risk[0]), {"base_source_scores": "", "meta_coefficients": "", "raw_meta_probability": math.nan, "calibration_type": "none (Kinetic blend)"}

    local_rows = local_explanation_rows(seed, "Kinetic-TopoGuard", kinetic_aligned, kinetic_risk, kinetic.risk_threshold, horizon, local_kinetic)
    source_references = {name: np.median(values, axis=0) for name, values in source_x.items()}
    source_meta = source.artifact_metadata()
    def local_source(index: int, group: str):
        source_name, columns = source_layout[group]
        matrices = {name: values[index:index + 1].copy() for name, values in source_x.items()}
        matrices[source_name][:, columns] = source_references[source_name][columns]
        count, risk, _, perturbed_bases, raw = _source_predict(source, matrices, [source_aligned[index]])
        base_payload = dict(zip(FEATURE_GROUPS, risk_base[index].astype(float).tolist()))
        return float(risk[0]), {
            "base_source_scores": json.dumps(base_payload, sort_keys=True),
            "meta_coefficients": json.dumps(source_meta["risk_meta_coefficients"], sort_keys=True),
            "raw_meta_probability": float(raw_risk[index]),
            "calibration_type": source.calibration_type,
        }
    local_rows.extend(local_explanation_rows(seed, "Source-Gated Kinetic-TopoGuard", source_aligned, source_risk, source.risk_threshold, horizon, local_source))

    gate_rows = []
    for outcome, coefficients in (("count_residual", source_meta["count_meta_coefficients"]), ("fragmentation_risk", source_meta["risk_meta_coefficients"])):
        for feature_source, coefficient in coefficients.items():
            gate_rows.append({
                "seed": seed, "outcome": outcome, "feature_source": feature_source,
                "coefficient": float(coefficient), "coefficient_sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                "calibration_type": source.calibration_type, "selected_threshold": source.risk_threshold,
            })

    kinetic_after_count = kinetic._predict_from_matrix(kinetic_x, kinetic_aligned)
    kinetic_after_risk = kinetic._risk_from_matrix(kinetic_x, kinetic_aligned, kinetic_scores, kinetic_after_count)
    source_after = _source_predict(source, source_x, source_aligned)[:2]
    unchanged = bool(
        np.array_equal(kinetic_before[0], kinetic_after_count)
        and np.array_equal(kinetic_before[1], kinetic_after_risk)
        and np.array_equal(source_before[0], source_after[0])
        and np.array_equal(source_before[1], source_after[1])
    )
    if not unchanged:
        raise RuntimeError("model predictions changed during explanation")

    seed_dir = output_dir / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(importance).to_csv(seed_dir / "grouped_importance.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(seed_dir / "gate_coefficients.csv", index=False)
    pd.DataFrame(local_rows).to_csv(seed_dir / "local_explanations.csv", index=False)
    completion = {
        "seed": seed, "status": "complete", "runtime_seconds": time.perf_counter() - started,
        "split_stratify_by": stratify, "run_id_intersections": intersections,
        "explanation_partition": "test, after train/validation model freezing",
        "model_selection_uses_explanation": False, "predictions_unchanged": unchanged,
        "shap": shap_status, "permutation_repeats": repeats,
    }
    (seed_dir / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return pd.DataFrame(importance), pd.DataFrame(gate_rows), pd.DataFrame(local_rows), completion


def aggregate_importance(per_seed: pd.DataFrame, bootstrap_rounds: int = 2000) -> pd.DataFrame:
    rows = []
    keys = ["Model", "method", "outcome", "feature_group"]
    for key, group in per_seed.groupby(keys, sort=True):
        values = group["importance_mean"].to_numpy(dtype=float)
        rng = np.random.default_rng(20260715 + sum(ord(char) for char in "|".join(key)))
        boot = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(bootstrap_rounds)])
        per_seed_ranks = []
        for _, seed_frame in per_seed[
            (per_seed["Model"] == key[0]) & (per_seed["method"] == key[1]) & (per_seed["outcome"] == key[2])
        ].groupby("seed"):
            ranked = seed_frame.sort_values(["importance_mean", "feature_group"], ascending=[False, True]).reset_index(drop=True)
            match = ranked.index[ranked["feature_group"] == key[3]]
            if len(match):
                per_seed_ranks.append(int(match[0]) + 1)
        rank_std = float(np.std(per_seed_ranks, ddof=1)) if len(per_seed_ranks) > 1 else 0.0
        rows.append({
            **dict(zip(keys, key)), "seed_count": int(group["seed"].nunique()),
            "importance_mean": float(np.mean(values)), "importance_median": float(np.median(values)),
            "bootstrap_ci95_low": float(np.quantile(boot, 0.025)), "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
            "mean_rank": float(np.mean(per_seed_ranks)), "rank_std": rank_std,
            "rank_stability": float(np.clip(1.0 - rank_std / max(len(GROUPS) - 1, 1), 0.0, 1.0)),
            "top_rank_fraction": float(np.mean(np.asarray(per_seed_ranks) == 1)),
        })
    return pd.DataFrame(rows)


def write_table(summary: pd.DataFrame, path: Path) -> None:
    display = summary[(summary["method"] == "grouped_permutation") & (summary["outcome"] == "event_f1")].copy()
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Model & Feature group & Mean & 95\\% CI & Rank stability \\\\", "\\midrule"]
    for _, row in display.sort_values(["Model", "importance_mean"], ascending=[True, False]).iterrows():
        model = str(row["Model"]).replace("Source-Gated Kinetic-TopoGuard", "Source-Gated")
        feature = str(row["feature_group"]).replace("_", "\\_")
        lines.append(f"{model} & {feature} & {row['importance_mean']:.3f} & [{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] & {row['rank_stability']:.2f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_importance(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[(summary["method"] == "grouped_permutation") & (summary["outcome"] == "event_f1")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for axis, model in zip(axes, MODELS):
        current = frame[frame["Model"] == model].sort_values("importance_mean")
        errors = np.vstack([current["importance_mean"] - current["bootstrap_ci95_low"], current["bootstrap_ci95_high"] - current["importance_mean"]])
        axis.barh(current["feature_group"], current["importance_mean"], xerr=errors, color="#3b82f6" if model == MODELS[0] else "#f59e0b", alpha=0.85)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(model)
        axis.set_xlabel("Event-F1 drop after run-block permutation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_local(local: pd.DataFrame, path: Path) -> None:
    selected = local[local["status"] == "selected"].copy()
    grouped = selected.groupby(["Model", "example_type", "feature_group"], as_index=False)["local_probability_contribution"].mean()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for row_index, model in enumerate(MODELS):
        for column_index, category in enumerate(LOCAL_CATEGORIES):
            axis = axes[row_index, column_index]
            current = grouped[(grouped["Model"] == model) & (grouped["example_type"] == category)].sort_values("local_probability_contribution")
            axis.barh(current["feature_group"], current["local_probability_contribution"], color="#64748b")
            axis.axvline(0.0, color="black", linewidth=0.7)
            axis.set_title(f"{model}\n{category}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen-model explainability with run-block grouped permutation and optional TreeSHAP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--shap-max-samples", type=int, default=256)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path, output_dir = args.config.resolve(), args.output_dir.resolve()
    config = load_config(config_path)
    seeds = list(args.seeds if args.seeds else config.training["seed_list"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed list must be non-empty and unique")
    if args.workers < 1 or args.permutation_repeats < 1 or args.bootstrap_rounds < 1:
        raise ValueError("workers, repeats, and bootstrap rounds must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, gate_frames, local_frames, completions, pending = [], [], [], [], []
    for seed in seeds:
        seed_dir = output_dir / "per_seed" / f"seed_{seed}"
        paths = [seed_dir / name for name in ("grouped_importance.csv", "gate_coefficients.csv", "local_explanations.csv", "complete.json")]
        if args.resume and all(path.is_file() for path in paths):
            frames.append(pd.read_csv(paths[0])); gate_frames.append(pd.read_csv(paths[1])); local_frames.append(pd.read_csv(paths[2])); completions.append(json.loads(paths[3].read_text(encoding="utf-8")))
        else:
            pending.append(seed)
    if args.workers == 1:
        for seed in pending:
            result = run_seed(seed, config_path, output_dir, args.permutation_repeats, args.shap_max_samples)
            frames.append(result[0]); gate_frames.append(result[1]); local_frames.append(result[2]); completions.append(result[3])
    elif pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {executor.submit(run_seed, seed, config_path, output_dir, args.permutation_repeats, args.shap_max_samples): seed for seed in pending}
            for future in as_completed(futures):
                result = future.result()
                frames.append(result[0]); gate_frames.append(result[1]); local_frames.append(result[2]); completions.append(result[3])
                print(f"[complete] seed {futures[future]}")

    per_seed = pd.concat(frames, ignore_index=True).sort_values(["seed", "Model", "method", "outcome", "feature_group"])
    gates = pd.concat(gate_frames, ignore_index=True).sort_values(["seed", "outcome", "feature_source"])
    local = pd.concat(local_frames, ignore_index=True).sort_values(["seed", "Model", "example_type", "feature_group"])
    expected_permutation = len(seeds) * len(MODELS) * len(GROUPS) * len(OUTCOMES)
    actual_permutation = len(per_seed[per_seed["method"] == "grouped_permutation"])
    if actual_permutation != expected_permutation:
        raise RuntimeError(f"grouped permutation coverage mismatch: {actual_permutation} != {expected_permutation}")
    if not all(item["predictions_unchanged"] and not item["model_selection_uses_explanation"] for item in completions):
        raise RuntimeError("explanation mutation or model-selection leakage detected")
    summary = aggregate_importance(per_seed, args.bootstrap_rounds)
    per_seed.to_csv(output_dir / "grouped_importance_per_seed.csv", index=False)
    summary.to_csv(output_dir / "grouped_importance_summary.csv", index=False)
    gates.to_csv(output_dir / "gate_coefficients.csv", index=False)
    local.to_csv(output_dir / "local_explanations.csv", index=False)
    plot_importance(summary, output_dir / "feature_importance.pdf")
    plot_local(local, output_dir / "local_explanation_examples.pdf")
    write_table(summary, output_dir / "explainability_table.tex")
    shutil.copy2(output_dir / "feature_importance.pdf", PAPER_FIGURE)
    shutil.copy2(output_dir / "explainability_table.tex", PAPER_TABLE)

    source_paths = [config_path, ROOT / "configs/paper_like_submission.json", ROOT / "pyproject.toml", Path(__file__).resolve(), ROOT / "fanet/training.py", ROOT / "fanet/source_gated.py", ROOT / "fanet/evaluation.py", ROOT / "fanet/dataset.py"]
    source_paths.extend(ROOT / f"outputs/source_gated_development/per_seed/seed_{seed}/source_gated_model.pkl" for seed in seeds)
    protocol = {
        "schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started, "config": relative_repo_path(config_path, ROOT),
        "per_seed_runtime_seconds": {str(item["seed"]): float(item["runtime_seconds"]) for item in completions},
        "total_seed_runtime_seconds": float(sum(float(item["runtime_seconds"]) for item in completions)),
        "config_sha256": sha256_file(config_path), "seed_list": seeds, "split_seed": int(config.sim["split_seed"]),
        "feature_groups": list(GROUPS), "models": list(MODELS),
        "grouped_permutation": {"unit": "complete run block", "equal_length_donor_required": True, "repeats": args.permutation_repeats, "outcomes": list(OUTCOMES)},
        "local_example_rule": "lexicographically first run_id, then earliest eligible time index for TP event, FP event, FN event, and TN interval",
        "test_partition_role": "post-freeze explanation only; no feature, threshold, hyperparameter, calibration, or model selection",
        "model_selection_uses_explanation": False,
        "prediction_immutability_verified": all(item["predictions_unchanged"] for item in completions),
        "shap": {str(item["seed"]): item["shap"] for item in completions},
        "package_versions": package_versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "source_files": build_file_manifest(source_paths, ROOT),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote explainability artifacts for {len(seeds)} seed(s); grouped permutation rows={actual_permutation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
