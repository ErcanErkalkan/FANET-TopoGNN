from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.external_validation import build_measured_link_snapshots
from fanet.miluv_adaptation import (
    ChronologicalSplit,
    chronological_split,
    fit_few_shot_count_adapter,
    select_calibration_and_threshold,
    verify_sequence_source_manifest,
)
from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file
from fanet.source_gated import SourceGatedKineticTopoGuard, fit_current_state_extratrees
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard, select_best_shallow
from scripts.run_miluv_validation import (
    PAIRS,
    ROBOTS,
    _load_pair_measurements,
    _load_positions,
    _measured_graph_trace,
)


DEFAULT_INPUT_ROOT = ROOT / "data" / "external_validation" / "raw" / "miluv"
DEFAULT_OUTPUT = ROOT / "outputs" / "miluv_adaptation"
DEFAULT_CONFIG = ROOT / "configs" / "source_gated_development.json"
PRIMARY_FPP_DBM = -90.0
FPP_SENSITIVITY_DBM = (-90.0, -92.0, -95.0)
EDGE_AGE_GRID_S = (1.0, 3.0, 6.0)
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
THRESHOLD_GRID = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
MODEL_ORDER = (
    "Current-state persistence baseline",
    "Current-state ExtraTrees",
    "Kinetic-TopoGuard",
    "Source-Gated Kinetic-TopoGuard",
    "Shallow ML",
)


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch", "torch-geometric")
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _discover_local_sequences(input_root: Path) -> list[tuple[Path, dict]]:
    found = []
    for manifest_path in sorted(input_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_sequence_source_manifest(manifest, ROOT)
        experiment = str(manifest.get("experiment", manifest_path.parent.name))
        required = [manifest_path.parent / robot / name for robot in ROBOTS for name in ("mocap.csv", "uwb_range.csv")]
        if all(path.is_file() for path in required):
            found.append((manifest_path.parent, manifest))
    if not found:
        raise FileNotFoundError("no locally extracted, provenance-verified MILUV three-robot sequence is available")
    return found


def _sequence_record(sequence_dir: Path, manifest: dict, sample_period_s: float) -> dict:
    trace = _load_positions(sequence_dir, sample_period_s)
    measurements = _load_pair_measurements(sequence_dir)
    duration = max(float(trace.timestamps_s[-1] - trace.timestamps_s[0]), sample_period_s)
    coverage = {}
    for left, right in PAIRS:
        pair = measurements[(measurements.robot_left == left) & (measurements.robot_right == right)]
        coverage[f"pair_{left}_{right}"] = {
            "observation_count": int(len(pair)),
            "observation_rate_hz": float(len(pair) / duration),
            "first_timestamp_s": float(pair.timestamp.min()) if len(pair) else None,
            "last_timestamp_s": float(pair.timestamp.max()) if len(pair) else None,
        }
    return {
        "sequence": str(manifest["experiment"]),
        "archive_file_id": int(manifest["archive_file_id"]),
        "archive_name": str(manifest["archive_name"]),
        "archive_member_files": [
            {
                "archive_member": row["archive_member"],
                "zip_crc32": row["zip_crc32"],
                "sha256": row["sha256"],
                "relative_path": row["relative_path"],
            }
            for row in manifest["files"]
        ],
        "time_start_s": float(trace.timestamps_s[0]),
        "time_end_s": float(trace.timestamps_s[-1]),
        "sample_count": int(len(trace.timestamps_s)),
        "sample_period_s": float(sample_period_s),
        "edge_observation_coverage": coverage,
    }


def _snapshots(sequence_dir: Path, config, threshold_dbm: float, hold_s: float, sample_period_s: float):
    trace = _load_positions(sequence_dir, sample_period_s)
    measurements = _load_pair_measurements(sequence_dir)
    distances = []
    for positions in trace.positions_m:
        distances.extend(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)[np.triu_indices(3, 1)])
    radius_m = float(np.quantile(np.asarray(distances), 0.95))
    selected, adjacency, frame = _measured_graph_trace(trace, measurements, threshold_dbm, hold_s)
    snapshots = build_measured_link_snapshots(
        selected,
        adjacency,
        radius_m=radius_m,
        horizon_steps=int(config.sim["forecast_horizon_steps"]),
        pi_resolution=int(config.sim["pi_resolution"]),
        pi_sigma=float(config.sim["pi_sigma"]),
        pi_max_radius=float(config.sim["pi_max_radius"]),
        run_label=sequence_dir.name,
        radio_scenario=f"miluv_fpp_{abs(int(threshold_dbm))}_hold_{hold_s:g}",
    )
    return snapshots, frame


def _models(seed: int, config, train, validation):
    horizon = int(config.sim["forecast_horizon_steps"])
    dt = float(config.sim["dt"])
    source_path = ROOT / "outputs" / "source_gated_development" / "per_seed" / f"seed_{seed}" / "source_gated_model.pkl"
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen Source-Gated artifact is missing: {relative_repo_path(source_path, ROOT)}")
    source = SourceGatedKineticTopoGuard.load(source_path)
    shallow = select_best_shallow(train, validation)
    shallow.model_name = "Shallow ML"
    return [
        fit_current_state_persistence(),
        fit_current_state_extratrees(train, validation, horizon, dt, seed),
        fit_kinetic_topoguard(train, validation, horizon, dt, seed),
        SimpleNamespace(model=source, model_name="Source-Gated Kinetic-TopoGuard", inference_ms=0.0),
        shallow,
    ], source_path


def _evaluate(
    model_name: str,
    snapshots: list,
    pred: np.ndarray,
    risk: np.ndarray,
    latency: np.ndarray,
    split: ChronologicalSplit,
    threshold: float,
    sample_period_s: float,
    horizon: int,
    bootstrap_rounds: int,
) -> dict:
    idx = split.test.astype(int)
    aligned = [snapshots[i] for i in idx]
    metrics, _, _ = evaluate_predictions(
        model_name,
        aligned,
        np.asarray(pred)[idx],
        np.asarray(risk)[idx],
        np.asarray(latency)[idx],
        dt=sample_period_s,
        bootstrap_rounds=bootstrap_rounds,
        horizon_steps=horizon,
        risk_threshold=float(threshold),
    )
    return metrics


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ("MAE", "Alert_Event_Precision", "Alert_Event_Recall", "Alert_Event_F1", "False_Alert_Events_per_minute", "Risk_Brier", "Risk_ECE")
    rows = []
    keys = ["Protocol", "Model", "FPP_Threshold_dBm", "primary_fpp_threshold"]
    for values, group in metrics.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row["replicate_count"] = int(len(group))
        row["seed_count"] = int(group.Seed.nunique())
        row["scenario_count"] = int(group.Sequence.nunique())
        for column in columns:
            data = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.nanmean(data))
            row[f"{column}_std"] = float(np.nanstd(data, ddof=1)) if len(data) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_table(summary: pd.DataFrame, destination: Path) -> None:
    primary = summary[summary.primary_fpp_threshold.astype(bool)].copy()
    rows = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Protocol & Model & Seeds & MAE & Event F1 & Brier \\", r"\midrule"]
    protocol_labels = {
        "zero_shot_frozen": "Zero-shot frozen",
        "chronological_calibration": "Chronological calibration",
        "few_shot_prediction_head": "Few-shot prediction head",
    }
    for item in primary.sort_values(["Protocol", "Model"]).itertuples():
        label = protocol_labels.get(str(item.Protocol), str(item.Protocol).replace("_", " ").title())
        model = str(item.Model).replace("_", r"\_")
        rows.append(
            f"{label} & {model} & {int(item.seed_count)} & {item.MAE_mean:.3f} & "
            f"{item.Alert_Event_F1_mean:.3f} & {item.Risk_Brier_mean:.3f} " + r"\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-free MILUV chronological calibration and few-shot adaptation study.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--sample-period-s", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--guard-seconds", type=float, default=2.0)
    parser.add_argument("--few-shot-labels", type=int, default=32)
    parser.add_argument("--minimum-calibration-rows", type=int, default=30)
    parser.add_argument("--minimum-test-rows", type=int, default=100)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing scientific output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    horizon = int(config.sim["forecast_horizon_steps"])
    if args.guard_seconds < horizon * args.sample_period_s:
        raise ValueError("guard interval must be at least the forecast horizon")
    local_sequences = _discover_local_sequences(args.input_root)
    sequence_records = [_sequence_record(path, manifest, args.sample_period_s) for path, manifest in local_sequences]
    (args.output_dir / "sequence_manifest.json").write_text(
        json.dumps({"sequence_count": len(sequence_records), "sequences": sequence_records}, indent=2), encoding="utf-8"
    )

    rows, splits, calibration_rows, curve_rows = [], [], [], []
    source_artifacts = []
    for seed in args.seeds:
        synthetic = build_dataset(config.sim, seed=int(seed))
        train, validation, _ = train_val_test_split(
            synthetic,
            split_seed=int(config.sim["split_seed"]),
            stratify_by=tuple(config.sim.get("split_stratify_by", ["mobility"])),
        )
        models, source_path = _models(int(seed), config, train, validation)
        source_artifacts.append(source_path)
        for sequence_dir, _manifest in local_sequences:
            measured = {}
            unavailable_measured = {}
            for threshold in FPP_SENSITIVITY_DBM:
                holds = EDGE_AGE_GRID_S if threshold == PRIMARY_FPP_DBM else (EDGE_AGE_GRID_S[-1],)
                for hold in holds:
                    snapshots, frame = _snapshots(sequence_dir, config, threshold, hold, args.sample_period_s)
                    timestamps = np.asarray([float(frame.timestamp_s.iloc[item.time_index]) for item in snapshots])
                    try:
                        split = chronological_split(timestamps, args.calibration_fraction, args.guard_seconds)
                    except ValueError as exc:
                        unavailable_measured[(threshold, hold)] = str(exc)
                        continue
                    if len(split.calibration) < int(args.minimum_calibration_rows):
                        unavailable_measured[(threshold, hold)] = (
                            f"calibration segment has {len(split.calibration)} rows; "
                            f"minimum is {args.minimum_calibration_rows}"
                        )
                        continue
                    if len(split.test) < int(args.minimum_test_rows):
                        unavailable_measured[(threshold, hold)] = (
                            f"independent test segment has {len(split.test)} rows; "
                            f"minimum is {args.minimum_test_rows}"
                        )
                        continue
                    measured[(threshold, hold)] = (snapshots, split)
            required_zero_shot = [(threshold, EDGE_AGE_GRID_S[-1]) for threshold in FPP_SENSITIVITY_DBM]
            missing_required = [key for key in required_zero_shot if key not in measured]
            if missing_required:
                raise RuntimeError(f"primary frozen MILUV protocol has no valid chronological split: {missing_required}")
            for result in models:
                predictions = {}
                for key, (snapshots, split) in measured.items():
                    pred, risk, latency, aligned, frozen_threshold = predict_generic(result, snapshots)
                    if len(aligned) != len(snapshots):
                        raise RuntimeError(f"unexpected MILUV prediction alignment for {result.model_name}")
                    predictions[key] = (np.clip(pred, 1.0, 3.0), np.clip(risk, 0.0, 1.0), latency, frozen_threshold)

                # A: frozen zero-shot, including the predeclared FPP sensitivity grid.
                for threshold in FPP_SENSITIVITY_DBM:
                    key = (threshold, EDGE_AGE_GRID_S[-1])
                    snapshots, split = measured[key]
                    pred, risk, latency, frozen_threshold = predictions[key]
                    metrics = _evaluate(result.model_name, snapshots, pred, risk, latency, split, frozen_threshold, args.sample_period_s, horizon, args.bootstrap_rounds)
                    metrics.update({"Seed": seed, "Sequence": sequence_dir.name, "Protocol": "zero_shot_frozen", "FPP_Threshold_dBm": threshold, "primary_fpp_threshold": threshold == PRIMARY_FPP_DBM, "Selected_Edge_Age_s": EDGE_AGE_GRID_S[-1], "Selected_Calibration": "frozen_simulation", "Selected_Risk_Threshold": frozen_threshold, "Adaptation_Label_Count": 0})
                    rows.append(metrics)

                # B: frozen base weights; age, calibrator and alert threshold selected on early calibration only.
                age_candidates = []
                for hold in EDGE_AGE_GRID_S:
                    key = (PRIMARY_FPP_DBM, hold)
                    if key not in measured:
                        calibration_rows.append({"Seed": seed, "Sequence": sequence_dir.name, "Model": result.model_name, "Protocol": "chronological_calibration", "Edge_Age_s": hold, "calibration": None, "available": False, "reason": unavailable_measured.get(key, "measured sequence unavailable")})
                        continue
                    snapshots, split = measured[key]
                    pred, risk, latency, _ = predictions[key]
                    labels = np.asarray([snap.frag_at_horizon for snap in snapshots], dtype=int)
                    calibrator, selected_threshold, selected, candidates = select_calibration_and_threshold(
                        risk, labels, snapshots, split, CALIBRATION_METHODS, THRESHOLD_GRID,
                        args.sample_period_s, horizon, int(seed),
                    )
                    for candidate in candidates:
                        calibration_rows.append({"Seed": seed, "Sequence": sequence_dir.name, "Model": result.model_name, "Protocol": "chronological_calibration", "Edge_Age_s": hold, **{k: v for k, v in candidate.items() if k != "fit_indices"}})
                    age_candidates.append((selected["calibration_brier"], selected["calibration_ece"], -selected["calibration_event_f1"], hold, calibrator, selected_threshold, selected))
                if not age_candidates:
                    raise RuntimeError(f"no edge-age candidate supports calibration, guard, and test for {sequence_dir.name}")
                _, _, _, hold, calibrator, selected_threshold, selected = min(age_candidates, key=lambda item: item[:4])
                snapshots, split = measured[(PRIMARY_FPP_DBM, hold)]
                pred, risk, latency, _ = predictions[(PRIMARY_FPP_DBM, hold)]
                calibrated = calibrator.predict(risk)
                metrics = _evaluate(result.model_name, snapshots, pred, calibrated, latency, split, selected_threshold, args.sample_period_s, horizon, args.bootstrap_rounds)
                metrics.update({"Seed": seed, "Sequence": sequence_dir.name, "Protocol": "chronological_calibration", "FPP_Threshold_dBm": PRIMARY_FPP_DBM, "primary_fpp_threshold": True, "Selected_Edge_Age_s": hold, "Selected_Calibration": selected["calibration"], "Selected_Risk_Threshold": selected_threshold, "Adaptation_Label_Count": int(len(split.calibration))})
                rows.append(metrics)

                # D: prediction-head adaptation on a predeclared small prefix; base model weights stay frozen.
                few = split.calibration[: min(int(args.few_shot_labels), len(split.calibration))]
                few_calibration_end_s = float(snapshots[int(few[-1])].time_index) * args.sample_period_s
                few_split = ChronologicalSplit(
                    few,
                    np.unique(np.r_[split.calibration[len(few):], split.guard]),
                    split.test,
                    few_calibration_end_s,
                    split.test_start_s,
                )
                labels = np.asarray([snap.frag_at_horizon for snap in snapshots], dtype=int)
                calibrator, selected_threshold, selected, candidates = select_calibration_and_threshold(
                    risk, labels, snapshots, few_split, CALIBRATION_METHODS, THRESHOLD_GRID,
                    args.sample_period_s, horizon, int(seed),
                )
                targets = np.asarray([snap.beta_target for snap in snapshots], dtype=float)
                count_adapter = fit_few_shot_count_adapter(pred, targets, few, few_split)
                adapted_pred = np.clip(count_adapter.predict(pred.reshape(-1, 1)), 1.0, 3.0)
                adapted_risk = calibrator.predict(risk)
                metrics = _evaluate(result.model_name, snapshots, adapted_pred, adapted_risk, latency, few_split, selected_threshold, args.sample_period_s, horizon, args.bootstrap_rounds)
                metrics.update({"Seed": seed, "Sequence": sequence_dir.name, "Protocol": "few_shot_prediction_head", "FPP_Threshold_dBm": PRIMARY_FPP_DBM, "primary_fpp_threshold": True, "Selected_Edge_Age_s": hold, "Selected_Calibration": selected["calibration"], "Selected_Risk_Threshold": selected_threshold, "Adaptation_Label_Count": int(len(few))})
                rows.append(metrics)

                for protocol_name, used_split in (("chronological_calibration", split), ("few_shot_prediction_head", few_split)):
                    splits.append({"Seed": seed, "Sequence": sequence_dir.name, "Model": result.model_name, "Protocol": protocol_name, "calibration_rows": int(len(used_split.calibration)), "guard_rows": int(len(used_split.guard)), "test_rows": int(len(used_split.test)), "calibration_end_s": used_split.calibration_end_s, "test_start_s": used_split.test_start_s, "guard_seconds": float(used_split.test_start_s - used_split.calibration_end_s), "test_used_for_selection": False})
                for score, label in zip(adapted_risk[few_split.test], labels[few_split.test]):
                    curve_rows.append({"Protocol": "few_shot_prediction_head", "score": float(score), "label": int(label)})
        print(f"completed MILUV adaptation seed={seed}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = _aggregate(metrics)
    metrics.to_csv(args.output_dir / "per_seed_or_scenario_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(splits).to_csv(args.output_dir / "split_manifest.csv", index=False)

    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    curves = pd.DataFrame(curve_rows)
    if not curves.empty:
        curves["bin"] = pd.cut(curves.score, np.linspace(0, 1, 11), include_lowest=True)
        plotted = curves.groupby("bin", observed=True).agg(predicted=("score", "mean"), observed=("label", "mean"))
        axis.plot(plotted.predicted, plotted.observed, "o-", label="few-shot test (frozen after prefix)")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    axis.set(xlabel="Predicted fragmentation probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "calibration_curves.pdf")
    plt.close(fig)

    existing_primary = ROOT / "outputs" / "miluv_validation" / "miluv_metrics_per_seed.csv"
    decision = {
        "existing_zero_shot_primary_evidence_preserved": existing_primary.is_file(),
        "existing_zero_shot_primary_artifact": relative_repo_path(existing_primary, ROOT),
        "adaptation_study_role": "secondary domain-shift analysis; not deployment readiness evidence",
        "leave_one_scenario_out": {"status": "available" if len(local_sequences) > 1 else "unavailable", "reason": None if len(local_sequences) > 1 else "only one locally extracted and provenance-verified compatible three-robot sequence"},
        "test_used_for_model_calibration_threshold_or_staleness_selection": False,
        "claims_excluded": ["deployment-ready", "IP packet delivery", "outdoor FANET transfer"],
    }
    (args.output_dir / "adaptation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "MILUV leakage-free adaptation/calibration",
        "primary_evidence": "existing frozen zero-shot transfer remains primary",
        "existing_primary_artifact_sha256": sha256_file(existing_primary) if existing_primary.is_file() else None,
        "seeds": [int(seed) for seed in args.seeds],
        "split": {"type": "chronological prefix / guard / independent suffix", "calibration_fraction": args.calibration_fraction, "guard_seconds": args.guard_seconds, "minimum_calibration_rows": int(args.minimum_calibration_rows), "minimum_test_rows": int(args.minimum_test_rows), "random_row_split": False},
        "protocols": {"A": "zero_shot_frozen", "B": "chronological_calibration", "C": decision["leave_one_scenario_out"], "D": "few_shot_prediction_head"},
        "base_model_weights_frozen_on_miluv": True,
        "few_shot_label_budget": int(args.few_shot_labels),
        "primary_fpp_threshold_dbm": PRIMARY_FPP_DBM,
        "fpp_threshold_sensitivity_dbm": list(FPP_SENSITIVITY_DBM),
        "edge_age_grid_seconds": list(EDGE_AGE_GRID_S),
        "edge_age_selection_split": "calibration only",
        "calibration_methods": list(CALIBRATION_METHODS),
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection_split": "calibration only",
        "models": list(MODEL_ORDER),
        "metrics": ["count MAE", "event precision", "event recall", "event F1", "false events/min", "Brier", "ECE"],
        "training_config": relative_repo_path(args.config, ROOT),
        "source_manifests": build_file_manifest([path / "manifest.json" for path, _ in local_sequences], ROOT),
        "model_artifacts": build_file_manifest(sorted(set(source_artifacts)), ROOT),
        "source_files": build_file_manifest([Path(__file__).resolve(), ROOT / "fanet" / "miluv_adaptation.py", args.config], ROOT),
        "git_commit": _git_commit(),
        "package_versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    table = ROOT / "paper" / "tables" / "generated" / "miluv_adaptation_table.tex"
    _write_table(summary, table)
    print(f"wrote {relative_repo_path(args.output_dir, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
