from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.external_validation import FlightTrace, build_measured_link_snapshots
from fanet.provenance import build_file_manifest, relative_repo_path, verify_manifest
from fanet.training import (
    fit_current_state_persistence,
    fit_kinetic_topoguard,
    select_best_shallow,
)


EXPERIMENT = "cirObstacles_3_random_0"
DEFAULT_INPUT = ROOT / "data" / "external_validation" / "raw" / "miluv" / EXPERIMENT
DEFAULT_OUTPUT = ROOT / "outputs" / "miluv_validation"
DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
ROBOTS = ("ifo001", "ifo002", "ifo003")
PAIRS = ((1, 2), (1, 3), (2, 3))


def _longest_true_slice(mask: np.ndarray) -> slice:
    best_start = best_stop = start = 0
    for index, value in enumerate(np.r_[mask, False]):
        if value:
            continue
        if index - start > best_stop - best_start:
            best_start, best_stop = start, index
        start = index + 1
    if best_stop <= best_start:
        raise RuntimeError("MILUV trace has no contiguous interval with all pair measurements")
    return slice(best_start, best_stop)


def _load_positions(input_dir: Path, sample_period_s: float) -> FlightTrace:
    frames = {}
    starts = []
    stops = []
    for robot in ROBOTS:
        frame = pd.read_csv(input_dir / robot / "mocap.csv").sort_values("timestamp")
        frames[robot] = frame
        starts.append(float(frame["timestamp"].min()))
        stops.append(float(frame["timestamp"].max()))
    start = np.ceil(max(starts) / sample_period_s) * sample_period_s
    stop = np.floor(min(stops) / sample_period_s) * sample_period_s
    timestamps = np.arange(start, stop + sample_period_s * 0.5, sample_period_s)
    positions = np.empty((len(timestamps), len(ROBOTS), 3), dtype=np.float32)
    columns = ["pose.position.x", "pose.position.y", "pose.position.z"]
    for robot_index, robot in enumerate(ROBOTS):
        frame = frames[robot]
        for axis, column in enumerate(columns):
            positions[:, robot_index, axis] = np.interp(
                timestamps,
                frame["timestamp"].to_numpy(dtype=float),
                frame[column].to_numpy(dtype=float),
            )
    return FlightTrace(timestamps, ROBOTS, positions)


def _load_pair_measurements(input_dir: Path) -> pd.DataFrame:
    frames = []
    for robot in ROBOTS:
        frame = pd.read_csv(input_dir / robot / "uwb_range.csv")
        frame["recorder"] = robot
        frames.append(frame)
    work = pd.concat(frames, ignore_index=True)
    work = work.drop_duplicates(subset=["from_id", "to_id", "timestamp"])
    work = work[
        (work["from_id"] >= 10)
        & (work["to_id"] >= 10)
        & ((work["from_id"] // 10) != (work["to_id"] // 10))
    ].copy()
    work["robot_left"] = np.minimum(work["from_id"] // 10, work["to_id"] // 10).astype(int)
    work["robot_right"] = np.maximum(work["from_id"] // 10, work["to_id"] // 10).astype(int)
    work["first_path_power_dbm"] = work[["fpp1", "fpp2"]].min(axis=1)
    return (
        work.groupby(["robot_left", "robot_right", "timestamp"], as_index=False)
        .agg(
            first_path_power_dbm=("first_path_power_dbm", "max"),
            ground_truth_range_m=("gt_range", "mean"),
        )
        .sort_values("timestamp")
    )


def _measured_graph_trace(
    trace: FlightTrace,
    measurements: pd.DataFrame,
    threshold_dbm: float,
    max_hold_s: float,
) -> tuple[FlightTrace, np.ndarray, pd.DataFrame]:
    timestamps = trace.timestamps_s.astype(float)
    pair_states = []
    pair_ages = []
    pair_powers = []
    for left, right in PAIRS:
        pair = measurements[
            (measurements["robot_left"] == left)
            & (measurements["robot_right"] == right)
        ].sort_values("timestamp")
        observed_time = pair["timestamp"].to_numpy(dtype=float)
        observed_power = pair["first_path_power_dbm"].to_numpy(dtype=float)
        indices = np.searchsorted(observed_time, timestamps, side="right") - 1
        valid = indices >= 0
        power = np.full(len(timestamps), np.nan)
        age = np.full(len(timestamps), np.inf)
        power[valid] = observed_power[indices[valid]]
        age[valid] = timestamps[valid] - observed_time[indices[valid]]
        pair_powers.append(power)
        pair_ages.append(age)
        pair_states.append((power >= threshold_dbm) & (age <= max_hold_s))
    known = np.all(np.asarray(pair_ages) <= max_hold_s, axis=0)
    keep = _longest_true_slice(known)
    selected_trace = FlightTrace(
        trace.timestamps_s[keep],
        trace.vehicle_ids,
        trace.positions_m[keep],
    )
    states = np.asarray(pair_states)[:, keep]
    adjacencies = np.zeros(
        (states.shape[1], len(ROBOTS), len(ROBOTS)),
        dtype=np.float32,
    )
    for pair_index, (left, right) in enumerate(PAIRS):
        adjacencies[:, left - 1, right - 1] = states[pair_index]
        adjacencies[:, right - 1, left - 1] = states[pair_index]
    frame = pd.DataFrame(
        {
            "timestamp_s": selected_trace.timestamps_s,
            "threshold_dbm": threshold_dbm,
        }
    )
    for pair_index, (left, right) in enumerate(PAIRS):
        frame[f"pair_{left}_{right}_active"] = states[pair_index].astype(int)
        frame[f"pair_{left}_{right}_power_dbm"] = np.asarray(pair_powers)[pair_index, keep]
        frame[f"pair_{left}_{right}_age_s"] = np.asarray(pair_ages)[pair_index, keep]
    frame["beta0"] = [
        1 if int(state.sum()) >= 2 else 2 if int(state.sum()) == 1 else 3
        for state in states.T
    ]
    return selected_trace, adjacencies, frame


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["Model"] = work["Model"].where(
        ~work["Model"].str.startswith("Shallow ML"),
        "Shallow ML",
    )
    metrics = [
        "MAE",
        "Raw_MAE",
        "Prediction_Clipped_Fraction",
        "Risk_F1",
        "Risk_PR_AUC",
        "Risk_Brier",
        "Alert_Event_Precision",
        "Alert_Event_Recall",
        "Alert_Event_F1",
        "False_Alert_Events_per_minute",
    ]
    grouped = work.groupby(
        ["Model", "FPP_Threshold_dBm", "Snapshots", "Fragmentation_Events"],
        as_index=False,
    )
    means = grouped[metrics].mean().rename(
        columns={column: f"{column}_mean" for column in metrics}
    )
    stds = grouped[metrics].std(ddof=1).fillna(0.0).rename(
        columns={column: f"{column}_std" for column in metrics}
    )
    return means.merge(
        stds,
        on=["Model", "FPP_Threshold_dBm", "Snapshots", "Fragmentation_Events"],
    )


def _write_table(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & FPP threshold & Events & MAE & Risk PR-AUC & Event F1 \\",
        r"\midrule",
    ]
    for item in summary.sort_values(["FPP_Threshold_dBm", "Model"]).itertuples():
        rows.append(
            f"{item.Model} & {item.FPP_Threshold_dBm:.0f} dBm & "
            f"{int(item.Fragmentation_Events)} & {item.MAE_mean:.3f} & "
            f"{item.Risk_PR_AUC_mean:.3f} & {item.Alert_Event_F1_mean:.3f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen simulation models on measured MILUV UWB topology.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--thresholds-dbm", type=float, nargs="+", default=[-90.0, -92.0, -95.0])
    parser.add_argument("--sample-period-s", type=float, default=0.1)
    parser.add_argument("--max-hold-s", type=float, default=6.0)
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Verify source provenance and refresh only the existing protocol JSON.",
    )
    args = parser.parse_args()

    manifest_path = args.input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"MILUV source manifest is missing: {relative_repo_path(manifest_path, ROOT)}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_verification = verify_manifest(manifest, ROOT)
    if not source_verification["valid"]:
        raise RuntimeError(
            "MILUV source provenance verification failed:\n"
            + "\n".join(source_verification["errors"])
        )
    if args.protocol_only:
        protocol_path = args.output_dir / "miluv_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"Cannot refresh missing protocol: {relative_repo_path(protocol_path, ROOT)}"
            )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol.update(
            {
                "source_manifest": relative_repo_path(manifest_path, ROOT),
                "source_files": build_file_manifest([manifest_path], ROOT),
                "training_config": relative_repo_path(args.config, ROOT),
            }
        )
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        print(f"Refreshed {relative_repo_path(protocol_path, ROOT)}")
        return 0
    config = load_config(args.config)
    horizon_steps = int(config.sim["forecast_horizon_steps"])
    trace = _load_positions(args.input_dir, args.sample_period_s)
    measurements = _load_pair_measurements(args.input_dir)
    pair_distances = []
    for positions in trace.positions_m:
        pair_distances.extend(
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)[
                np.triu_indices(len(ROBOTS), 1)
            ]
        )
    radius_m = float(np.quantile(np.asarray(pair_distances), 0.95))

    measured_sets = {}
    trace_rows = []
    for threshold in args.thresholds_dbm:
        selected_trace, adjacencies, frame = _measured_graph_trace(
            trace,
            measurements,
            threshold,
            args.max_hold_s,
        )
        measured_sets[threshold] = build_measured_link_snapshots(
            selected_trace,
            adjacencies,
            radius_m=radius_m,
            horizon_steps=horizon_steps,
            pi_resolution=int(config.sim["pi_resolution"]),
            pi_sigma=float(config.sim["pi_sigma"]),
            pi_max_radius=float(config.sim["pi_max_radius"]),
            run_label=EXPERIMENT,
            radio_scenario=f"miluv_fpp_{abs(int(threshold))}",
        )
        trace_rows.append(frame)

    rows = []
    predictions = []
    for seed in args.seeds:
        synthetic = build_dataset(config.sim, seed=seed)
        train, val, _ = train_val_test_split(
            synthetic,
            split_seed=int(config.sim["split_seed"]),
            stratify_by=tuple(config.sim.get("split_stratify_by", ["mobility"])),
        )
        models = [
            fit_current_state_persistence(),
            select_best_shallow(train, val),
            fit_kinetic_topoguard(
                train,
                val,
                horizon_steps,
                float(config.sim["dt"]),
                seed,
            ),
        ]
        for threshold, snapshots in measured_sets.items():
            event_count = sum(
                snapshots[index - 1].beta_current <= 1.0
                and snapshots[index].beta_current > 1.0
                for index in range(1, len(snapshots))
            )
            for result in models:
                pred, risk, latency, aligned, risk_threshold = predict_generic(
                    result,
                    snapshots,
                )
                raw_pred = np.asarray(pred, dtype=float)
                clipped = np.clip(raw_pred, 1.0, len(ROBOTS))
                target = np.asarray([item.beta_target for item in aligned], dtype=float)
                metrics, _, _ = evaluate_predictions(
                    result.model_name,
                    aligned,
                    clipped,
                    risk,
                    latency,
                    dt=args.sample_period_s,
                    bootstrap_rounds=int(config.evaluation["bootstrap_rounds"]),
                    horizon_steps=horizon_steps,
                    risk_threshold=risk_threshold,
                )
                metrics.update(
                    {
                        "Seed": seed,
                        "FPP_Threshold_dBm": threshold,
                        "Snapshots": len(aligned),
                        "Fragmentation_Events": event_count,
                        "Raw_MAE": float(np.mean(np.abs(target - raw_pred))),
                        "Prediction_Clipped_Fraction": float(np.mean(raw_pred != clipped)),
                    }
                )
                rows.append(metrics)
                predictions.extend(
                    {
                        "Seed": seed,
                        "Model": result.model_name,
                        "FPP_Threshold_dBm": threshold,
                        "time_index": item.time_index,
                        "beta_current": item.beta_current,
                        "beta_target": item.beta_target,
                        "prediction_raw": float(raw),
                        "prediction_clipped": float(clip),
                        "risk_score": float(score),
                        "risk_threshold": float(risk_threshold),
                    }
                    for item, raw, clip, score in zip(aligned, raw_pred, clipped, risk)
                )
        print(f"completed MILUV seed={seed}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics = pd.DataFrame(rows)
    summary = _aggregate(raw_metrics)
    raw_metrics.to_csv(args.output_dir / "miluv_metrics_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "miluv_metrics_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "miluv_predictions.csv", index=False)
    pd.concat(trace_rows, ignore_index=True).to_csv(
        args.output_dir / "miluv_measured_topology_trace.csv",
        index=False,
    )
    _write_table(summary, args.output_dir / "miluv_validation_table.tex")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for robot_index, robot in enumerate(trace.vehicle_ids):
        axes[0].plot(
            trace.positions_m[:, robot_index, 0],
            trace.positions_m[:, robot_index, 1],
            label=robot,
        )
    primary_trace = trace_rows[0]
    axes[1].step(
        primary_trace["timestamp_s"],
        primary_trace["beta0"],
        where="post",
    )
    axes[0].set(xlabel="Vicon x (m)", ylabel="Vicon y (m)", title="Measured three-UAV motion")
    axes[0].axis("equal")
    axes[0].legend(frameon=False)
    axes[1].set(
        xlabel="Time (s)",
        ylabel=r"Measured-link $\beta_0$",
        title=f"UWB topology at {args.thresholds_dbm[0]:.0f} dBm",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "miluv_validation.png", dpi=220)
    fig.savefig(args.output_dir / "miluv_validation.pdf")
    plt.close(fig)

    protocol = {
        "source_manifest": relative_repo_path(manifest_path, ROOT),
        "source_files": build_file_manifest([manifest_path], ROOT),
        "training_config": relative_repo_path(args.config, ROOT),
        "dataset_doi": manifest["dataset_doi"],
        "experiment": EXPERIMENT,
        "scope": "Measured three-UAV motion and inter-robot UWB first-path-power topology; successful ranging quality, not IP packet-delivery or routing ground truth.",
        "sample_period_s": args.sample_period_s,
        "forecast_horizon_steps": horizon_steps,
        "forecast_horizon_s": horizon_steps * args.sample_period_s,
        "first_path_power_thresholds_dbm": args.thresholds_dbm,
        "primary_threshold_dbm": args.thresholds_dbm[0],
        "threshold_selection": (
            "Engineering sensitivity grid fixed before model evaluation; no threshold is "
            "optimized against MILUV component or event outcomes."
        ),
        "max_causal_hold_s": args.max_hold_s,
        "continuity_rule": "Use the longest contiguous interval where every robot pair has a causal measurement no older than max_causal_hold_s.",
        "radius_context_m": radius_m,
        "training_domain": "No MILUV samples are used for fitting or threshold selection; models and risk thresholds are frozen from simulation train/validation runs.",
        "seeds": args.seeds,
    }
    (args.output_dir / "miluv_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
