from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, train_val_test_split
from fanet.geometry import pairwise_distances
from fanet.graph_utils import adjacency_from_radius, betti_zero

SPEC_PATH = ROOT / "configs" / "revision_motion_model_ablation.json"


def _split(raw: dict, snapshots):
    stratify = list(raw["sim"].get("split_stratify_by", ["mobility"]))
    for field in ("graph_policy", "radio_scenario"):
        if len({getattr(snapshot, field) for snapshot in snapshots}) > 1 and field not in stratify:
            stratify.append(field)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(raw["sim"]["split_seed"]),
        stratify_by=tuple(stratify),
    )
    sets = [{snapshot.run_id for snapshot in part} for part in (train, validation, test)]
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        raise RuntimeError("run_id leakage in motion-model ablation split")
    groups = [{snapshot.split_group_id for snapshot in part} for part in (train, validation, test)]
    if (groups[0] & groups[1]) or (groups[0] & groups[2]) or (groups[1] & groups[2]):
        raise RuntimeError("split_group_id leakage across train/validation/test")
    return test


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _project(snapshot, previous, tau: float, dt: float, mode: str) -> np.ndarray:
    p = snapshot.positions.astype(float)
    v = snapshot.velocities.astype(float)
    if mode == "no_motion":
        return p.copy()
    if mode == "constant_velocity":
        return p + v * tau
    if mode == "constant_acceleration":
        a = (v - previous.velocities.astype(float)) / max(dt, 1e-12)
        return p + v * tau + 0.5 * a * tau * tau
    raise ValueError(f"unknown mode: {mode}")


def _snapshot_metrics(snapshot, previous, future, horizon: int, dt: float, mode: str) -> dict:
    tau = horizon * dt
    predicted = _project(snapshot, previous, tau, dt, mode)
    actual = future.positions.astype(float)
    radius = max(float(snapshot.radius), 1e-9)

    pos_rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    pred_dist = pairwise_distances(predicted)
    actual_dist = pairwise_distances(actual)
    current_dist = pairwise_distances(snapshot.positions.astype(float))
    tri = np.triu_indices(snapshot.n_nodes, k=1)
    pred_pairs = pred_dist[tri]
    actual_pairs = actual_dist[tri]
    current_pairs = current_dist[tri]
    pair_mae = float(np.mean(np.abs(pred_pairs - actual_pairs))) if pred_pairs.size else 0.0

    pred_adj = adjacency_from_radius(pred_dist, radius)
    actual_adj = adjacency_from_radius(actual_dist, radius)
    pred_beta = int(betti_zero(pred_adj))
    actual_beta = int(betti_zero(actual_adj))

    pred_cross_out = (current_pairs <= radius) & (pred_pairs > radius)
    actual_cross_out = (current_pairs <= radius) & (actual_pairs > radius)
    crossing_tp = int(np.sum(pred_cross_out & actual_cross_out))
    crossing_fp = int(np.sum(pred_cross_out & ~actual_cross_out))
    crossing_fn = int(np.sum(~pred_cross_out & actual_cross_out))

    pred_frag_geom = int(pred_beta > 1)
    actual_frag_geom = int(actual_beta > 1)
    physical_frag = int(snapshot.frag_at_horizon)
    return {
        "Position_RMSE_m": pos_rmse,
        "Pair_Distance_MAE_m": pair_mae,
        "Pair_Distance_MAE_over_radius": pair_mae / radius,
        "Predicted_Geometric_Beta0": pred_beta,
        "Actual_Future_Geometric_Beta0": actual_beta,
        "Geometric_Beta0_Absolute_Error": abs(pred_beta - actual_beta),
        "Geometric_Beta0_Exact": float(pred_beta == actual_beta),
        "Geometric_Fragmentation_Accuracy": float(pred_frag_geom == actual_frag_geom),
        "Physical_Fragmentation_Proxy_Accuracy": float(pred_frag_geom == physical_frag),
        "Outbound_Radius_Crossing_TP": crossing_tp,
        "Outbound_Radius_Crossing_FP": crossing_fp,
        "Outbound_Radius_Crossing_FN": crossing_fn,
    }


def _run_seed(seed: int, spec: dict) -> list[dict]:
    raw = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    dt = float(raw["sim"]["dt"])
    rows: list[dict] = []

    for horizon in [int(x) for x in spec["horizon_steps"]]:
        sim = dict(raw["sim"])
        sim["forecast_horizon_steps"] = horizon
        snapshots = build_dataset(sim, seed=int(seed))
        test = _split({**raw, "sim": sim}, snapshots)
        grouped: dict[str, list] = {}
        for snap in test:
            grouped.setdefault(snap.run_id, []).append(snap)

        snapshot_rows = []
        for run_id, seq in grouped.items():
            ordered = sorted(seq, key=lambda item: item.time_index)
            by_time = {int(item.time_index): item for item in ordered}
            for snap in ordered:
                previous = by_time.get(int(snap.time_index) - 1)
                future = by_time.get(int(snap.future_time_index))
                # simulate_run clamps the final h labels to the run end; exclude
                # those rows rather than scoring them as full-horizon forecasts.
                if previous is None or future is None:
                    continue
                if int(future.time_index) - int(snap.time_index) != horizon:
                    continue
                for mode in spec["projection_modes"]:
                    snapshot_rows.append({
                        "seed": int(seed),
                        "run_id": run_id,
                        "time_index": int(snap.time_index),
                        "horizon_steps": horizon,
                        "horizon_s": horizon * dt,
                        "Projection_Mode": mode,
                        **_snapshot_metrics(snap, previous, future, horizon, dt, mode),
                    })

        frame = pd.DataFrame(snapshot_rows)
        if frame.empty:
            raise RuntimeError(f"seed={seed}, horizon={horizon}: no common-support rows")
        mode_groups = list(frame.groupby("Projection_Mode", sort=False))
        n_rows = {name: len(group) for name, group in mode_groups}
        if len(set(n_rows.values())) != 1 or set(n_rows) != set(spec["projection_modes"]):
            raise RuntimeError(f"seed={seed}, horizon={horizon}: common support differs by mode")
        for mode, group in mode_groups:
            tp = int(group["Outbound_Radius_Crossing_TP"].sum())
            fp = int(group["Outbound_Radius_Crossing_FP"].sum())
            fn = int(group["Outbound_Radius_Crossing_FN"].sum())
            crossing_f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
            rows.append({
                "seed": int(seed),
                "horizon_steps": horizon,
                "horizon_s": horizon * dt,
                "Projection_Mode": mode,
                "Common_Support_Rows": int(len(group)),
                "Position_RMSE_m": float(group["Position_RMSE_m"].mean()),
                "Pair_Distance_MAE_m": float(group["Pair_Distance_MAE_m"].mean()),
                "Pair_Distance_MAE_over_radius": float(group["Pair_Distance_MAE_over_radius"].mean()),
                "Geometric_Beta0_Absolute_Error": float(group["Geometric_Beta0_Absolute_Error"].mean()),
                "Geometric_Beta0_Exact": float(group["Geometric_Beta0_Exact"].mean()),
                "Geometric_Fragmentation_Accuracy": float(group["Geometric_Fragmentation_Accuracy"].mean()),
                "Physical_Fragmentation_Proxy_Accuracy": float(group["Physical_Fragmentation_Proxy_Accuracy"].mean()),
                "Outbound_Radius_Crossing_F1": crossing_f1,
                "Outbound_Radius_Crossing_TP": tp,
                "Outbound_Radius_Crossing_FP": fp,
                "Outbound_Radius_Crossing_FN": fn,
            })
        print(f"completed motion ablation seed={seed} horizon={horizon}", flush=True)
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Position_RMSE_m", "Pair_Distance_MAE_m", "Pair_Distance_MAE_over_radius",
        "Geometric_Beta0_Absolute_Error", "Geometric_Beta0_Exact",
        "Geometric_Fragmentation_Accuracy", "Physical_Fragmentation_Proxy_Accuracy",
        "Outbound_Radius_Crossing_F1",
    ]
    rows = []
    for (horizon, horizon_s, mode), group in frame.groupby(["horizon_steps", "horizon_s", "Projection_Mode"], sort=True):
        row = {
            "horizon_steps": int(horizon),
            "horizon_s": float(horizon_s),
            "Projection_Mode": mode,
            "Seeds": int(group["seed"].nunique()),
            "Common_Support_Rows_mean": float(group["Common_Support_Rows"].mean()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            spread = 0.0 if len(values) < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Position_RMSE_m", "Pair_Distance_MAE_m", "Geometric_Beta0_Absolute_Error", "Geometric_Beta0_Exact", "Outbound_Radius_Crossing_F1"]
    rows = []
    for horizon, part in frame.groupby("horizon_steps"):
        pivot = part.pivot(index="seed", columns="Projection_Mode", values=metrics)
        for metric in metrics:
            if (metric, "constant_velocity") not in pivot.columns or (metric, "constant_acceleration") not in pivot.columns:
                continue
            delta = pivot[(metric, "constant_acceleration")] - pivot[(metric, "constant_velocity")]
            rows.append({
                "horizon_steps": int(horizon),
                "metric": metric,
                "delta_CA_minus_CV_mean": float(delta.mean()),
                "delta_CA_minus_CV_median": float(delta.median()),
                "seeds": int(delta.notna().sum()),
            })
    return pd.DataFrame(rows)


def _validate(frame: pd.DataFrame, spec: dict) -> None:
    expected_seeds = {int(x) for x in spec["seed_list"]}
    expected_horizons = {int(x) for x in spec["horizon_steps"]}
    expected_modes = set(spec["projection_modes"])
    if set(frame["seed"].astype(int)) != expected_seeds:
        raise RuntimeError("motion ablation seed coverage mismatch")
    for seed in expected_seeds:
        part = frame[frame["seed"].astype(int) == seed]
        if set(part["horizon_steps"].astype(int)) != expected_horizons:
            raise RuntimeError(f"seed={seed}: incomplete horizon coverage")
        for horizon in expected_horizons:
            modes = set(part[part["horizon_steps"].astype(int) == horizon]["Projection_Mode"])
            if modes != expected_modes:
                raise RuntimeError(f"seed={seed}, horizon={horizon}: projection-mode coverage mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run no-motion/CV/CA geometric projection ablation on 20 revision seeds.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    seed_spec = json.loads((ROOT / spec["seed_source"]).read_text(encoding="utf-8"))
    spec["seed_list"] = [int(x) for x in seed_spec["seed_list"]]
    if len(spec["seed_list"]) != 20 or len(set(spec["seed_list"])) != 20:
        raise RuntimeError("Expected 20 unique revision seeds.")

    output_dir = ROOT / spec["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.workers <= 1:
        for seed in spec["seed_list"]:
            rows.extend(_run_seed(seed, spec))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(spec["seed_list"]))) as pool:
            for seed_rows in pool.map(_run_seed, spec["seed_list"], [spec] * len(spec["seed_list"])):
                rows.extend(seed_rows)

    frame = pd.DataFrame(rows)
    _validate(frame, spec)
    summary = _summary(frame)
    deltas = _paired_deltas(frame)
    frame.to_csv(output_dir / "revision_motion_model_ablation_per_seed.csv", index=False)
    summary.to_csv(output_dir / "revision_motion_model_ablation_summary.csv", index=False)
    deltas.to_csv(output_dir / "revision_motion_model_ablation_paired_deltas.csv", index=False)
    protocol = {**spec, "protocol_status": "complete", "seed_list": spec["seed_list"], "n_seeds": len(spec["seed_list"])}
    (output_dir / "revision_motion_model_ablation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nCA minus CV paired deltas:\n", deltas.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
