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

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import alert_event_metrics, predict_generic
from fanet.source_gated import fit_current_state_extratrees, fit_source_gated_kinetic_topoguard
from fanet.training import fit_kinetic_topoguard

SPEC_PATH = ROOT / "configs" / "revision_event_protocol_20seed.json"


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
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        raise RuntimeError("run_id leakage in revision event-protocol split")
    return train, validation, test


def _event_rate(snapshots: list, dt: float) -> float:
    grouped: dict[str, list] = {}
    for snap in snapshots:
        grouped.setdefault(snap.run_id, []).append(snap)
    events = 0
    duration_minutes = 0.0
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item.time_index)
        duration_minutes += len(ordered) * dt / 60.0
        events += sum(
            ordered[index - 1].beta_current <= 1.0 and ordered[index].beta_current > 1.0
            for index in range(1, len(ordered))
        )
    return float(events / max(duration_minutes, 1e-12))


def _density_groups(snapshots: list, scores: np.ndarray, dt: float):
    grouped: dict[str, list[tuple[object, float]]] = {}
    for snap, score in zip(snapshots, scores):
        grouped.setdefault(snap.run_id, []).append((snap, float(score)))
    rows = []
    for run_id, pairs in grouped.items():
        ordered = sorted(pairs, key=lambda item: item[0].time_index)
        run_snaps = [item[0] for item in ordered]
        rows.append((run_id, _event_rate(run_snaps, dt), ordered))
    rows.sort(key=lambda item: (item[1], item[0]))
    chunks = np.array_split(np.arange(len(rows)), 3)
    labels = ("Low tertile", "Middle tertile", "High tertile")
    result = []
    for label, chunk in zip(labels, chunks):
        selected = [rows[int(index)] for index in chunk]
        flattened = [pair for _, _, ordered in selected for pair in ordered]
        result.append((label, [item[0] for item in flattened], np.asarray([item[1] for item in flattened], dtype=float)))
    return result


def _metric_row(seed, model_name, dimension, level, snapshots, scores, threshold, dt, warning_steps, cooldown_steps):
    metrics = alert_event_metrics(
        snapshots,
        scores,
        threshold=float(threshold),
        dt=float(dt),
        horizon_steps=int(warning_steps),
        cooldown_steps=int(cooldown_steps),
    )
    return {
        "seed": int(seed),
        "Model": model_name,
        "Dimension": dimension,
        "Level": level,
        "Warning_window_steps": int(warning_steps),
        "Cooldown_steps": int(cooldown_steps),
        "Selected_threshold": float(threshold),
        "Ground_truth_events_per_minute": _event_rate(snapshots, dt),
        **metrics,
    }


def _run_seed(seed: int, spec: dict) -> list[dict]:
    config = load_config(ROOT / spec["base_config"])
    horizon = int(config.sim["forecast_horizon_steps"])
    dt = float(config.sim["dt"])
    snapshots = build_dataset(config.sim, seed=int(seed))
    train, validation, test = _split(config, snapshots)
    models = [
        fit_current_state_extratrees(train, validation, horizon, dt, int(seed)),
        fit_kinetic_topoguard(train, validation, horizon, dt, seed=int(seed)),
        fit_source_gated_kinetic_topoguard(
            train,
            validation,
            horizon,
            dt,
            int(seed),
            parameters=config.training["source_gated"],
        ),
    ]
    rows: list[dict] = []
    for result in models:
        _, scores, _, aligned, threshold = predict_generic(result, test)
        for value in spec["warning_window_steps"]:
            rows.append(_metric_row(seed, result.model_name, "Warning window", f"{value * dt:.1f} s", aligned, scores, threshold, dt, int(value), horizon))
        for value in spec["cooldown_steps"]:
            rows.append(_metric_row(seed, result.model_name, "Alert refractory period", f"{value * dt:.1f} s", aligned, scores, threshold, dt, horizon, int(value)))
        for label, group_snaps, group_scores in _density_groups(aligned, scores, dt):
            rows.append(_metric_row(seed, result.model_name, "Measured event density", label, group_snaps, group_scores, threshold, dt, horizon, horizon))
    print(f"completed revision event-protocol seed={seed}", flush=True)
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Ground_truth_events_per_minute", "Alert_Event_Precision", "Alert_Event_Recall",
        "Alert_Event_F1", "False_Alert_Events_per_minute",
    ]
    rows = []
    for (model, dimension, level), group in frame.groupby(["Model", "Dimension", "Level"], sort=False):
        row = {"Model": model, "Dimension": dimension, "Level": level, "Seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            spread = 0.0 if len(values) < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows)


def _validate(frame: pd.DataFrame, spec: dict) -> None:
    expected_seeds = {int(x) for x in spec["seed_list"]}
    if set(frame["seed"].astype(int)) != expected_seeds:
        raise RuntimeError("Seed coverage mismatch in revision event-protocol output.")
    expected_rows_per_model = len(spec["warning_window_steps"]) + len(spec["cooldown_steps"]) + 3
    for seed in expected_seeds:
        part = frame[frame["seed"].astype(int) == seed]
        if part["Model"].nunique() != 3:
            raise RuntimeError(f"seed={seed}: expected 3 models")
        for model, group in part.groupby("Model"):
            if len(group) != expected_rows_per_model:
                raise RuntimeError(f"seed={seed}, model={model}: incomplete sensitivity coverage")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen 20-seed post-review event-protocol robustness study.")
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
    frame.to_csv(output_dir / "revision_event_protocol_20seed_per_seed.csv", index=False)
    summary.to_csv(output_dir / "revision_event_protocol_20seed_summary.csv", index=False)
    protocol = {
        **spec,
        "protocol_status": "complete",
        "seed_list": spec["seed_list"],
        "n_seeds": len(spec["seed_list"]),
        "base_forecast_horizon_steps": int(load_config(ROOT / spec["base_config"]).sim["forecast_horizon_steps"]),
        "test_role": "descriptive post-review scoring robustness; thresholds frozen within each model/seed",
    }
    (output_dir / "revision_event_protocol_20seed_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
