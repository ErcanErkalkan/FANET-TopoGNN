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
from fanet.dataset import build_dataset, relabel_forecast_horizon, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.source_gated import fit_current_state_extratrees, fit_source_gated_kinetic_topoguard
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard, select_best_shallow

SPEC_PATH = ROOT / "configs" / "revision_horizon_20seed.json"


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
    overlap = {
        "train_validation": sorted(sets[0] & sets[1]),
        "train_test": sorted(sets[0] & sets[2]),
        "validation_test": sorted(sets[1] & sets[2]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"run_id leakage: {overlap}")
    return train, validation, test, stratify


def _run_seed(seed: int, spec: dict) -> list[dict]:
    config = load_config(ROOT / spec["base_config"])
    horizons = [int(value) for value in spec["horizon_steps"]]
    sim = dict(config.sim)
    sim["forecast_horizon_steps"] = max(horizons)
    base_snapshots = build_dataset(sim, seed=int(seed))
    dt = float(sim["dt"])
    rows: list[dict] = []

    for horizon in horizons:
        snapshots = relabel_forecast_horizon(base_snapshots, horizon)
        train, validation, test, stratify = _split(config, snapshots)
        models = [
            fit_current_state_persistence(),
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
            select_best_shallow(train, validation),
        ]
        for result in models:
            predictions, scores, latency, aligned, threshold = predict_generic(result, test)
            metrics, _, _ = evaluate_predictions(
                result.model_name,
                aligned,
                predictions,
                scores,
                latency,
                dt=dt,
                bootstrap_rounds=100,
                horizon_steps=horizon,
                risk_threshold=threshold,
            )
            rows.append({
                "seed": int(seed),
                "horizon_steps": horizon,
                "horizon_s": horizon * dt,
                "selected_threshold": float(threshold),
                "split_stratify_by": ";".join(stratify),
                **metrics,
            })
        print(f"completed revision horizon seed={seed} horizon={horizon}", flush=True)
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "MAE", "R2", "Risk_F1", "Risk_PR_AUC", "Risk_Brier", "Risk_ECE",
        "Alert_Event_Precision", "Alert_Event_Recall", "Alert_Event_F1",
        "False_Alert_Events_per_minute", "Inference_ms",
    ]
    rows = []
    for (h_steps, h_s, model), group in frame.groupby(["horizon_steps", "horizon_s", "Model"], sort=True):
        row = {
            "horizon_steps": int(h_steps),
            "horizon_s": float(h_s),
            "Model": model,
            "Seeds": int(group["seed"].nunique()),
        }
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
    expected_horizons = {int(x) for x in spec["horizon_steps"]}
    if set(frame["seed"].astype(int)) != expected_seeds:
        raise RuntimeError("Seed coverage mismatch in revision horizon output.")
    for seed in expected_seeds:
        part = frame[frame["seed"].astype(int) == seed]
        if set(part["horizon_steps"].astype(int)) != expected_horizons:
            raise RuntimeError(f"seed={seed}: incomplete horizon coverage")
        for horizon in expected_horizons:
            rows = part[part["horizon_steps"].astype(int) == horizon]
            if len(rows) != 5:
                raise RuntimeError(f"seed={seed}, horizon={horizon}: expected 5 models, found {len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen 20-seed post-review horizon robustness study.")
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
    frame["Model"] = frame["Model"].str.replace(r"^Shallow ML.*$", "Shallow ML", regex=True)
    _validate(frame, spec)
    summary = _summary(frame)
    frame.to_csv(output_dir / "revision_horizon_20seed_per_seed.csv", index=False)
    summary.to_csv(output_dir / "revision_horizon_20seed_summary.csv", index=False)
    protocol = {
        **spec,
        "protocol_status": "complete",
        "seed_list": spec["seed_list"],
        "n_seeds": len(spec["seed_list"]),
        "dt_seconds": float(load_config(ROOT / spec["base_config"]).sim["dt"]),
        "test_role": "descriptive post-review robustness; no replacement of original locked test",
    }
    (output_dir / "revision_horizon_20seed_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
