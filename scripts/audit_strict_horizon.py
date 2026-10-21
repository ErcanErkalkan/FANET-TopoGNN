"""Audit exact-horizon labels against complete simulated raw trajectories.

This is a data-integrity check, NOT a new model evaluation, and never alters
historical metrics/locked confirmatory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset, split_by_run, validate_full_horizon_labels


def audit(config_path: Path, seed: int) -> dict:
    resolved = load_config(config_path)
    sim = dict(resolved.sim)
    if sim.get("horizon_label_policy") != "strict_full_horizon":
        raise ValueError("Audit requires an explicitly strict_full_horizon configuration")
    horizon = int(sim["forecast_horizon_steps"])
    legacy_sim = {**sim, "horizon_label_policy": "legacy_clamp"}
    legacy = build_dataset(legacy_sim, seed=seed)
    strict = build_dataset(sim, seed=seed)
    checked = validate_full_horizon_labels(strict, horizon)
    complete = split_by_run(legacy)
    censored = split_by_run(strict)
    if set(complete) != set(censored):
        raise RuntimeError("Strict and raw simulation run identifiers do not match")
    excluded = 0
    for run_id, original in complete.items():
        original = sorted(original, key=lambda item: item.time_index)
        clean = sorted(censored[run_id], key=lambda item: item.time_index)
        lookup = {int(item.time_index): item for item in original}
        if len(lookup) != len(original):
            raise RuntimeError(f"Duplicate raw timestamp in {run_id}")
        expected = [item for item in original if int(item.time_index) + horizon in lookup]
        if [s.time_index for s in clean] != [s.time_index for s in expected]:
            raise RuntimeError(f"Wrong strict-label support in {run_id}")
        excluded += len(original) - len(clean)
        for snap in clean:
            future = lookup[int(snap.time_index) + horizon]
            if float(snap.beta_target) != float(future.beta_current):
                raise RuntimeError(f"Wrong target beta0 in {run_id}:{snap.time_index}")
            if int(snap.frag_at_horizon) != int(float(future.beta_current) > 1.0):
                raise RuntimeError(f"Wrong risk label in {run_id}:{snap.time_index}")
            if float(snap.beta_current) != float(lookup[int(snap.time_index)].beta_current):
                raise RuntimeError(f"Changed present-state graph in {run_id}:{snap.time_index}")
    if len(legacy) != len(strict) + excluded:
        raise RuntimeError("Audit row conservation failed")
    cfg_sha = hashlib.sha256(
        json.dumps(resolved.raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "pass",
        "kind": "data_label_integrity_only",
        "experiment_name": resolved.experiment_name,
        "config_sha256": cfg_sha,
        "seed": int(seed),
        "label_policy": "strict_full_horizon",
        "horizon_steps": horizon,
        "raw_rows": len(legacy),
        "full_horizon_rows": len(strict),
        "right_censored_rows_excluded": excluded,
        "run_count": len(complete),
        "checked": checked,
        "manuscript_superiority_claim": False,
        "historical_outputs_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all strict targets against untruncated t+h ground truth.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/strict_v4_smoke.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, help="Optional NEW audit JSON path; existing files are never overwritten")
    args = parser.parse_args()
    payload = audit(args.config, args.seed)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            f.write(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
