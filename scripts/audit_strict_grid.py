"""Validate exact-horizon labeling across a registered seed/horizon grid.

This performs DATA-LABEL checks only, not model training, R1 numerical replay
or R2 seed-paired inferential validation.
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
from fanet.dataset import (
    build_dataset, relabel_forecast_horizon, split_by_run,
    validate_full_horizon_labels,
)


def audit_grid(config_path: Path, seeds: list[int], horizons: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use a nonempty list of unique seeds")
    if not horizons or len(set(horizons)) != len(horizons) or min(horizons) <= 0:
        raise ValueError("Use a nonempty list of unique positive horizons")
    resolved = load_config(config_path)
    raw_sim = {**resolved.sim, "horizon_label_policy": "legacy_clamp"}
    fingerprint = hashlib.sha256(
        json.dumps(resolved.raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for seed in seeds:
        # Complete trajectories are mandatory for *every* target horizon.
        full = build_dataset(raw_sim, seed=seed)
        original_runs = split_by_run(full)
        originals = {
            name: {int(s.time_index): s for s in snaps}
            for name, snaps in original_runs.items()
        }
        for name, snaps in original_runs.items():
            if len(originals[name]) != len(snaps):
                raise RuntimeError(f"Repeated timestamp in original run: {name}")
        for horizon in horizons:
            selected = relabel_forecast_horizon(full, horizon, strict_full_horizon=True)
            if not selected:
                raise RuntimeError(f"No full-horizon samples for seed={seed}, h={horizon}")
            check = validate_full_horizon_labels(selected, horizon)
            expected = sum(
                sum(int(s.time_index) + horizon in originals[name] for s in snaps)
                for name, snaps in original_runs.items()
            )
            if expected != len(selected):
                raise RuntimeError(f"Support mismatch seed={seed}, horizon={horizon}")
            for item in selected:
                reference = originals[item.run_id][int(item.time_index) + horizon]
                if item.beta_target != reference.beta_current:
                    raise RuntimeError(f"Target mismatch seed={seed}, run={item.run_id}, h={horizon}")
                if item.frag_at_horizon != int(reference.beta_current > 1):
                    raise RuntimeError(f"Risk mismatch seed={seed}, run={item.run_id}, h={horizon}")
            rows.append({
                "seed": seed,
                "horizon_steps": horizon,
                "raw_rows": len(full),
                "valid_rows": len(selected),
                "right_censored_rows": len(full) - len(selected),
                "runs": check["runs_checked"],
                "status": "pass",
            })
    return {
        "kind": "strict_v4_data_label_grid_only",
        "status": "pass",
        "input_config_sha256": fingerprint,
        "seeds": seeds,
        "horizons": horizons,
        "completed_seed_horizon_cells": len(rows),
        "not_a_model_comparison": True,
        "locked_eaai_results_unchanged": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict t+h labels across a registered grid")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/publication_compact.json")
    parser.add_argument("--seed-config", type=Path, help="Config holding training.seed_list")
    parser.add_argument("--seed", type=int, default=7, help="Pilot seed if --seed-config is omitted")
    parser.add_argument("--horizons", default="2,4,6,10,15,20")
    parser.add_argument("--out", type=Path, help="New JSON location; existing files are protected")
    args = parser.parse_args()
    seeds = (
        [int(x) for x in load_config(args.seed_config).training["seed_list"]]
        if args.seed_config else [args.seed]
    )
    horizons = [int(x.strip()) for x in args.horizons.split(",")]
    result = audit_grid(args.config, seeds, horizons)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            f.write(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
