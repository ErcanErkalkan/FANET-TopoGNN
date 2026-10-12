from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fanet.config import load_config
import run_closed_loop_controller_validation as closed

SPEC_PATH = ROOT / "configs" / "revision_packet_loss_sensitivity.json"


def _materialize(base_path: Path, scenario: dict, destination: Path) -> None:
    raw = load_config(base_path).raw
    raw["experiment_name"] = f"revision_packet_loss_{scenario['name']}"
    raw["output_dir"] = str(destination.parent.relative_to(ROOT))
    packet = raw["packet_simulation"]
    packet["link_failure_probability_per_attempt"] = float(scenario["link_failure_probability_per_attempt"])
    packet["intervention_transition_failure_probability_per_attempt"] = float(scenario["intervention_transition_failure_probability_per_attempt"])
    packet["max_retransmissions"] = int(scenario["max_retransmissions"])
    packet["link_failure_interpretation"] = "Post-review stress setting; not a field-calibrated link-failure rate."
    packet["intervention_transition_interpretation"] = "Post-review stress setting; not a measured UAV handover-loss rate."
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _run_scenario_seed(seed: int, config_path: Path):
    metrics, logs, metadata = closed.run_seed(int(seed), config_path, None)
    return int(seed), metrics, logs, metadata


def _scenario_summary(metrics: pd.DataFrame, scenario: str) -> pd.DataFrame:
    metrics_to_report = [
        "Packet_Delivery_Ratio", "Packet_Deadline_Success", "Link_Failure_Drop_Rate",
        "Intervention_Transition_Drop_Rate", "No_Route_Drop_Rate",
        "Relay_Travel_m_per_Simulated_Minute", "Action_Duty_Cycle",
    ]
    rows = []
    for (policy, load, density), group in metrics.groupby(["Policy", "Traffic_Load", "Topology_Density_Nodes"], sort=True):
        row = {
            "Scenario": scenario,
            "Policy": policy,
            "Traffic_Load": int(load),
            "Topology_Density_Nodes": int(density),
            "Seeds": int(group["seed"].nunique()),
        }
        for metric in metrics_to_report:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            spread = 0.0 if len(values) < 2 else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_deltas(all_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "Policy", "Traffic_Load", "Topology_Density_Nodes"]
    baseline = all_metrics[all_metrics["Scenario"] == "baseline"].set_index(keys)
    rows = []
    for scenario, part in all_metrics[all_metrics["Scenario"] != "baseline"].groupby("Scenario"):
        current = part.set_index(keys)
        common = baseline.index.intersection(current.index)
        for metric in ["Packet_Delivery_Ratio", "Packet_Deadline_Success", "Link_Failure_Drop_Rate", "Intervention_Transition_Drop_Rate"]:
            delta = current.loc[common, metric].astype(float) - baseline.loc[common, metric].astype(float)
            rows.append({
                "Scenario": scenario,
                "Metric": metric,
                "Paired_Rows": int(delta.notna().sum()),
                "Delta_vs_Baseline_mean": float(delta.mean()),
                "Delta_vs_Baseline_median": float(delta.median()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-review packet/link/transition-loss sensitivity without overwriting manuscript assets.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    seed_spec = json.loads((ROOT / spec["seed_source"]).read_text(encoding="utf-8"))
    seeds = [int(x) for x in seed_spec["seed_list"][: int(spec["seed_count"])]]
    if len(seeds) != int(spec["seed_count"]) or len(set(seeds)) != len(seeds):
        raise RuntimeError("Packet sensitivity seed selection is not unique/complete.")

    output_root = ROOT / spec["output_dir"]
    config_root = output_root / "materialized_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    all_metric_frames = []
    all_pair_frames = []

    for scenario in spec["scenarios"]:
        name = str(scenario["name"])
        scenario_dir = output_root / name
        config_path = config_root / f"{name}.json"
        _materialize(ROOT / spec["base_config"], scenario, config_path)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        metric_frames = []
        log_frames = []
        metadata = []
        pending = []
        for seed in seeds:
            seed_dir = scenario_dir / "per_seed" / f"seed_{seed}"
            mpath, lpath, cpath = seed_dir / "metrics.csv", seed_dir / "action_log.csv", seed_dir / "complete.json"
            if args.resume and mpath.is_file() and lpath.is_file() and cpath.is_file():
                metric_frames.append(pd.read_csv(mpath))
                log_frames.append(pd.read_csv(lpath))
                metadata.append(json.loads(cpath.read_text(encoding="utf-8")))
            else:
                pending.append(seed)

        def save_seed(seed, result):
            _, metrics, logs, meta = result
            seed_dir = scenario_dir / "per_seed" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(seed_dir / "metrics.csv", index=False)
            logs.to_csv(seed_dir / "action_log.csv", index=False)
            (seed_dir / "complete.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            metric_frames.append(metrics)
            log_frames.append(logs)
            metadata.append(meta)

        if args.workers <= 1:
            for seed in pending:
                save_seed(seed, _run_scenario_seed(seed, config_path))
                print(f"[complete] scenario={name} seed={seed}", flush=True)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {executor.submit(_run_scenario_seed, seed, config_path): seed for seed in pending}
                for future in as_completed(futures):
                    seed = futures[future]
                    save_seed(seed, future.result())
                    print(f"[complete] scenario={name} seed={seed}", flush=True)

        metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["seed", "Policy", "Traffic_Load", "Topology_Density_Nodes"])
        logs = pd.concat(log_frames, ignore_index=True).sort_values(["seed", "policy", "traffic_load", "run_id", "time_index"])
        closed.validate_paired_initial_conditions(logs.to_dict("records"))
        max_error = float(pd.to_numeric(metrics["Drop_Accounting_Error"], errors="coerce").abs().max())
        if max_error > 1e-8:
            raise RuntimeError(f"scenario={name}: drop-accounting error {max_error}")
        if set(metrics["seed"].astype(int)) != set(seeds):
            raise RuntimeError(f"scenario={name}: seed coverage mismatch")

        config = load_config(config_path)
        rounds = int(config.raw["closed_loop"]["bootstrap_rounds"])
        paired = closed._paired_tests(metrics, rounds, config.raw["closed_loop"]["acceptable_action_cost"])
        summary = _scenario_summary(metrics, name)
        metrics.insert(0, "Scenario", name)
        paired.insert(0, "Scenario", name)
        metrics.to_csv(scenario_dir / "metrics_per_seed.csv", index=False)
        summary.to_csv(scenario_dir / "scenario_summary.csv", index=False)
        paired.to_csv(scenario_dir / "paired_tests.csv", index=False)
        logs.to_csv(scenario_dir / "action_log.csv", index=False)
        (scenario_dir / "protocol.json").write_text(json.dumps({
            "status": "complete",
            "scenario": scenario,
            "seeds": seeds,
            "paired_initial_conditions": True,
            "max_drop_accounting_error": max_error,
            "claim_boundary": spec["claim_rule"],
            "per_seed_protocol": metadata,
        }, indent=2) + "\n", encoding="utf-8")
        all_metric_frames.append(metrics)
        all_pair_frames.append(paired)

    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    all_paired = pd.concat(all_pair_frames, ignore_index=True)
    combined_summary = pd.concat([
        _scenario_summary(group.drop(columns=["Scenario"]), scenario)
        for scenario, group in all_metrics.groupby("Scenario", sort=False)
    ], ignore_index=True)
    deltas = _baseline_deltas(all_metrics)
    all_metrics.to_csv(output_root / "all_scenarios_metrics_per_seed.csv", index=False)
    all_paired.to_csv(output_root / "all_scenarios_paired_tests.csv", index=False)
    combined_summary.to_csv(output_root / "all_scenarios_summary.csv", index=False)
    deltas.to_csv(output_root / "paired_scenario_deltas_vs_baseline.csv", index=False)
    final_protocol = {
        **spec,
        "protocol_status": "complete",
        "seed_list": seeds,
        "n_seeds": len(seeds),
        "n_scenarios": len(spec["scenarios"]),
        "test_role": "paired descriptive engineering stress sensitivity",
    }
    (output_root / "revision_packet_loss_sensitivity_protocol.json").write_text(json.dumps(final_protocol, indent=2) + "\n", encoding="utf-8")
    print(combined_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
