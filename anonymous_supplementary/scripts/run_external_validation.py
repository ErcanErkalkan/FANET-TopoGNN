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
from fanet.external_validation import build_trace_snapshots, load_flight_trace_csv, pairwise_distance_quantiles
from fanet.training import fit_kinetic_topoguard, fit_union_find_oracle, select_best_shallow


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Model"] = frame["Model"].where(~frame["Model"].str.startswith("Shallow ML"), "Shallow ML")
    metric_columns = [
        "MAE",
        "Raw_MAE",
        "Prediction_Clipped_Fraction",
        "MSE",
        "R2",
        "Risk_F1",
        "Risk_Precision",
        "Risk_Recall",
        "Risk_PR_AUC",
        "Risk_ROC_AUC",
        "Risk_Brier",
        "Risk_ECE",
        "False_Alarms_per_minute",
        "Inference_ms",
    ]
    available = [column for column in metric_columns if column in frame.columns]
    grouped = frame.groupby(["Model", "Radius_quantile", "Radius_m", "Fragmentation_rate"], as_index=False)
    means = grouped[available].mean().rename(columns={column: f"{column}_mean" for column in available})
    stds = grouped[available].std(ddof=1).fillna(0.0).rename(columns={column: f"{column}_std" for column in available})
    return means.merge(stds, on=["Model", "Radius_quantile", "Radius_m", "Fragmentation_rate"])


def _plot_trace(trace, radii: dict[float, float], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for vehicle_idx, vehicle_id in enumerate(trace.vehicle_ids):
        positions = trace.positions_m[:, vehicle_idx]
        axes[0].plot(positions[:, 0], positions[:, 1], linewidth=1.3, label=vehicle_id)
        axes[0].scatter(positions[0, 0], positions[0, 1], s=18)
    axes[0].set_xlabel("Local east (m)")
    axes[0].set_ylabel("Local north (m)")
    axes[0].set_title("Measured field trajectories")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for left in range(len(trace.vehicle_ids)):
        for right in range(left + 1, len(trace.vehicle_ids)):
            distance = np.linalg.norm(trace.positions_m[:, left] - trace.positions_m[:, right], axis=1)
            axes[1].plot(trace.timestamps_s, distance, linewidth=1.0, label=f"{trace.vehicle_ids[left]}-{trace.vehicle_ids[right]}")
    for quantile, radius in sorted(radii.items()):
        axes[1].axhline(radius, linestyle="--", linewidth=1.0, label=f"q{int(100 * quantile)} radius")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Pairwise distance (m)")
    axes[1].set_title("Distance and radius sensitivity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "external_trace_overview.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "external_trace_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_latex_table(aggregate: pd.DataFrame, path: Path) -> None:
    rows = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Model & Radius & Frag. rate & MAE & Clip rate & Risk $F_1$ & Brier \\\\",
        "\\midrule",
    ]
    for _, item in aggregate.sort_values(["Radius_quantile", "Model"]).iterrows():
        model = str(item["Model"]).replace("Union-Find detection oracle", "Union--find diagnostic")
        model = model.replace("Shallow ML", "Shallow ML").replace("_", "\\_")
        radius = f"q{int(round(100 * item['Radius_quantile']))} ({item['Radius_m']:.1f} m)"
        rows.append(
            f"{model} & {radius} & {item['Fragmentation_rate']:.3f} & "
            f"{item['MAE_mean']:.3f} $\\pm$ {item['MAE_std']:.3f} & "
            f"{item['Prediction_Clipped_Fraction_mean']:.3f} & "
            f"{item['Risk_F1_mean']:.3f} $\\pm$ {item['Risk_F1_std']:.3f} & "
            f"{item['Risk_Brier_mean']:.3f} $\\pm$ {item['Risk_Brier_std']:.3f} \\\\" 
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on simulation and evaluate on a public real-flight motion trace.")
    parser.add_argument("--config", default="configs/publication_compact_provenance.json")
    parser.add_argument("--trace", type=Path, default=Path("data/external_validation/derived/forestry_multidrone_trace.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_validation"))
    args = parser.parse_args()
    config = load_config(args.config)
    trace = load_flight_trace_csv(args.trace)
    radius_quantiles = (0.25, 0.50, 0.75)
    radii = pairwise_distance_quantiles(trace, radius_quantiles)
    horizon_steps = int(config.sim.get("forecast_horizon_steps", config.evaluation["warning_horizon_steps"]))
    dt = float(np.median(np.diff(trace.timestamps_s)))
    training_dt = float(config.sim["dt"])
    if not np.isclose(dt, training_dt, atol=1e-6):
        raise ValueError(
            f"External sample interval {dt:.6f}s does not match training interval {training_dt:.6f}s; "
            "rerun trace extraction at the benchmark sample rate"
        )
    external_sets = {
        quantile: build_trace_snapshots(
            trace,
            radius_m=radius,
            horizon_steps=horizon_steps,
            pi_resolution=int(config.sim["pi_resolution"]),
            pi_sigma=float(config.sim["pi_sigma"]),
            pi_max_radius=float(config.sim["pi_max_radius"]),
        )
        for quantile, radius in radii.items()
    }

    rows = []
    prediction_rows = []
    for seed in config.training["seed_list"]:
        print(f"[seed {seed}] building simulation training data")
        synthetic = build_dataset(config.sim, seed=int(seed))
        stratify = tuple(config.sim.get("split_stratify_by", ["mobility"]))
        train_data, val_data, _ = train_val_test_split(synthetic, split_seed=int(config.sim["split_seed"]), stratify_by=stratify)
        models = [
            fit_union_find_oracle(),
            select_best_shallow(train_data, val_data),
            fit_kinetic_topoguard(train_data, val_data, horizon_steps=horizon_steps, dt=training_dt, seed=int(seed)),
        ]
        for quantile, snapshots in external_sets.items():
            fragmentation_rate = float(np.mean([item.frag_at_horizon for item in snapshots]))
            for result in models:
                preds, risks, inference_ms, aligned, threshold = predict_generic(result, snapshots)
                raw_preds = np.asarray(preds, dtype=float)
                max_beta = float(max(item.n_nodes for item in aligned))
                clipped_preds = np.clip(raw_preds, 1.0, max_beta)
                targets = np.asarray([item.beta_target for item in aligned], dtype=float)
                metrics, _, _ = evaluate_predictions(
                    result.model_name,
                    aligned,
                    clipped_preds,
                    risks,
                    inference_ms,
                    dt=dt,
                    bootstrap_rounds=int(config.evaluation["bootstrap_rounds"]),
                    horizon_steps=horizon_steps,
                    risk_threshold=threshold,
                )
                metrics.update(
                    {
                        "Raw_MAE": float(np.mean(np.abs(targets - raw_preds))),
                        "Prediction_Clipped_Fraction": float(np.mean(raw_preds != clipped_preds)),
                        "Seed": int(seed),
                        "Radius_quantile": float(quantile),
                        "Radius_m": float(radii[quantile]),
                        "Fragmentation_rate": fragmentation_rate,
                        "Training_domain": "synthetic_simulation",
                        "Test_motion_domain": "real_forestry_field_trace",
                        "Link_labels": "counterfactual_radius_graph",
                    }
                )
                rows.append(metrics)
                prediction_rows.extend(
                    {
                        "Seed": int(seed),
                        "Model": result.model_name,
                        "Radius_quantile": float(quantile),
                        "Radius_m": float(radii[quantile]),
                        "time_index": int(item.time_index),
                        "beta_target": float(item.beta_target),
                        "prediction_raw": float(raw_pred),
                        "prediction_clipped": float(clipped_pred),
                        "risk_score": float(risk),
                        "risk_threshold": float(threshold),
                    }
                    for item, raw_pred, clipped_pred, risk in zip(aligned, raw_preds, clipped_preds, risks)
                )
                print(
                    f"[seed {seed}] q={quantile:.2f} {result.model_name}: "
                    f"MAE={metrics['MAE']:.3f}, F1={metrics['Risk_F1']:.3f}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    aggregate = _aggregate(raw)
    raw.to_csv(args.output_dir / "external_metrics_per_seed.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "external_predictions.csv", index=False)
    aggregate.to_csv(args.output_dir / "external_metrics_summary.csv", index=False)
    _write_latex_table(aggregate, args.output_dir / "external_metrics_table.tex")
    _plot_trace(trace, radii, args.output_dir)
    protocol = {
        "training_config": args.config,
        "training_seeds": [int(seed) for seed in config.training["seed_list"]],
        "test_trace": str(args.trace),
        "test_vehicle_count": len(trace.vehicle_ids),
        "test_duration_s": float(trace.timestamps_s[-1] - trace.timestamps_s[0]),
        "test_sample_interval_s": dt,
        "forecast_horizon_steps": horizon_steps,
        "forecast_horizon_s": horizon_steps * dt,
        "radius_selection": "25th, 50th, and 75th percentiles of all observed pairwise distances; selected without reference to model outcomes",
        "radii_m": {str(key): value for key, value in radii.items()},
        "scope_boundary": "Real flight motion only. The source has no packet/RF ground truth; graph labels are deterministic radius sensitivity scenarios.",
    }
    (args.output_dir / "external_validation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote external validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
