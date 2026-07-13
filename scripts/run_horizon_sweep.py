from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, relabel_forecast_horizon, train_val_test_split
from fanet.evaluation import evaluate_predictions, predict_generic
from fanet.training import fit_current_state_persistence, fit_kinetic_topoguard, select_best_shallow


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "horizon_sweep"


def _mean_ci(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            values = group[metric].astype(float)
            mean = float(values.mean())
            spread = 0.0 if len(values) < 2 else 1.96 * float(values.std(ddof=1)) / len(values) ** 0.5
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - spread
            row[f"{metric}_ci95_high"] = mean + spread
        rows.append(row)
    return pd.DataFrame(rows)


def _run_seed(seed: int, raw: dict, horizons: list[int]) -> list[dict]:
    rows: list[dict] = []
    sim = copy.deepcopy(raw["sim"])
    sim["forecast_horizon_steps"] = max(int(value) for value in horizons)
    base_snapshots = build_dataset(sim, seed=seed)
    for horizon in horizons:
        snapshots = relabel_forecast_horizon(base_snapshots, int(horizon))
        train, val, test = train_val_test_split(
            snapshots,
            split_seed=int(sim["split_seed"]),
            stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
        )
        models = [
            fit_kinetic_topoguard(train, val, int(horizon), float(sim["dt"]), seed=seed),
            select_best_shallow(train, val),
            fit_current_state_persistence(),
        ]
        for result in models:
            preds, scores, latency, aligned, threshold = predict_generic(result, test)
            summary, _, _ = evaluate_predictions(
                result.model_name,
                aligned,
                preds,
                scores,
                latency,
                dt=float(sim["dt"]),
                bootstrap_rounds=100,
                horizon_steps=int(horizon),
                risk_threshold=threshold,
            )
            rows.append(
                {
                    "seed": seed,
                    "horizon_steps": int(horizon),
                    "horizon_s": float(horizon) * float(sim["dt"]),
                    **summary,
                }
            )
        print(f"completed seed={seed} horizon={horizon}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a leakage-safe forecast-horizon sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 6, 10, 15, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.workers <= 1:
        for seed in args.seeds:
            rows.extend(_run_seed(seed, raw, args.horizons))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seeds))) as pool:
            for seed_rows in pool.map(
                _run_seed,
                args.seeds,
                [raw] * len(args.seeds),
                [args.horizons] * len(args.seeds),
            ):
                rows.extend(seed_rows)

    per_seed = pd.DataFrame(rows)
    per_seed["Model"] = per_seed["Model"].str.replace(
        r"^Shallow ML.*$",
        "Shallow ML",
        regex=True,
    )
    metrics = [
        "MAE",
        "R2",
        "Risk_F1",
        "Risk_PR_AUC",
        "Alert_Event_Precision",
        "Alert_Event_Recall",
        "Alert_Event_F1",
        "False_Alert_Events_per_minute",
        "Inference_ms",
    ]
    summary = _mean_ci(per_seed, ["horizon_steps", "horizon_s", "Model"], metrics)
    per_seed.to_csv(args.output_dir / "horizon_sweep_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "horizon_sweep_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for model, group in summary.groupby("Model"):
        group = group.sort_values("horizon_s")
        axes[0].plot(group["horizon_s"], group["MAE_mean"], marker="o", label=model)
        axes[1].plot(group["horizon_s"], group["Alert_Event_F1_mean"], marker="o", label=model)
    axes[0].set(xlabel="Forecast horizon (s)", ylabel="MAE", title="Horizon sensitivity: regression")
    axes[1].set(xlabel="Forecast horizon (s)", ylabel="Event F1", title="Horizon sensitivity: fragmentation events")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "horizon_sweep.png", dpi=220)
    fig.savefig(args.output_dir / "horizon_sweep.pdf")
    plt.close(fig)

    protocol = {
        "config": str(args.config.relative_to(ROOT)),
        "seeds": args.seeds,
        "workers": args.workers,
        "horizon_steps": args.horizons,
        "horizon_seconds": [float(value) * float(raw["sim"]["dt"]) for value in args.horizons],
        "selection": "All model and risk hyperparameters are selected on validation runs; reported metrics use disjoint test runs.",
        "trajectory_reuse": "Each seed's physical trajectory and radio realizations are simulated once; only future labels are deterministically re-indexed for each horizon.",
    }
    (args.output_dir / "horizon_sweep_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
