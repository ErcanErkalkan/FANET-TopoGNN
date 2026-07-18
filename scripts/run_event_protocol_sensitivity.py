from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import build_dataset, train_val_test_split
from fanet.evaluation import alert_event_metrics, predict_generic
from fanet.provenance import relative_repo_path
from fanet.training import fit_kinetic_topoguard


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "event_protocol_sensitivity"


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


def _density_groups(snapshots: list, scores: np.ndarray, dt: float) -> list[tuple[str, list, np.ndarray]]:
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
        result.append(
            (
                label,
                [item[0] for item in flattened],
                np.asarray([item[1] for item in flattened], dtype=float),
            )
        )
    return result


def _metric_row(
    seed: int,
    dimension: str,
    level: str,
    snapshots: list,
    scores: np.ndarray,
    threshold: float,
    dt: float,
    warning_steps: int,
    cooldown_steps: int,
) -> dict:
    metrics = alert_event_metrics(
        snapshots,
        scores,
        threshold=threshold,
        dt=dt,
        horizon_steps=warning_steps,
        cooldown_steps=cooldown_steps,
    )
    return {
        "seed": int(seed),
        "Dimension": dimension,
        "Level": level,
        "Warning_window_steps": int(warning_steps),
        "Cooldown_steps": int(cooldown_steps),
        "Ground_truth_events_per_minute": _event_rate(snapshots, dt),
        **metrics,
    }


def _run_seed(seed: int, raw: dict, warning_steps: list[int], cooldown_steps: list[int]) -> list[dict]:
    sim = copy.deepcopy(raw["sim"])
    horizon = int(sim["forecast_horizon_steps"])
    dt = float(sim["dt"])
    snapshots = build_dataset(sim, seed=seed)
    train, val, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    model = fit_kinetic_topoguard(train, val, horizon, dt, seed=seed)
    _, scores, _, aligned, threshold = predict_generic(model, test)
    rows = []
    for value in warning_steps:
        rows.append(
            _metric_row(
                seed,
                "Warning window",
                f"{value * dt:.1f} s",
                aligned,
                scores,
                threshold,
                dt,
                warning_steps=int(value),
                cooldown_steps=horizon,
            )
        )
    for value in cooldown_steps:
        rows.append(
            _metric_row(
                seed,
                "Alert refractory period",
                f"{value * dt:.1f} s",
                aligned,
                scores,
                threshold,
                dt,
                warning_steps=horizon,
                cooldown_steps=int(value),
            )
        )
    for label, group_snaps, group_scores in _density_groups(aligned, scores, dt):
        rows.append(
            _metric_row(
                seed,
                "Measured event density",
                label,
                group_snaps,
                group_scores,
                threshold,
                dt,
                warning_steps=horizon,
                cooldown_steps=horizon,
            )
        )
    print(f"completed event-protocol sensitivity seed={seed}", flush=True)
    return rows


def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Ground_truth_events_per_minute",
        "Alert_Event_Precision",
        "Alert_Event_Recall",
        "Alert_Event_F1",
        "False_Alert_Events_per_minute",
    ]
    rows = []
    for (dimension, level), group in frame.groupby(["Dimension", "Level"], sort=False):
        row = {"Dimension": dimension, "Level": level, "Seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            spread = 0.0 if len(values) < 2 else 1.96 * float(values.std(ddof=1)) / len(values) ** 0.5
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95_low"] = float(values.mean() - spread)
            row[f"{metric}_ci95_high"] = float(values.mean() + spread)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Sensitivity dimension & Level & Events/min & Precision & Recall & Event F1 & False events/min \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Dimension']} & {row['Level']} & "
            f"{row['Ground_truth_events_per_minute_mean']:.2f} & "
            f"{row['Alert_Event_Precision_mean']:.3f} & "
            f"{row['Alert_Event_Recall_mean']:.3f} & "
            f"{row['Alert_Event_F1_mean']:.3f} & "
            f"{row['False_Alert_Events_per_minute_mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    dimensions = list(summary["Dimension"].drop_duplicates())
    fig, axes = plt.subplots(1, len(dimensions), figsize=(11.0, 3.4), sharey=True)
    for axis, dimension in zip(np.atleast_1d(axes), dimensions):
        group = summary[summary["Dimension"] == dimension]
        axis.plot(group["Level"], group["Alert_Event_F1_mean"], marker="o", label="Event F1")
        axis.plot(
            group["Level"],
            group["False_Alert_Events_per_minute_mean"],
            marker="s",
            label="False events/min",
        )
        axis.set_title(dimension)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean test metric")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "event_protocol_sensitivity.png", dpi=220)
    fig.savefig(output_dir / "event_protocol_sensitivity.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate event-scoring sensitivity without model or threshold reselection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--warning-steps", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--cooldown-steps", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.workers <= 1:
        for seed in args.seeds:
            rows.extend(_run_seed(seed, raw, args.warning_steps, args.cooldown_steps))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seeds))) as pool:
            for seed_rows in pool.map(
                _run_seed,
                args.seeds,
                [raw] * len(args.seeds),
                [args.warning_steps] * len(args.seeds),
                [args.cooldown_steps] * len(args.seeds),
            ):
                rows.extend(seed_rows)

    per_seed = pd.DataFrame(rows)
    summary = _summarise(per_seed)
    per_seed.to_csv(args.output_dir / "event_protocol_sensitivity_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "event_protocol_sensitivity_summary.csv", index=False)
    _write_table(summary, args.output_dir / "event_protocol_sensitivity_table.tex")
    _plot(summary, args.output_dir)
    protocol = {
        "config": relative_repo_path(args.config, ROOT),
        "seeds": args.seeds,
        "base_forecast_horizon_steps": int(raw["sim"]["forecast_horizon_steps"]),
        "dt_seconds": float(raw["sim"]["dt"]),
        "warning_window_steps": args.warning_steps,
        "cooldown_steps": args.cooldown_steps,
        "event_density_strata": "Within-seed tertiles of measured connected-to-fragmented events per test-run minute.",
        "selection_boundary": "The model, validation-selected score threshold, and test split are fixed before every descriptive sensitivity calculation.",
        "interpretation": "Descriptive protocol robustness analysis; it is not part of the locked confirmatory hypothesis family.",
    }
    (args.output_dir / "event_protocol_sensitivity_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
