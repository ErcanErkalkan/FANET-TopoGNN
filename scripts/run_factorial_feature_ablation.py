from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.dataset import Snapshot, build_dataset, train_val_test_split
from fanet.geometry import pairwise_distances
from fanet.graph_utils import adjacency_from_radius, betti_zero
from fanet.training import shallow_feature_vector


DEFAULT_CONFIG = ROOT / "configs" / "publication_compact.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "factorial_feature_ablation"
SOURCES = ("graph", "topology", "kinematic")


def _summary(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(values.mean()) if values.size else 0.0,
            float(values.std()) if values.size else 0.0,
            float(np.quantile(values, 0.05)) if values.size else 0.0,
            float(np.quantile(values, 0.5)) if values.size else 0.0,
            float(np.quantile(values, 0.95)) if values.size else 0.0,
        ],
        dtype=np.float32,
    )


def _components(snapshot: Snapshot, previous: Snapshot | None, horizon_steps: int, dt: float) -> dict[str, np.ndarray]:
    current = np.asarray(
        [
            snapshot.beta_current,
            previous.beta_current if previous is not None else snapshot.beta_current,
            snapshot.beta_current - previous.beta_current if previous is not None else 0.0,
            snapshot.n_nodes / 30.0,
            snapshot.radius / 1000.0,
        ],
        dtype=np.float32,
    )
    graph = shallow_feature_vector(snapshot)
    topology = np.concatenate([snapshot.pi, _summary(snapshot.pi)]).astype(np.float32)
    tau = float(horizon_steps) * float(dt)
    current_dist = pairwise_distances(snapshot.positions.astype(float))
    projected = snapshot.positions.astype(float) + snapshot.velocities.astype(float) * tau
    projected_dist = pairwise_distances(projected)
    tri = np.triu_indices(snapshot.n_nodes, k=1)
    radius = max(float(snapshot.radius), 1e-6)
    current_pairs = current_dist[tri] / radius
    projected_pairs = projected_dist[tri] / radius
    delta = projected_pairs - current_pairs
    projected_adj = adjacency_from_radius(projected_dist, radius)
    speeds = np.linalg.norm(snapshot.velocities.astype(float), axis=1) / 30.0
    kinematic = np.concatenate(
        [
            _summary(current_pairs),
            _summary(projected_pairs),
            _summary(delta),
            _summary(speeds),
            np.asarray(
                [
                    float(betti_zero(projected_adj)),
                    float(np.mean((current_pairs <= 1.0) & (projected_pairs > 1.0))),
                    float(np.mean((current_pairs > 1.0) & (projected_pairs <= 1.0))),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return {"current": current, "graph": graph, "topology": topology, "kinematic": kinematic}


def _component_matrices(
    snapshots: list[Snapshot],
    horizon_steps: int,
    dt: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = {"current": [], "graph": [], "topology": [], "kinematic": []}
    targets = []
    risks = []
    grouped: dict[str, list[Snapshot]] = {}
    for snap in snapshots:
        grouped.setdefault(snap.run_id, []).append(snap)
    for sequence in grouped.values():
        previous = None
        for snap in sorted(sequence, key=lambda item: item.time_index):
            parts = _components(snap, previous, horizon_steps, dt)
            for name in rows:
                rows[name].append(parts[name])
            targets.append(float(snap.beta_target))
            risks.append(int(snap.frag_at_horizon))
            previous = snap
    return (
        {name: np.asarray(values) for name, values in rows.items()},
        np.asarray(targets),
        np.asarray(risks),
    )


def _select_matrix(parts: dict[str, np.ndarray], selected_sources: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([parts["current"], *(parts[name] for name in selected_sources)], axis=1)


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 19)
    ranked = [
        (f1_score(labels, scores >= threshold, zero_division=0), float(threshold))
        for threshold in candidates
    ]
    return max(ranked, key=lambda item: (item[0], item[1]))[1]


def _mean_ci(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("feature_sources"):
        row = {"feature_sources": model, "seeds": int(group["seed"].nunique())}
        for metric in ["MAE", "R2", "Risk_F1"]:
            values = group[metric].astype(float)
            spread = 1.96 * float(values.std(ddof=1)) / len(values) ** 0.5
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95_low"] = float(values.mean() - spread)
            row[f"{metric}_ci95_high"] = float(values.mean() + spread)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean")


def _run_seed(seed: int, sim: dict, combinations: list[tuple[str, ...]]) -> list[dict]:
    horizon = int(sim["forecast_horizon_steps"])
    snapshots = build_dataset(sim, seed=seed)
    train, val, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    train_parts, y_train, risk_train = _component_matrices(train, horizon, float(sim["dt"]))
    val_parts, y_val, risk_val = _component_matrices(val, horizon, float(sim["dt"]))
    test_parts, y_test, risk_test = _component_matrices(test, horizon, float(sim["dt"]))
    rows = []
    for sources in combinations:
        x_train = _select_matrix(train_parts, sources)
        x_val = _select_matrix(val_parts, sources)
        x_test = _select_matrix(test_parts, sources)
        residual_train = y_train - x_train[:, 0]
        regressor = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=2,
            random_state=seed + 101,
            n_jobs=2,
        )
        regressor.fit(x_train, residual_train)
        residual_val = regressor.predict(x_val)
        scales = np.linspace(0.0, 1.0, 11)
        scale = min(
            scales,
            key=lambda value: mean_absolute_error(
                y_val,
                np.clip(x_val[:, 0] + value * residual_val, 1.0, None),
            ),
        )
        prediction = np.clip(x_test[:, 0] + float(scale) * regressor.predict(x_test), 1.0, None)
        classifier = ExtraTreesClassifier(
            n_estimators=128,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed + 211,
            n_jobs=2,
        )
        classifier.fit(x_train, risk_train)
        val_score = classifier.predict_proba(x_val)[:, 1]
        threshold = _select_threshold(risk_val, val_score)
        test_score = classifier.predict_proba(x_test)[:, 1]
        label = "current-only" if not sources else "+".join(sources)
        rows.append(
            {
                "seed": seed,
                "feature_sources": label,
                "feature_count": int(x_train.shape[1]),
                "residual_scale": float(scale),
                "selected_threshold": threshold,
                "MAE": float(mean_absolute_error(y_test, prediction)),
                "R2": float(r2_score(y_test, prediction)),
                "Risk_F1": float(f1_score(risk_test, test_score >= threshold, zero_division=0)),
            }
        )
    print(f"completed seed={seed}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an equal-learner factorial feature-source ablation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    sim = raw["sim"]
    combinations = [
        combo
        for size in range(0, len(SOURCES) + 1)
        for combo in itertools.combinations(SOURCES, size)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(int(args.workers), len(args.seeds)))
    if worker_count == 1:
        batches = [_run_seed(seed, sim, combinations) for seed in args.seeds]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            batches = list(
                executor.map(
                    _run_seed,
                    args.seeds,
                    [sim] * len(args.seeds),
                    [combinations] * len(args.seeds),
                )
            )
    rows = [row for batch in batches for row in batch]

    per_seed = pd.DataFrame(rows)
    summary = _mean_ci(per_seed)
    per_seed.to_csv(args.output_dir / "factorial_ablation_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "factorial_ablation_summary.csv", index=False)

    plot = summary.sort_values("MAE_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].barh(plot["feature_sources"], plot["MAE_mean"], color="#2f6db3")
    axes[1].barh(plot["feature_sources"], plot["Risk_F1_mean"], color="#3b7a57")
    axes[0].set_xlabel("MAE")
    axes[1].set_xlabel("Risk F1")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "factorial_feature_ablation.png", dpi=220)
    fig.savefig(args.output_dir / "factorial_feature_ablation.pdf")
    plt.close(fig)

    protocol = {
        "config": str(args.config.relative_to(ROOT)),
        "seeds": args.seeds,
        "parallel_workers": worker_count,
        "feature_sources": list(SOURCES),
        "learner": "128-tree ExtraTrees regression on current-beta residual plus 128-tree ExtraTrees risk classifier for every row",
        "selection": "Residual scale and risk threshold selected on validation runs; final metrics use disjoint test runs.",
    }
    (args.output_dir / "factorial_ablation_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
