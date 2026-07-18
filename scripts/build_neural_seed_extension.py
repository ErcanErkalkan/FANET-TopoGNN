from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fanet.provenance import build_file_manifest
CANONICAL_DIR = ROOT / "outputs/publication_compact"
EXTENSION_DIR = ROOT / "outputs/publication_neural_extension"
SOURCE_GATED_DIR = ROOT / "outputs/source_gated_development"
PAPER_LIKE_DIR = ROOT / "outputs/paper_like_submission"
FULL_RERUN_DIR = ROOT / "outputs/publication_neural_full_5seed_v1"
DEFAULT_OUTPUT = ROOT / "outputs/publication_neural_5seed_extension"
PAPER_SHORT_TABLE = ROOT / "paper/tables/generated/neural_seed_extension_table.tex"
PAPER_FULL_TABLE = ROOT / "paper/tables/generated/neural_seed_extension_full_table.tex"

SEEDS = (7, 17, 27, 37, 47)
EXPECTED_MODELS = (
    "Current-state persistence baseline",
    "Current-state ExtraTrees",
    "Shallow ML",
    "Kinetic-TopoGuard",
    "Source-Gated Kinetic-TopoGuard",
    "GCN",
    "GAT",
    "GraphSAGE",
    "PI+MLP",
    "FANET-TopoGNN",
    "FANET-TopoGNN (concat)",
    "T-GCN (w=5)",
    "STGCN (w=5)",
    "TGN (w=5)",
)
NEURAL_MODELS = frozenset(EXPECTED_MODELS[5:])
CANONICAL_LABELS = {
    "tgcn:5": "T-GCN (w=5)",
    "stgcn:5": "STGCN (w=5)",
    "tgn:5": "TGN (w=5)",
}
METRIC_COLS = (
    "MAE",
    "R2",
    "Risk_F1",
    "Risk_Brier",
    "Risk_ECE",
    "Alert_Event_Precision",
    "Alert_Event_Recall",
    "Alert_Event_F1",
    "False_Alert_Events_per_minute",
    "Inference_ms",
)


def canonical_model(name: object) -> str:
    text = CANONICAL_LABELS.get(str(name), str(name))
    return "Shallow ML" if text.startswith("Shallow ML") else text


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _read_seed_metrics(base: Path, seed: int) -> pd.DataFrame:
    path = base / "per_seed" / f"seed_{seed}" / "metrics_overall.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["Model"] = frame["Model"].map(canonical_model)
    frame["source_artifact"] = path.relative_to(ROOT).as_posix()
    return frame


def collect_complete_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all executed rows and explicitly record every missing/failed run."""
    rerun_paths = [FULL_RERUN_DIR / "per_seed" / f"seed_{seed}" / "metrics_overall.csv" for seed in SEEDS]
    if all(path.is_file() for path in rerun_paths):
        rerun = pd.concat([_read_seed_metrics(FULL_RERUN_DIR, seed) for seed in SEEDS], ignore_index=True)
        rerun = rerun[rerun["Model"].isin(EXPECTED_MODELS)].copy()
        observed = set(zip(rerun["seed"].astype(int), rerun["Model"].astype(str)))
        expected = {(seed, model) for seed in SEEDS for model in EXPECTED_MODELS}
        if observed == expected:
            return rerun.sort_values(["Model", "seed"]).reset_index(drop=True), pd.DataFrame(columns=["seed", "Model", "source", "reason"])
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    legacy_models = set(EXPECTED_MODELS) - {"Current-state ExtraTrees", "Source-Gated Kinetic-TopoGuard"}
    for seed in SEEDS:
        base = CANONICAL_DIR if seed in {7, 17, 27} else EXTENSION_DIR
        try:
            frame = _read_seed_metrics(base, seed)
        except Exception as exc:
            for model in sorted(legacy_models):
                failures.append({"seed": seed, "Model": model, "source": base.relative_to(ROOT).as_posix(), "reason": repr(exc)})
        else:
            frames.append(frame[frame["Model"].isin(legacy_models)].copy())

    source_path = SOURCE_GATED_DIR / "per_seed_metrics.csv"
    if source_path.is_file():
        source = pd.read_csv(source_path)
        source["Model"] = source["Model"].map(canonical_model)
        source["Model_Backend"] = "scikit_learn"
        source["source_artifact"] = source_path.relative_to(ROOT).as_posix()
        selected = source[source["Model"].isin({"Current-state ExtraTrees", "Source-Gated Kinetic-TopoGuard"})].copy()
        missing_persistence = source[
            (source["Model"] == "Current-state persistence baseline") & source["seed"].isin({37, 47})
        ].copy()
        frames.extend([selected, missing_persistence])
    else:
        for seed in SEEDS:
            for model in ("Current-state ExtraTrees", "Source-Gated Kinetic-TopoGuard"):
                failures.append({"seed": seed, "Model": model, "source": source_path.relative_to(ROOT).as_posix(), "reason": "source metrics file missing"})

    for seed in (37, 47):
        try:
            shallow = _read_seed_metrics(PAPER_LIKE_DIR, seed)
        except Exception as exc:
            failures.append({"seed": seed, "Model": "Shallow ML", "source": PAPER_LIKE_DIR.relative_to(ROOT).as_posix(), "reason": repr(exc)})
        else:
            frames.append(shallow[shallow["Model"] == "Shallow ML"].copy())

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined[combined["seed"].isin(SEEDS) & combined["Model"].isin(EXPECTED_MODELS)].copy()
        duplicates = combined.duplicated(["seed", "Model"], keep=False)
        if duplicates.any():
            rows = combined.loc[duplicates, ["seed", "Model", "source_artifact"]].to_dict("records")
            raise RuntimeError(f"Duplicate model/seed rows detected: {rows}")

    observed = set(zip(combined.get("seed", []), combined.get("Model", [])))
    already_failed = {(int(row["seed"]), str(row["Model"])) for row in failures}
    for seed in SEEDS:
        for model in EXPECTED_MODELS:
            if (seed, model) not in observed and (seed, model) not in already_failed:
                failures.append({"seed": seed, "Model": model, "source": "", "reason": "model/seed row absent from executed artifacts"})
    failure_frame = pd.DataFrame(failures, columns=["seed", "Model", "source", "reason"])
    return combined.sort_values(["Model", "seed"]).reset_index(drop=True), failure_frame


def validate_coverage(frame: pd.DataFrame, failures: pd.DataFrame) -> None:
    if not failures.empty:
        raise RuntimeError(f"Incomplete model runs; see training_failures.csv: {failures.to_dict('records')}")
    expected = {(seed, model) for seed in SEEDS for model in EXPECTED_MODELS}
    observed = set(zip(frame["seed"].astype(int), frame["Model"].astype(str)))
    if observed != expected:
        raise RuntimeError(f"Model/seed coverage mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    neural = frame[frame["Model"].isin(NEURAL_MODELS)]
    wrong = neural[neural["Model_Backend"].astype(str) != "pytorch"]
    if not wrong.empty:
        raise RuntimeError(f"PyTorch-named model used a surrogate/non-PyTorch backend: {wrong[['seed', 'Model', 'Model_Backend']].to_dict('records')}")
    canonical_sim = json.loads((ROOT / "configs/publication_compact.json").read_text(encoding="utf-8"))["sim"]
    extension_sim = json.loads((ROOT / "configs/publication_neural_extension.json").read_text(encoding="utf-8"))["sim"]
    gated_sim = json.loads((SOURCE_GATED_DIR / "protocol.json").read_text(encoding="utf-8"))["resolved_config"]["sim"]
    if canonical_sim != extension_sim or canonical_sim != gated_sim:
        raise RuntimeError("Source experiments do not use identical simulation/split definitions")
    # Kinetic-TopoGuard was executed in both source families. Exact agreement is an
    # independent guard that the deterministic realization and held-out split match.
    gated_reference = pd.read_csv(SOURCE_GATED_DIR / "per_seed_metrics.csv")
    gated_reference = gated_reference[gated_reference["Model"] == "Kinetic-TopoGuard"].sort_values("seed")
    primary_reference = frame[frame["Model"] == "Kinetic-TopoGuard"].sort_values("seed")
    for column in ("MAE", "R2", "Alert_Event_F1", "False_Alert_Events_per_minute"):
        if not np.allclose(primary_reference[column], gated_reference[column], equal_nan=True):
            raise RuntimeError(f"Shared-realization cross-check failed for Kinetic-TopoGuard {column}")


def _torch_parameter_counts() -> dict[str, tuple[int, int]]:
    try:
        from fanet.models import (
            FANETTopoGNN,
            FANETTopoGNNConcat,
            GATEncoder,
            GCNEncoder,
            GraphRegressor,
            PIRegressor,
            SAGEEncoder,
            TemporalRegressor,
        )
    except Exception as exc:  # pragma: no cover - exercised when torch is unavailable
        raise RuntimeError("PyTorch implementations are required to audit neural parameter counts") from exc
    hidden, input_dim, pi_dim, dropout = 48, 4, 16 * 16, 0.15
    models = {
        "GCN": GraphRegressor(GCNEncoder(input_dim, hidden, dropout), hidden, dropout),
        "GAT": GraphRegressor(GATEncoder(input_dim, hidden, dropout), hidden, dropout),
        "GraphSAGE": GraphRegressor(SAGEEncoder(input_dim, hidden, dropout), hidden, dropout),
        "PI+MLP": PIRegressor(pi_dim, hidden, dropout),
        "FANET-TopoGNN": FANETTopoGNN(input_dim, pi_dim, hidden, dropout),
        "FANET-TopoGNN (concat)": FANETTopoGNNConcat(input_dim, pi_dim, hidden, dropout),
        "T-GCN (w=5)": TemporalRegressor(GCNEncoder(input_dim, hidden, dropout), hidden, "tgcn", dropout),
        "STGCN (w=5)": TemporalRegressor(GCNEncoder(input_dim, hidden, dropout), hidden, "stgcn", dropout),
        "TGN (w=5)": TemporalRegressor(GCNEncoder(input_dim, hidden, dropout), hidden, "tgn", dropout),
    }
    return {
        name: (sum(parameter.numel() for parameter in model.parameters()), sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        for name, model in models.items()
    }


def build_model_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    counts = _torch_parameter_counts()
    torch_version = json.loads((CANONICAL_DIR / "summary.json").read_text(encoding="utf-8"))["torch_version"]
    sklearn_version = metadata.version("scikit-learn")
    specifications = {
        "Current-state persistence baseline": ("current component count", 1, "none", "none", "none", "none", "none", "deterministic no-change", "fixed score threshold 0.5"),
        "Current-state ExtraTrees": ("current", 1, "ExtraTrees regressor/classifier", "not applicable", "not applicable", "not applicable", "not applicable", "validation event F1 with predefined tie-break", "validation-selected threshold"),
        "Shallow ML": ("density, minimum distance, degree summaries, largest component, clustering", 1, "validation-selected shallow learner", "model-specific", "model-specific", "model-specific", "model-specific", "validation MAE", "fixed score threshold 0.5"),
        "Kinetic-TopoGuard": ("current, graph, topology, kinematic", 1, "ExtraTrees/GBR/RF candidate set", "model-specific", "model-specific", "model-specific", "model-specific", "validation MAE and validation risk objective", "validation-selected threshold"),
        "Source-Gated Kinetic-TopoGuard": ("current, graph, topology, kinematic source experts", 1, "ExtraTrees experts + ridge/logistic meta-model", "model-specific", "model-specific", "model-specific", "model-specific", "OOF training plus validation hyperparameter/calibration selection", "validation event F1 with predefined tie-break"),
        "GCN": ("node position/velocity and adjacency", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "GAT": ("node position/velocity and adjacency", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "GraphSAGE": ("node position/velocity and adjacency", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "PI+MLP": ("persistence image (16x16)", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "FANET-TopoGNN": ("node position/velocity, adjacency, persistence image", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "FANET-TopoGNN (concat)": ("node position/velocity, adjacency, persistence image", 1, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "T-GCN (w=5)": ("five graph snapshots: node position/velocity and adjacency", 5, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "STGCN (w=5)": ("five graph snapshots: node position/velocity and adjacency", 5, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
        "TGN (w=5)": ("five graph snapshots: node position/velocity and adjacency", 5, "Adam", 0.001, 64, 18, "patience 5", "validation MSE", "fixed score threshold 0.5"),
    }
    rows = []
    for model in EXPECTED_MODELS:
        group = frame[frame["Model"] == model]
        backend = str(group["Model_Backend"].iloc[0])
        total, trainable = counts.get(model, (0 if model == "Current-state persistence baseline" else math.nan, 0 if model == "Current-state persistence baseline" else math.nan))
        feature_set, window, optimizer, lr, batch, epochs, stopping, objective, threshold = specifications[model]
        recorded_times = group["Training_Time_s"].dropna().astype(float) if "Training_Time_s" in group else pd.Series(dtype=float)
        rows.append({
            "Model": model,
            "input_feature_set": feature_set,
            "temporal_window": window,
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "parameter_count_scope": "exact torch architecture" if model in NEURAL_MODELS else ("zero deterministic parameters" if model == "Current-state persistence baseline" else "not comparable/not retained in legacy fitted artifacts"),
            "optimizer": optimizer,
            "learning_rate": lr,
            "batch_size": batch,
            "max_epochs": epochs,
            "early_stopping": stopping,
            "validation_objective": objective,
            "threshold_selection_rule": threshold,
            "training_time_seconds": float(recorded_times.mean()) if not recorded_times.empty else math.nan,
            "training_time_status": "mean of five recorded per-seed training durations" if len(recorded_times) == len(SEEDS) else "not recorded per model by legacy runs; no value inferred from total runtime",
            "inference_time_ms_mean": float(group["Inference_ms"].mean()),
            "backend": backend,
            "backend_version": torch_version if backend == "pytorch" else (sklearn_version if backend == "scikit_learn" else "not applicable"),
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(map(str, sorted(group["seed"].astype(int).unique()))),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in EXPECTED_MODELS:
        group = frame[frame["Model"] == model]
        row: dict[str, object] = {"Model": model, "seed_count": int(group["seed"].nunique())}
        for column in METRIC_COLS:
            values = group[column].astype(float).to_numpy()
            mean = float(np.mean(values))
            if len(values) > 1:
                delta = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
            else:
                delta = math.nan
            row.update({f"{column}_mean": mean, f"{column}_ci95_low": mean - delta, f"{column}_ci95_high": mean + delta})
        row["backends"] = ",".join(sorted(set(group["Model_Backend"].astype(str))))
        rows.append(row)
    return pd.DataFrame(rows)


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def write_full_table(metrics: pd.DataFrame, protocol: pd.DataFrame, path: Path) -> None:
    merged = metrics.merge(protocol[["Model", "parameter_count", "seed_count"]], on=["Model", "seed_count"], how="left")
    lines = [
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Model & Seeds & Params & Event P & Event R & Event $F_1$ [95\\% CI] & False/min & MAE \\\\", "\\midrule",
    ]
    for _, row in merged.iterrows():
        params = "--" if pd.isna(row["parameter_count"]) else f"{int(row['parameter_count']):,}"
        lines.append(
            f"{_latex_escape(str(row['Model']))} & {int(row['seed_count'])} & {params} & "
            f"{row['Alert_Event_Precision_mean']:.3f} & {row['Alert_Event_Recall_mean']:.3f} & "
            f"{row['Alert_Event_F1_mean']:.3f} [{row['Alert_Event_F1_ci95_low']:.3f}, {row['Alert_Event_F1_ci95_high']:.3f}] & "
            f"{row['False_Alert_Events_per_minute_mean']:.2f} & {row['MAE_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_short_table(metrics: pd.DataFrame, path: Path) -> None:
    # Predeclared compact selection: primary non-neural methods plus the best neural MAE and event-F1 rows.
    neural = metrics[metrics["Model"].isin(NEURAL_MODELS)]
    selected = {
        "Current-state persistence baseline", "Current-state ExtraTrees", "Kinetic-TopoGuard", "Source-Gated Kinetic-TopoGuard",
        str(neural.sort_values(["MAE_mean", "Model"]).iloc[0]["Model"]),
        str(neural.sort_values(["Alert_Event_F1_mean", "Model"], ascending=[False, True]).iloc[0]["Model"]),
    }
    subset = metrics[metrics["Model"].isin(selected)].copy()
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Model & Seeds & MAE & Event $F_1$ & False/min \\\\", "\\midrule"]
    for _, row in subset.iterrows():
        lines.append(f"{_latex_escape(str(row['Model']))} & {int(row['seed_count'])} & {row['MAE_mean']:.3f} & {row['Alert_Event_F1_mean']:.3f} & {row['False_Alert_Events_per_minute_mean']:.2f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build a non-selective five-seed baseline comparison from executed artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed, failures = collect_complete_metrics()
    failures.to_csv(args.output_dir / "training_failures.csv", index=False)
    validate_coverage(per_seed, failures)
    protocol = build_model_protocol(per_seed)
    metrics = aggregate_metrics(per_seed)
    per_seed.to_csv(args.output_dir / "per_seed_full_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "full_metrics.csv", index=False)
    protocol.to_csv(args.output_dir / "model_protocol.csv", index=False)
    write_full_table(metrics, protocol, args.output_dir / "neural_seed_extension_full_table.tex")
    write_short_table(metrics, args.output_dir / "neural_seed_extension_table.tex")
    shutil.copy2(args.output_dir / "neural_seed_extension_full_table.tex", PAPER_FULL_TABLE)
    shutil.copy2(args.output_dir / "neural_seed_extension_table.tex", PAPER_SHORT_TABLE)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "publication_neural_5seed_extension_full_coverage",
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "expected_models": list(EXPECTED_MODELS),
        "complete_models": metrics[metrics["seed_count"] == len(SEEDS)]["Model"].tolist(),
        "source_artifacts": sorted(per_seed["source_artifact"].unique().tolist()),
        "surrogate_policy": "PyTorch-named rows must have Model_Backend=pytorch; otherwise the build fails.",
        "short_table_selection_rule": "primary non-neural methods plus the neural model with lowest MAE and the neural model with highest event F1; ties are resolved by model name",
        "full_table_location": "paper/tables/generated/neural_seed_extension_full_table.tex",
        "training_time_limitation": "Legacy runs did not retain per-model training time. Missing values remain explicit and are not inferred from total runtime.",
        "package_versions": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "simpy", "torch", "torch-geometric")
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "source_files": build_file_manifest(
            [
                ROOT / path for path in sorted(per_seed["source_artifact"].unique())
            ]
            + [
                ROOT / "configs/publication_compact.json",
                ROOT / "configs/publication_neural_extension.json",
                ROOT / "configs/source_gated_development.json",
                Path(__file__).resolve(),
                ROOT / "fanet/models.py",
                ROOT / "fanet/training.py",
                ROOT / "fanet/pipeline.py",
            ],
            ROOT,
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "full_coverage_protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote complete {len(metrics)}-model comparison for {len(SEEDS)} shared seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
