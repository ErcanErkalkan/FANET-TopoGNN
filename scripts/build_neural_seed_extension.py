from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "outputs/publication_compact"
EXTENSION_DIR = ROOT / "outputs/publication_neural_extension"
DEFAULT_OUTPUT = ROOT / "outputs/publication_neural_5seed_extension"
PAPER_TABLE = ROOT / "paper/tables/generated/neural_seed_extension_table.tex"

MODELS = [
    "Kinetic-TopoGuard",
    "GCN",
    "GAT",
    "GraphSAGE",
    "PI+MLP",
    "FANET-TopoGNN",
    "FANET-TopoGNN (concat)",
    "tgcn:5",
    "stgcn:5",
    "tgn:5",
]
CANONICAL_LABELS = {
    "tgcn:5": "T-GCN (w=5)",
    "stgcn:5": "STGCN (w=5)",
    "tgn:5": "TGN (w=5)",
}
NEURAL_LABELS = {CANONICAL_LABELS.get(model, model) for model in MODELS if model != "Kinetic-TopoGuard"}
METRIC_COLS = [
    "MAE",
    "R2",
    "Risk_F1",
    "Risk_PR_AUC",
    "Risk_ROC_AUC",
    "Alert_Event_Recall",
    "Alert_Event_F1",
    "False_Alert_Events_per_minute",
    "False_Alarms_per_minute",
    "Inference_ms",
]


def _canonical_model(name: object) -> str:
    return CANONICAL_LABELS.get(str(name), str(name))


def _read_seed_metrics(base: Path, seed: int) -> pd.DataFrame:
    path = base / "per_seed" / f"seed_{seed}" / "metrics_overall.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["Model"] = df["Model"].map(_canonical_model)
    keep = {_canonical_model(name) for name in MODELS}
    return df[df["Model"].isin(keep)].copy()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, group in df.groupby("Model", sort=False):
        row = {"Model": model, "seed_count": int(group["seed"].nunique())}
        for col in METRIC_COLS:
            values = group[col].astype(float)
            mean = float(values.mean())
            sem = float(values.sem(ddof=1)) if len(values) > 1 else 0.0
            delta = 1.96 * sem
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_low"] = mean - delta
            row[f"{col}_ci95_high"] = mean + delta
        backends = sorted(set(group.get("Model_Backend", pd.Series(dtype=str)).astype(str)))
        row["backends"] = ",".join(backends)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def _write_table(metrics: pd.DataFrame, output_path: Path, models: list[str]) -> None:
    display = metrics[metrics["Model"].isin(models)].copy()
    display = display.sort_values("MAE_mean")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & Seeds & MAE & Sample $F_1$ & Event $F_1$ & Backend \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        backend = "PyTorch" if row["backends"] == "pytorch" else "scikit-learn"
        lines.append(
            f"{row['Model']} & {int(row['seed_count'])} & "
            f"{float(row['MAE_mean']):.3f} & {float(row['Risk_F1_mean']):.3f} & "
            f"{float(row['Alert_Event_F1_mean']):.3f} & "
            f"{backend} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the canonical 3-seed run with the additional neural extension seeds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frames = []
    for seed in [7, 17, 27]:
        frames.append(_read_seed_metrics(CANONICAL_DIR, seed))
    for seed in [37, 47]:
        frames.append(_read_seed_metrics(EXTENSION_DIR, seed))
    per_seed = pd.concat(frames, ignore_index=True)

    seed_counts = per_seed.groupby("Model")["seed"].nunique()
    missing = [model for model in ["Kinetic-TopoGuard", *sorted(NEURAL_LABELS)] if int(seed_counts.get(model, 0)) != 5]
    if missing:
        raise RuntimeError(f"Expected five seeds for every extension model; incomplete models: {missing}")

    neural_backends = per_seed[per_seed["Model"].isin(NEURAL_LABELS)].groupby("Model")["Model_Backend"].unique()
    wrong_backend = {model: sorted(map(str, values)) for model, values in neural_backends.items() if set(map(str, values)) != {"pytorch"}}
    if wrong_backend:
        raise RuntimeError(f"Non-PyTorch neural backend detected: {wrong_backend}")

    metrics = _aggregate(per_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    _write_table(
        metrics,
        args.output_dir / "neural_seed_extension_table.tex",
        [
            "Kinetic-TopoGuard",
            "STGCN (w=5)",
            "TGN (w=5)",
            "PI+MLP",
            "FANET-TopoGNN (concat)",
        ],
    )
    _write_table(
        metrics,
        args.output_dir / "neural_seed_extension_full_table.tex",
        ["Kinetic-TopoGuard", *sorted(NEURAL_LABELS)],
    )

    payload = {
        "experiment_name": "publication_neural_5seed_extension",
        "source_dirs": [str(CANONICAL_DIR.relative_to(ROOT)), str(EXTENSION_DIR.relative_to(ROOT))],
        "seeds": [7, 17, 27, 37, 47],
        "n_seeds": 5,
        "models": metrics["Model"].astype(str).tolist(),
        "neural_backend": "pytorch",
        "source_cache_versions": {
            "canonical": json.loads((CANONICAL_DIR / "summary.json").read_text(encoding="utf-8")).get("cache_version"),
            "extension": json.loads((EXTENSION_DIR / "summary.json").read_text(encoding="utf-8")).get("cache_version"),
        },
        "scope": "Five-seed extension for neural-family stability; primary confirmatory inference remains the 20-seed Kinetic-TopoGuard-versus-shallow profile.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "neural_seed_extension_table.tex", PAPER_TABLE)
    print(f"Wrote {args.output_dir.relative_to(ROOT)}")
    print(metrics[["Model", "seed_count", "MAE_mean", "Risk_F1_mean", "backends"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
