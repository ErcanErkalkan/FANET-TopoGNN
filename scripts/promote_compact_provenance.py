from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/publication_compact_provenance"
TARGET = ROOT / "outputs/publication_compact"
NEURAL_TASKS = {
    "GCN",
    "GAT",
    "GraphSAGE",
    "PI+MLP",
    "FANET-TopoGNN",
    "FANET-TopoGNN (concat)",
    "tgcn:5",
    "stgcn:5",
    "tgn:5",
}


def main() -> None:
    canonical_config = json.loads((ROOT / "configs/publication_compact.json").read_text(encoding="utf-8"))
    executed_config = json.loads((ROOT / "configs/publication_compact_provenance.json").read_text(encoding="utf-8"))
    for config in (canonical_config, executed_config):
        config.pop("experiment_name", None)
        config.pop("output_dir", None)
    if canonical_config != executed_config:
        raise SystemExit("Refusing promotion; canonical and provenance scientific configurations differ")

    summary_path = SOURCE / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Incomplete source: {summary_path} does not exist")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backend = summary.get("model_backend", {})
    wrong = {task: backend.get(task) for task in NEURAL_TASKS if backend.get(task) != "pytorch"}
    if wrong:
        raise SystemExit(f"Refusing promotion; neural backend mismatch: {wrong}")
    if summary.get("torch_available") is not True or summary.get("surrogate_used") is not False:
        raise SystemExit("Refusing promotion; PyTorch/surrogate summary flags are invalid")

    expected_seeds = [7, 17, 27]
    for seed in expected_seeds:
        seed_dir = SOURCE / "per_seed" / f"seed_{seed}"
        metrics_path = seed_dir / "metrics_overall.csv"
        if not metrics_path.exists():
            raise SystemExit(f"Refusing promotion; missing {metrics_path}")
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != 12:
            raise SystemExit(f"Refusing promotion; seed {seed} has {len(metrics)} model rows, expected 12")
        neural_rows = metrics[metrics["Model_Backend"] == "pytorch"]
        if len(neural_rows) != len(NEURAL_TASKS):
            raise SystemExit(f"Refusing promotion; seed {seed} has {len(neural_rows)} PyTorch rows")

    TARGET.mkdir(parents=True, exist_ok=True)
    for source_path in SOURCE.rglob("*"):
        if source_path.is_dir():
            continue
        relative = source_path.relative_to(SOURCE)
        target_path = TARGET / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    canonical_summary_path = TARGET / "summary.json"
    canonical_summary = json.loads(canonical_summary_path.read_text(encoding="utf-8"))
    canonical_summary["source_experiment_name"] = canonical_summary.get("experiment_name")
    canonical_summary["experiment_name"] = "publication_compact"
    canonical_summary["output_dir"] = "outputs/publication_compact"
    canonical_summary["backend_provenance_status"] = "verified_pytorch_rerun"
    canonical_summary_path.write_text(json.dumps(canonical_summary, indent=2), encoding="utf-8")

    runtime_path = TARGET / "runtime_profile.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["source_experiment_name"] = runtime.get("experiment_name")
    runtime["experiment_name"] = "publication_compact"
    runtime["scope"] = (
        "End-to-end provenance rerun on the local CPU host while confirmatory and data-preparation jobs "
        "were also active; this duration is not an isolated performance benchmark."
    )
    runtime["note"] = (
        "Use metrics_overall.csv for model inference measurements. The concurrent wall-clock duration is "
        "retained for execution provenance only and is not used for a speed claim."
    )
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    provenance = {
        "canonical_output": "outputs/publication_compact",
        "executed_config": "configs/publication_compact_provenance.json",
        "scientific_config_equivalence": "The executed config differs from configs/publication_compact.json only in experiment_name and output_dir.",
        "validated_seeds": expected_seeds,
        "validated_neural_backend": "pytorch",
        "torch_version": summary.get("torch_version"),
        "device": summary.get("device"),
        "surrogate_used": summary.get("surrogate_used"),
        "timing_scope": "The provenance rerun shared the host with confirmatory and data-preparation jobs; its wall-clock duration is not used as an isolated performance benchmark.",
    }
    (TARGET / "provenance_rerun.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Promoted validated compact artifacts from {SOURCE} to {TARGET}")


if __name__ == "__main__":
    main()
