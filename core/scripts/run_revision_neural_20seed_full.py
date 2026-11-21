from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "revision_neural_20seed_full.json"
MATERIALIZED_PATH = ROOT / "configs" / "revision_neural_20seed_full.materialized.json"
DOMAIN = "FANET-TopoGNN:REVISION:neural-full-20seed:v1"

NEURAL_DISPLAY_NAMES = {
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

NON_NEURAL_REQUIRED = {
    "Current-state persistence baseline",
    "Current-state ExtraTrees",
    "Kinetic-TopoGuard",
    "Source-Gated Kinetic-TopoGuard",
}


def _generated_seeds(count: int = 20) -> list[int]:
    seeds: list[int] = []
    counter = 0
    while len(seeds) < count:
        digest = hashlib.sha256(f"{DOMAIN}:{counter}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed and seed not in seeds:
            seeds.append(seed)
        counter += 1
    return seeds


def _json_seed_values(value, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(_json_seed_values(child, str(child_key).lower()))
    elif isinstance(value, list):
        if "seed" in key:
            for item in value:
                if isinstance(item, int):
                    found.add(int(item))
        for item in value:
            found.update(_json_seed_values(item, key))
    elif isinstance(value, int) and "seed" in key:
        found.add(int(value))
    return found


def _prior_seeds() -> set[int]:
    excluded = {SPEC_PATH.resolve(), MATERIALIZED_PATH.resolve()}
    # These frozen post-review studies intentionally share the revision stream.
    revision_outputs = {
        "revision_motion_model_ablation", "revision_horizon_20seed",
        "revision_event_protocol_20seed", "revision_neural_20seed_full",
        "revision_packet_loss_sensitivity",
    }

    def shared_revision_output(path: Path) -> bool:
        relative = path.relative_to(ROOT)
        return relative.parts[0] == "outputs" and relative.parts[1] in revision_outputs

    found: set[int] = set()
    for path in [*ROOT.glob("configs/*.json"), *ROOT.glob("outputs/**/*.json")]:
        if path.resolve() in excluded or shared_revision_output(path):
            continue
        try:
            found.update(_json_seed_values(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    for path in ROOT.glob("outputs/**/*.csv"):
        if shared_revision_output(path):
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda name: "seed" in str(name).lower())
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        for column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            found.update(int(value) for value in numeric.astype(int).unique())
    return found


def _verified_torch() -> str:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "A verified PyTorch backend is required; refusing surrogate execution."
        ) from exc
    return str(torch.__version__)


def _materialize(spec: dict) -> dict:
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    base["experiment_name"] = "revision_neural_20seed_full"
    base["output_dir"] = spec["output_dir"]
    base.setdefault("training", {})
    base["training"].update(spec["training_overrides"])
    base["training"]["seed_list"] = spec["seed_list"]
    base["training"]["selected_models"] = spec["selected_models"]
    return base


def _normalize_model_name(name: str) -> str:
    return str(name).strip()


def _validate_completed_outputs(spec: dict) -> None:
    output_dir = ROOT / spec["output_dir"]
    missing_files: list[str] = []
    coverage_errors: list[str] = []
    backend_errors: list[str] = []

    for seed in spec["seed_list"]:
        path = output_dir / "per_seed" / f"seed_{int(seed)}" / "metrics_overall.csv"
        if not path.is_file():
            missing_files.append(str(path.relative_to(ROOT)))
            continue

        frame = pd.read_csv(path)
        if "Model" not in frame.columns or "Model_Backend" not in frame.columns:
            coverage_errors.append(
                f"seed={seed}: metrics_overall.csv lacks Model or Model_Backend"
            )
            continue

        names = {_normalize_model_name(value) for value in frame["Model"].dropna()}
        if len(frame) != len(spec["selected_models"]):
            coverage_errors.append(
                f"seed={seed}: expected {len(spec['selected_models'])} model rows, found {len(frame)}"
            )

        for required in NON_NEURAL_REQUIRED:
            if required not in names:
                coverage_errors.append(f"seed={seed}: missing {required}")
        if not any(name.startswith("Shallow ML") for name in names):
            coverage_errors.append(f"seed={seed}: missing Shallow ML row")

        for neural in NEURAL_DISPLAY_NAMES:
            rows = frame[frame["Model"].astype(str).str.strip() == neural]
            if rows.empty:
                coverage_errors.append(f"seed={seed}: missing neural model {neural}")
                continue
            backends = {str(value).strip().lower() for value in rows["Model_Backend"].dropna()}
            if backends != {"pytorch"}:
                backend_errors.append(
                    f"seed={seed}: {neural} backend(s)={sorted(backends)}; expected ['pytorch']"
                )

    if missing_files or coverage_errors or backend_errors:
        messages = []
        if missing_files:
            messages.append("Missing per-seed files:\n  " + "\n  ".join(missing_files))
        if coverage_errors:
            messages.append("Coverage errors:\n  " + "\n  ".join(coverage_errors))
        if backend_errors:
            messages.append("Backend errors:\n  " + "\n  ".join(backend_errors))
        raise RuntimeError("\n".join(messages))

    audit = {
        "status": "complete",
        "experiment_name": "revision_neural_20seed_full",
        "n_seeds": len(spec["seed_list"]),
        "n_models_per_seed": len(spec["selected_models"]),
        "n_neural_models_per_seed": len(NEURAL_DISPLAY_NAMES),
        "neural_backend_required": "pytorch",
        "partial_results_allowed_for_claims": False,
        "seed_generation_domain": DOMAIN,
        "seed_list": spec["seed_list"],
        "claim_boundary": spec["claim_rule"],
    }
    (output_dir / "revision_fairness_completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the post-review 20-new-seed full model-family fairness extension."
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-workers", type=int, default=1)
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = _generated_seeds()
    configured = [int(seed) for seed in spec["seed_list"]]
    if configured != expected or len(set(configured)) != 20:
        raise RuntimeError("Seed list does not match the domain-separated revision stream.")

    if len(spec["selected_models"]) != 14:
        raise RuntimeError("Fairness protocol requires exactly 14 manuscript-facing models.")
    if set(spec["neural_models_requiring_pytorch"]) != NEURAL_DISPLAY_NAMES:
        raise RuntimeError("Neural model inventory does not match the frozen nine-model set.")

    overlap = sorted(set(configured).intersection(_prior_seeds()))
    if overlap:
        raise RuntimeError(f"Revision seeds already appear in prior evidence: {overlap}")

    materialized = _materialize(spec)
    MATERIALIZED_PATH.write_text(json.dumps(materialized, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared 20 unused revision seeds in {MATERIALIZED_PATH.relative_to(ROOT)}")
    print("Frozen model inventory: 14 total, including 9 PyTorch neural models.")
    if args.prepare_only:
        print("Status: prepared only; no results were executed or claimed.")
        return

    torch_version = _verified_torch()
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--config",
        str(MATERIALIZED_PATH),
        "--seed-workers",
        str(max(1, args.seed_workers)),
    ]
    if args.resume:
        command.append("--resume")
    print(f"Verified PyTorch {torch_version}; executing 14 models on all 20 revision seeds.")
    subprocess.run(command, cwd=ROOT, check=True)
    _validate_completed_outputs(spec)
    print("PASS: 14-model x 20-seed coverage complete; all nine neural backends are PyTorch.")


if __name__ == "__main__":
    main()
