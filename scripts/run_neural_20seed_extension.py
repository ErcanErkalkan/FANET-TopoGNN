from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "eaai_neural_20seed_extension.json"
MATERIALIZED_PATH = ROOT / "configs" / "eaai_neural_20seed_extension.materialized.json"
DOMAIN = "FANET-TopoGNN:EAAI:P1:neural-20seed-extension:v1"


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
    found: set[int] = set()
    for path in [*ROOT.glob("configs/*.json"), *ROOT.glob("outputs/**/*.json")]:
        if path.resolve() in excluded:
            continue
        try:
            found.update(_json_seed_values(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    for path in ROOT.glob("outputs/**/*.csv"):
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
            "A verified PyTorch backend is required; refusing scikit-learn surrogate execution."
        ) from exc
    return str(torch.__version__)


def _materialize(spec: dict) -> dict:
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    base["experiment_name"] = "eaai_neural_20seed_extension"
    base["output_dir"] = spec["output_dir"]
    base["training"].update(spec["training_overrides"])
    base["training"]["seed_list"] = spec["seed_list"]
    base["training"]["selected_models"] = spec["selected_models"]
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prepared 20-new-seed neural extension without surrogate fallbacks."
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-workers", type=int, default=1)
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = _generated_seeds()
    configured = [int(seed) for seed in spec["seed_list"]]
    if configured != expected or len(set(configured)) != 20:
        raise RuntimeError("Seed list does not match the domain-separated 20-seed stream.")
    overlap = sorted(set(configured).intersection(_prior_seeds()))
    if overlap:
        raise RuntimeError(f"Prepared seeds already appear in prior evidence: {overlap}")

    materialized = _materialize(spec)
    MATERIALIZED_PATH.write_text(json.dumps(materialized, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared 20 unused seeds in {MATERIALIZED_PATH.relative_to(ROOT)}")
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
    print(f"Verified PyTorch {torch_version}; executing all five models on all 20 seeds.")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
