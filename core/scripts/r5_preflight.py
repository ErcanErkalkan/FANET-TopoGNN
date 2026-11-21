"""R5 pre-execution gate: no simulations, no mutations to frozen research outputs.

Run from repository root:
    python scripts/r5_preflight.py --spec configs/revision_neural_20seed_full.json \
      --report outputs/r5_preflight/R5_GATE.json

Full-system execution remains blocked if SimPy or any other hard dependency is missing.
The gate is NOT a scientific validation of the 20 x 9 experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REQUIRED_MODULES = ("numpy", "pandas", "scipy", "sklearn", "torch", "simpy")
DOMAIN = "FANET-TopoGNN:REVISION:neural-full-20seed:v1"
NEURAL_MODELS = {
    "GCN", "GAT", "GraphSAGE", "PI+MLP", "FANET-TopoGNN",
    "FANET-TopoGNN (concat)", "tgcn:5", "stgcn:5", "tgn:5",
}


def canonical_seeds() -> list[int]:
    out = []
    counter = 0
    while len(out) < 20:
        d = hashlib.sha256(f"{DOMAIN}:{counter}".encode()).digest()
        seed = int.from_bytes(d[:4], "big") & 0x7FFFFFFF
        if seed and seed not in out:
            out.append(seed)
        counter += 1
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_preflight(*, root: Path, spec_path: Path, available=module_available) -> dict:
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    blockers = []
    notices = []
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = (root / str(spec["base_config"])).resolve()
    if not base.is_relative_to(root) or not base.is_file():
        raise ValueError("base_config is missing or points outside the repository")
    from fanet.config import load_config
    base_cfg = load_config(base).raw  # resolve the actual extends chain, not the raw child JSON
    sim = base_cfg["sim"]
    seeds = [int(s) for s in spec["seed_list"]]
    tasks = list(map(str, spec["selected_models"]))
    if seeds != canonical_seeds():
        blockers.append("seed list differs from the frozen 20-seed sequence")
    if len(tasks) != 14 or len(set(tasks)) != 14:
        blockers.append("the 14-model inventory is missing or duplicated")
    if set(spec.get("neural_models_requiring_pytorch", [])) != NEURAL_MODELS:
        blockers.append("the nine real PyTorch models do not match the frozen inventory")
    if not NEURAL_MODELS.issubset(set(tasks)):
        blockers.append("some neural models are absent from selected_models")
    if spec.get("protocol_status") not in ("prepared_not_executed", "prepared"):
        notices.append("protocol_status differs from original prepared-not-executed marker")
    if "revision_neural_20seed_full" not in str(spec.get("output_dir", "")):
        blockers.append("experiment output does not identify the revision project")

    time_steps = int(sim["time_steps"])
    h = int(sim.get("forecast_horizon_steps", 0))
    window = max([1] + [int(t.split(":")[1]) for t in tasks if ":" in t and t.split(":")[1].isdigit()])
    dt = float(sim.get("dt", 0))
    if h < 1 or dt <= 0 or time_steps < h + window:
        blockers.append("time_steps must accommodate the complete forecast horizon and temporal window")
    runs_per_setting = int(sim["runs_per_setting"])
    sizes = sim["swarm_sizes"]
    mobility = sim["mobility_models"]
    policies = sim.get("graph_policies", [sim.get("graph_policy", "adaptive")])
    radio = sim.get("radio_scenarios", [{"name": "default"}])
    if any(not isinstance(v, list) or not v for v in (sizes, mobility, policies, radio)) or runs_per_setting < 1:
        blockers.append("simulation factors or runs_per_setting are empty")
    runs_per_seed = runs_per_setting * len(sizes) * len(mobility) * len(policies) * len(radio)
    estimated_complete = runs_per_seed * time_steps
    estimated_labeled = runs_per_seed * max(0, time_steps - h)
    deps = {name: bool(available(name)) for name in REQUIRED_MODULES}
    missing = [name for name, ok in deps.items() if not ok]
    if missing:
        blockers.append("missing full-pipeline dependencies: " + ", ".join(missing))
    for path in ("main.py", "fanet/dataset.py", "fanet/evaluation.py", "fanet/pipeline.py", "fanet/r4_evidence.py", "fanet/reporting.py", "fanet/manuscript.py"):
        if not (root / path).is_file():
            blockers.append(f"missing project source: {path}")
    if (root / "outputs/revision_neural_20seed_full_r4_isolated").exists():
        notices.append("isolated R4 output folder already exists; do not overwrite without explicit provenance audit")
    return {
        "schema": "fanet_r5_execution_preflight_v1",
        "status": "READY_TO_ATTEMPT_FULL_PIPELINE" if not blockers else "NO_GO",
        "scientific_submission": "HOLD",
        "source_config_sha256": sha256(base),
        "frozen_spec_sha256": sha256(spec_path),
        "seed_list": seeds,
        "neural_model_count": 9,
        "model_count": len(tasks),
        "temporal_window": window,
        "horizon_steps": h,
        "dt_s": dt,
        "estimated_runs_per_seed": runs_per_seed,
        "estimated_complete_frames_per_seed": estimated_complete,
        "estimated_eligible_frames_per_seed": estimated_labeled,
        "estimated_total_eligible_frames_20seed": estimated_labeled * len(seeds),
        "dependencies": deps,
        "blockers": blockers,
        "notices": notices,
        "scope": "Pre-execution design check, not actual 20-seed training, packet PDR or scientific gate PASS",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=Path("configs/revision_neural_20seed_full.json"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", type=Path, default=Path("outputs/r5_preflight/R5_EXECUTION_GATE.json"))
    args = ap.parse_args(argv)
    report = run_preflight(root=args.root, spec_path=args.root / args.spec if not args.spec.is_absolute() else args.spec)
    report_path = args.report if args.report.is_absolute() else args.root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":report["status"], "blockers":report["blockers"],"report":str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "READY_TO_ATTEMPT_FULL_PIPELINE" else 2


if __name__ == "__main__":
    sys.exit(main())
