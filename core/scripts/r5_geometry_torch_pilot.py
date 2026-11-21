"""R5 independent geometry/radio/real-PyTorch engineering pilot (NOT full SimPy).

Requires actual FANET modules; never uses made-up snapshots or a surrogate backend.
Stores fresh outputs in a new, explicitly provided directory and refuses overwrite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_pilot(*, root: Path, output_dir: Path, seed: int = 101, epochs: int = 1) -> dict:
    root = Path(root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Pilot output already exists: refusing replacement of evidence")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    from fanet.dataset import build_dataset_views, train_val_test_split
    from fanet.r4_evidence import (
        split_early_stopping_validation, _validation_partition,
        export_torch_evidence, verify_evidence, evidence_location,
    )
    from fanet.training import train_torch_model
    from fanet.evaluation import predict_generic
    import torch

    torch.set_num_threads(1)
    sim = json.loads((root / "configs/publication_compact.json").read_text(encoding="utf-8"))["sim"]
    # An intentionally reduced *engineering-only* design, retaining real motion,
    # physical radio, paired policy/scenario variants, and group-wise splitting.
    sim.update(time_steps=30, swarm_sizes=[10], runs_per_setting=9,
               mobility_models=["rwp", "gm", "mission"],
               pi_resolution=8, forecast_horizon_steps=6)
    started = time.perf_counter()
    views = build_dataset_views(sim, seed=seed)
    tr, val, te, _ = train_val_test_split(
        views.forecast_eligible, split_seed=2026,
        stratify_by=("mobility", "n_nodes", "link_model", "graph_policy", "radio_scenario"),
        return_mapping=True,
    )
    early, val_remainder = split_early_stopping_validation(val, seed)
    fit_mask, select_mask = _validation_partition(val_remainder, seed)
    groups = {role:{str(s.split_group_id) for s in seq} for role,seq in {
        "train":tr,"early_stopping":early,"calibration": [s for s,ok in zip(val_remainder,fit_mask) if ok],
        "threshold_selection":[s for s,ok in zip(val_remainder,select_mask) if ok],"test":te}.items()}
    for a, x in groups.items():
        if not x:
            raise ValueError(f"Empty validation role: {a}")
        for b,y in groups.items():
            if a != b and x & y:
                raise RuntimeError(f"Leaked split_group_id between {a} and {b}")
    full_keys = {(s.run_id,s.time_index) for s in views.full_trajectory}
    eligible_keys = {(s.run_id,s.time_index) for s in views.forecast_eligible}
    if not eligible_keys.issubset(full_keys):
        raise RuntimeError("Forecast view not derived from full trajectory")
    h=int(sim["forecast_horizon_steps"])
    by_run = {}
    for s in views.full_trajectory:
        by_run.setdefault(s.run_id, []).append(s)
    for run,seq in by_run.items():
        seen={int(s.time_index):s for s in seq}
        for snap in seq:
            expected = seen.get(int(snap.time_index)+h)
            if (snap.run_id, snap.time_index) in eligible_keys:
                if expected is None or snap.beta_target != expected.beta_current:
                    raise RuntimeError("Invalid strict target / future time index")
            elif expected is not None:
                raise RuntimeError("An eligible forecast frame was incorrectly dropped")
    for name,seq in [("calibration",[s for s,ok in zip(val_remainder,fit_mask) if ok]),
                     ("threshold_selection",[s for s,ok in zip(val_remainder,select_mask) if ok])]:
        if set(s.frag_at_horizon for s in seq) != {0,1}:
            raise RuntimeError(f"{name} has no positive/negative outcome in the pilot")

    # Actual PyTorch GCN on real FANET snapshots, *not* 20-seed full-scale R4.
    fitted = train_torch_model("GCN", tr, early, {
        "hidden_dim":8, "dropout":0.0, "learning_rate":0.001,
        "weight_decay":0.0, "batch_size":32, "epochs":epochs, "patience":1,
    }, int(tr[0].pi.shape[0]), seed=seed)
    if not isinstance(fitted.model, torch.nn.Module):
        raise RuntimeError("PyTorch backend was not used")
    v_hat, _, _, v_aligned, _ = predict_generic(fitted, val_remainder)
    t_hat, _, _, t_aligned, _ = predict_generic(fitted, te)
    output_dir.mkdir(parents=True)
    seed_dir=output_dir / f"pilot_seed_{seed}"
    manifest=export_torch_evidence(
        seed_dir=seed_dir,seed=seed,task_name="GCN",model=fitted.model,
        train_cfg={"epochs":epochs,"type":"small_engineering_pilot_not_paper"},
        config_sha256=hashlib.sha256(json.dumps(sim,sort_keys=True).encode()).hexdigest(),
        horizon_steps=h,validation_snapshots=v_aligned,validation_beta_hat=v_hat,
        test_snapshots=t_aligned,test_beta_hat=t_hat,
        early_stopping_group_ids=sorted(groups["early_stopping"]),
    )
    evidence=evidence_location(seed_dir,"GCN")
    if not verify_evidence(seed_dir,"GCN",manifest["config_sha256"],require_role_isolation=True):
        raise RuntimeError("Real-model evidence integrity verification failed")
    state=torch.load(evidence / "checkpoint_state_dict.pt",map_location="cpu",weights_only=True)
    if not state or not all(bool(torch.isfinite(t).all()) for t in state.values()):
        raise RuntimeError("State dict missing or invalid")
    with (evidence / "test.csv").open(newline="",encoding="utf-8") as f:
        test_rows=list(csv.DictReader(f))
    if len(test_rows) != len(te):
        raise RuntimeError("Saved predictions and test split disagree")
    result={
        "schema":"fanet_r5_real_geometry_torch_pilot_v1",
        "status":"ENGINEERING_PILOT_PASS",
        "paper_evidence":"NOT_VALID_FOR_PUBLICATION_COMPARISON",
        "full_system_simpy_test":"NOT_EXECUTED",
        "seed":seed,"epochs":epochs,"model":"GCN", "backend":"PyTorch",
        "full_snapshots":len(views.full_trajectory),
        "valid_forecast_snapshots":len(views.forecast_eligible),
        "trajectory_runs":len(by_run),
        "full_trajectory_fragmentation_events":sum(
            s2.beta_current > 1 and s1.beta_current <= 1
            for seq in by_run.values()
            for s1,s2 in zip(sorted(seq,key=lambda s:s.time_index),sorted(seq,key=lambda s:s.time_index)[1:])
        ),
        "role_rows": {n:sum(str(s.split_group_id) in g for s in views.forecast_eligible)
                      for n,g in groups.items()},
        "role_groups":{n:len(g) for n,g in groups.items()},
        "role_outcomes":{n:{"positives":sum(s.frag_at_horizon for s in views.forecast_eligible if str(s.split_group_id) in g),
                            "negatives":sum(1-s.frag_at_horizon for s in views.forecast_eligible if str(s.split_group_id) in g)}
                         for n,g in groups.items()},
        "checkpoint_sha256":hashlib.sha256((evidence/"checkpoint_state_dict.pt").read_bytes()).hexdigest(),
        "evidence_export_pass":True,"test_rows_exported":len(test_rows),
        "selected_threshold":manifest["threshold_selection"]["selected_threshold"],
        "elapsed_seconds":round(time.perf_counter()-started,3),
        "scientific_submission":"HOLD",
    }
    (output_dir / "R5_REAL_GEOMETRY_TORCH_PILOT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--epochs",type=int,default=1)
    x=ap.parse_args(argv)
    result=run_pilot(root=x.root,output_dir=x.output_dir,seed=x.seed,epochs=x.epochs)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
