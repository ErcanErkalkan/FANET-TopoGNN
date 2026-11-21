"""Prepare a safe *true SimPy pipeline* R5 smoke run; never touch frozen results.

Default is PREPARE ONLY. --run explicitly attempts to execute the full main.py
with SimPy on an isolated one-seed, three-model engineering configuration.
This is not the 20-seed x nine-model paper experiment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_smoke_config(root: Path) -> dict:
    from fanet.config import load_config
    root = Path(root)
    base = load_config(root / "configs" / "publication_neural_full_5seed.json").raw
    base["experiment_name"] = "r5_full_system_smoke_not_publication"
    base["output_dir"] = "outputs/r5_full_system_smoke_seed101"
    base["sim"].update(time_steps=30,swarm_sizes=[10],runs_per_setting=9,
                       mobility_models=["rwp", "gm", "mission"],pi_resolution=8,
                       forecast_horizon_steps=6)
    base["training"].update(seed_list=[101],selected_models=[
        "current_state_extratrees","GCN","tgcn:5"
    ], epochs=1,patience=1,hidden_dim=8,batch_size=32,
       r4_export_raw_evidence=True,r4_strict_validation_roles=True)
    base["evaluation"]["bootstrap_rounds"] = 5
    return base


def prepare(*, root:Path, config_path:Path, report_path:Path|None=None) -> dict:
    root = Path(root).resolve()
    cfg = build_smoke_config(root)
    out = root / cfg["output_dir"]
    if out.exists():
        raise FileExistsError("Pilot output already exists; refusing to reuse or overwrite")
    config_path = Path(config_path)
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != cfg:
            raise FileExistsError("Existing pilot config differs; refusing overwrite")
    else:
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    missing=[name for name in ("simpy","torch","pandas","numpy","scipy","sklearn")
             if importlib.util.find_spec(name) is None]
    result={"schema":"fanet_r5_full_system_pilot_preparation_v1",
            "status":"PREPARED_ONLY_NOT_EXECUTED",
            "missing_dependencies":missing,
            "pilot_models":cfg["training"]["selected_models"],
            "seed_list":cfg["training"]["seed_list"],
            "output_dir":cfg["output_dir"],
            "scientific_submission":"HOLD",
            "scope":"Actual full pipeline must be executed separately with --run on a SimPy-capable computer."}
    if report_path is not None:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result


def main(argv=None)->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--config",type=Path,default=None)
    ap.add_argument("--report",type=Path,default=None)
    ap.add_argument("--run",action="store_true",help="Explicitly run full SimPy/torch pipeline")
    args=ap.parse_args(argv)
    root=args.root.resolve()
    config=args.config or root/"configs/r5_full_system_smoke.json"
    if not config.is_absolute():config=root/config
    report=args.report or root/"outputs/r5_preflight/R5_FULL_SYSTEM_SMOKE_PREPARATION.json"
    if not report.is_absolute():report=root/report
    outcome=prepare(root=root,config_path=config,report_path=report)
    if not args.run:
        print(json.dumps(outcome,indent=2))
        return 0
    if outcome["missing_dependencies"]:
        print("NO_GO: missing: "+", ".join(outcome["missing_dependencies"]),file=sys.stderr)
        return 2
    # No '--resume': fresh one-seed engineering smoke only.
    subprocess.run([sys.executable,str(root/'main.py'),'--config',str(config),'--seed-workers','1'],cwd=root,check=True)
    from fanet.r4_evidence import verify_evidence
    from fanet.pipeline import _config_signature
    from fanet.config import load_config
    sig=_config_signature(load_config(config))
    seed_dir=root/"outputs/r5_full_system_smoke_seed101/per_seed/seed_101"
    for task in ("GCN","tgcn:5"):
        if not verify_evidence(seed_dir,task,sig,require_role_isolation=True):
            raise RuntimeError(f"Missing or invalid full pipeline neural evidence: {task}")
    done={"status":"ENGINEERING_SMOKE_PASS","scientific_submission":"HOLD",
          "description":"1 seed, 3 models; not a 20-seed fairness or PDR scientific claim"}
    (report.parent/"R5_FULL_SYSTEM_SMOKE_RESULT.json").write_text(json.dumps(done,indent=2)+"\n")
    print(json.dumps(done))
    return 0

if __name__=="__main__":
    sys.exit(main())
