"""Explicit, isolated full-scale R7 20-seed neural experiment launcher.

Prepare-only is safe, never trains. --run is intentionally explicit and does not
claim scientific superiority or complete packet-level verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def prepare(root:Path, *, config_path:Path, report_path:Path) -> dict:
    from fanet.config import load_config
    from scripts.r5_preflight import canonical_seeds
    root=Path(root).resolve();config_path=Path(config_path).resolve();report_path=Path(report_path).resolve()
    spec=json.loads((root/'configs/revision_neural_20seed_full.json').read_text())
    if list(map(int,spec['seed_list'])) != canonical_seeds():
        raise ValueError('Original 20-seed integrity check failed')
    if len(spec['selected_models'])!=14 or len(spec['neural_models_requiring_pytorch'])!=9:
        raise ValueError('Frozen fourteen-model inventory mismatch')
    config=load_config(root/spec['base_config']).raw
    config['experiment_name']='r7_full_20seed_strict_common_support_not_submitted'
    config['output_dir']='outputs/r7_full_20seed_isolated'
    training=config['training']
    training.update(spec['training_overrides'])
    training['seed_list']=list(map(int,spec['seed_list']))
    training['selected_models']=list(spec['selected_models'])
    training['r4_export_raw_evidence']=True
    training['r4_strict_validation_roles']=True
    training['r7_full_truth_event_diagnostics']=True
    training['r7_export_all_model_scores']=True
    if training.get('epochs') != 100:
        raise ValueError('The full frozen training budget of 100 epochs changed')
    if config['sim'].get('forecast_horizon_steps')!=6 or config['sim'].get('time_steps')!=200:
        raise ValueError('Frozen horizon or trajectory time steps changed')
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:
            raise FileExistsError('A different R7 scientific config already exists')
    else:
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    deps={k:importlib.util.find_spec(k) is not None for k in ('torch','numpy','scipy','pandas','sklearn','simpy')}
    output=root/config['output_dir']
    report={
        'schema':'fanet_r7_full_20seed_execution_preflight_v1',
        'status':'PREPARED_ONLY_NOT_EXECUTED',
        'new_output_directory':str(output.relative_to(root)),
        'seed_count':20, 'model_count':14,'neural_model_count':9,'epochs_per_neural_model':100,
        'expected_neural_checkpoints':180,
        'forecast_horizon_steps':6,'temporal_common_window':5,
        'runs_per_seed':108,'full_frames_per_seed':21600,
        'strict_h6_frames_per_seed':20952,
        'estimated_test_rows_per_seed_per_model':6840,
        'source_spec_sha256':hashlib.sha256((root/'configs/revision_neural_20seed_full.json').read_bytes()).hexdigest(),
        'generated_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'dependencies':deps,
        'model_training_enabled_without_simpy':all(deps[x] for x in ('torch','numpy','scipy','pandas','sklearn')),
        'packet_simpy_stage_enabled':deps['simpy'],
        'known_open_gates':['classical neural matched validation calibration',
                            'SimPy packet experiment',
                            'h6 v2_1/v3_1 lineage',
                            'full closed loop radio/motion budgets',
                            'publication ethics/bibliography/style'],
        'scientific_submission':'HOLD',
    }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--report',type=Path)
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--seed-workers',type=int,default=1)
    args=ap.parse_args()
    root=args.root.resolve()
    cfg=args.config or root/'configs/r7_full_20seed_isolated.json'
    report_path=args.report or root/'reports/R7_FULL_20SEED_PREFLIGHT.json'
    gate=prepare(root,config_path=cfg,report_path=report_path)
    if not args.run:
        print(json.dumps({'status':'PREPARED_ONLY','expected_neural_checkpoints':gate['expected_neural_checkpoints'],
                          'output':gate['new_output_directory'],'packet_simpy_available':gate['packet_simpy_stage_enabled']}))
        return 0
    if not gate['model_training_enabled_without_simpy']:
        raise RuntimeError('Required AI dependencies missing')
    output_dir=root/gate['new_output_directory']
    if output_dir.exists() and not args.resume:
        raise FileExistsError('Isolated R7 full 20-seed output already exists; use --resume after independent audit')
    command=[sys.executable,str(root/'main.py'),'--config',str(cfg),'--seed-workers',str(max(1,args.seed_workers))]
    if args.resume:command.append('--resume')
    subprocess.run(command,cwd=root,check=True)
    from fanet.pipeline import _seed_complete
    from fanet.config import load_config
    c=load_config(cfg)
    misses=[seed for seed in c.training['seed_list'] if not _seed_complete(output_dir/'per_seed'/f'seed_{seed}',c)]
    if misses:
        raise RuntimeError('20-seed training incomplete: '+str(misses))
    print('COMPLETE: 20 seeds with 14 model cached results and 9 neural evidence files per seed.')
    print('Scientific conclusion remains HOLD until full event/p-value/parity/packet/method audits pass.')
    return 0

if __name__=='__main__':sys.exit(main())
