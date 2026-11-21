"""R7 engineering evidence gate: actual 14-model pilot, common support,
independent all-nine checkpoint replay, archived R2/R4/R5 arithmetic.

This gate is intentionally not the journal-submission gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from fanet.config import load_config
from fanet.dataset import build_dataset_views, train_val_test_split
from fanet.evaluation import alert_event_metrics_full_trajectory
from fanet.pipeline import _config_signature, _seed_complete
from fanet.r7_protocol import select_full_test_trajectory,validate_support_inventory
from fanet.r4_evidence import verify_evidence
from scripts.r7_historical_gate import gate as historical_gate


def _source_hashes(root:Path)->dict:
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'fanet').glob('*.py'))}


def audit(root:Path,config_path:Path, *, seed:int, output:Path|None=None)->dict:
    root=Path(root).resolve();config_path=Path(config_path).resolve()
    cfg=load_config(config_path)
    if not cfg.training.get('r7_full_truth_event_diagnostics') or not cfg.training.get('r7_export_all_model_scores'):
        raise ValueError('R7 full-truth evaluation and all-model raw scores are required')
    if not cfg.training.get('r4_strict_validation_roles'):
        raise ValueError('R4 isolation roles must be enabled')
    frozen=json.loads((root/'configs/revision_neural_20seed_full.json').read_text())
    expected=list(frozen['selected_models'])
    if list(cfg.training['selected_models'])!=expected:raise ValueError('14-model inventory has drifted')
    seed_dir=root/cfg.raw['output_dir']/'per_seed'/f'seed_{seed}'
    if not _seed_complete(seed_dir,cfg):raise ValueError('Incomplete model cache or missing evidence')
    metrics=pd.read_csv(seed_dir/'metrics_overall.csv')
    truth_table=pd.read_csv(seed_dir/'r7_full_truth_event_metrics.csv')
    if len(metrics)!=14 or len(truth_table)!=14:raise ValueError('Incomplete one-seed metric tables')
    if len(set(metrics['Model']))!=14 or len(set(truth_table['Model']))!=14:
        raise ValueError('Duplicate model label')
    score_paths=[seed_dir/'r7_scored_predictions'/ (''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in task)+'.csv') for task in expected]
    # Identical naming to the source executor's _cache_stem, including Unicode safety.
    from fanet.pipeline import _cache_stem
    score_paths=[seed_dir/'r7_scored_predictions'/f'{_cache_stem(task)}.csv' for task in expected]
    names=[json.loads(p.with_suffix('.json').read_text())['model'] for p in score_paths]
    coverage=validate_support_inventory(score_paths,names)
    if coverage['rows_per_model']!=480:
        raise ValueError('Engineering frozen 5-frame decision support differs from source pilot')
    views=build_dataset_views(cfg.sim,seed=seed)
    _,_,test=train_val_test_split(views.forecast_eligible,
            split_seed=int(cfg.sim.get('split_seed',seed)),
            stratify_by=tuple(list(cfg.sim.get('split_stratify_by',['mobility']))+['graph_policy','radio_scenario']))
    full_test=select_full_test_trajectory(views.full_trajectory,test)
    by_time={(str(s.run_id),int(s.time_index)):s for s in test}
    common_keys=None;summary=[];event_truth=None
    for path,model in zip(score_paths,names):
        with path.open(newline='',encoding='utf-8') as fh:records=list(csv.DictReader(fh))
        keys=[(r['run_id'],int(r['time_index'])) for r in records]
        if any(key not in by_time for key in keys):raise ValueError('Unrecognized test decision times')
        if common_keys is None:common_keys=set(keys)
        elif set(keys)!=common_keys:raise ValueError('Model mismatch in common decision times')
        snaps=[by_time[k] for k in keys]
        for snap,row in zip(snaps,records):
            if int(row['future_time_index'])!=int(snap.future_time_index) or row['split_group_id']!=str(snap.split_group_id):
                raise ValueError('Stored test truth alignment mismatch')
            if float(row['beta_target'])!=float(snap.beta_target) or int(row['frag_at_horizon'])!=int(snap.frag_at_horizon):
                raise ValueError('Stored test labels differ from regenerated geometry truth')
        risks=np.asarray([float(r['risk_score']) for r in records])
        threshold=float(records[0]['selected_threshold'])
        replay=alert_event_metrics_full_trajectory(snaps,risks,full_test,threshold=threshold,
                         dt=cfg.sim['dt'],horizon_steps=cfg.sim['forecast_horizon_steps'])
        record=truth_table.loc[truth_table['Model']==model]
        model_metric=metrics.loc[metrics['Model']==model]
        if len(record)!=1 or len(model_metric)!=1:raise ValueError('Recorded model ledger mismatch')
        record=record.iloc[0];model_metric=model_metric.iloc[0]
        for key,val in replay.items():
            if abs(float(record[key])-float(val))>1e-10:
                raise ValueError(f'Full event recomputation disagrees: {model}/{key}')
        if abs(float(model_metric['R7_Full_Timeline_Event_F1'])-float(replay['Alert_Event_F1']))>1e-10:
            raise ValueError('Per-model metrics table disagrees with full-truth ledger')
        n=int(replay['Ground_Truth_Fragmentation_Events'])
        if event_truth is None:event_truth=n
        elif n!=event_truth:raise ValueError('Models used different ground-truth event inventories')
        positive=int(np.sum(risks>=threshold))
        summary.append({'model':model,'score_kind':records[0]['score_kind'],
                        'n_scored':len(records),'n_positive_decisions':positive,
                        'risk_positive_decision_fraction':round(positive/len(records),6),
                        'risk_f1':float(model_metric['Risk_F1']),
                        'legacy_event_f1':float(model_metric['Alert_Event_F1']),
                        'full_event_f1':float(replay['Alert_Event_F1']),
                        'full_truth_events':n})
    if event_truth!=33 or len(full_test)!=720:raise ValueError('R7 pilot geometry/full truth count drift')
    neural=frozen['neural_models_requiring_pytorch']
    if len(neural)!=9:raise ValueError('Incorrect frozen neural inventory')
    cfg_signature=_config_signature(cfg)
    for name in neural:
        if not verify_evidence(seed_dir,name,cfg_signature,require_role_isolation=True):
            raise ValueError('Neural checkpoint provenance does not pass: '+name)
    replay=json.loads((root/'reports/R7_NINE_NEURAL_INDEPENDENT_REPLAY.json').read_text())
    if replay['status']!='PASS' or replay['neural_checkpoint_count']!=9 or replay['total_test_predictions']!=4320:
        raise ValueError('Independent nine-neural replay not passed')
    if any(float(replay[k])>1e-7 for k in ['max_beta_prediction_abs_error','max_probability_abs_error','max_calibration_coef_abs_error']):
        raise ValueError('Replay numeric deviation above tolerance')
    historic=historical_gate(root/'references/historical')
    data={
        'schema':'fanet_r7_integrated_engineering_gate_v1',
        'status':'SEED_LEVEL_14_MODEL_ENGINEERING_PASS_PUBLISHING_HOLD',
        'seed':seed,'training_epochs_per_neural_model':int(cfg.training['epochs']),
        'frozen_model_inventory_count':len(expected),
        'neural_checkpoint_verified_count':9,
        'source_used':'FANET geometry/physical radio; not SimPy packet queue',
        'full_truth_test_rows':len(full_test),
        'full_truth_fragmentation_events':event_truth,
        'forecast_scored_test_rows_per_model':coverage['rows_per_model'],
        'all_model_scores_common_support':True,
        'nine_model_numeric_replay':{'test_predictions':replay['total_test_predictions'],
             'max_error':max(float(replay[k]) for k in ['max_beta_prediction_abs_error','max_probability_abs_error','max_calibration_coef_abs_error'])},
        'model_results':summary,
        'historical_cross_archive_status':historic['status'],
        'frozen_20seed_full_training':'NOT_EXECUTED',
        'full_packet_simpy':'NOT_EXECUTED_SIMPY_DEPENDENCY_UNAVAILABLE',
        'classical_neural_validation_parity':'NOT_VERIFIED_CLASSICAL_MODELS_USE_ORIGINAL_VALIDATION_PROTOCOL',
        'full_14_model_run_postprocessing':'PARTIAL_TIMED_OUT_AFTER_SEED_TABLES',
        'source_sha256':_source_hashes(root),
        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'research_release':'HOLD',
    }
    if output is not None:
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return data

if __name__=='__main__':
    output=ROOT/'reports/R7_INTEGRATED_GATE.json'
    result=audit(ROOT,ROOT/'configs/r7_one_seed_fourteen_models.json',seed=101,output=output)
    print(json.dumps({k:result[k] for k in ('status','neural_checkpoint_verified_count','full_truth_fragmentation_events','forecast_scored_test_rows_per_model','research_release')},ensure_ascii=False))
