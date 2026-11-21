"""Independent R6 replay of the R5 GCN engineering pilot.

Scientific scope is deliberately narrow: one seed, one GCN, one epoch. Recreates
*real* FANET motion/radio snapshots, loads the saved torch state_dict (never a
whole pickle model), independently checks every archive row, and recomputes
calibration + the selection threshold without test-label tuning.

No SimPy/packet-network claim and no model-to-model superiority claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_TASK = 'GCN'
REQUIRED = {'checkpoint_state_dict.pt', 'validation_calibration_fit.csv',
            'validation_threshold_selection.csv', 'test.csv',
            'calibration.json', 'threshold_selection.json'}


class ReplayFailure(RuntimeError):
    pass


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise ReplayFailure(msg)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    _assert(len(rows)>0,f'Empty CSV: {path.name}')
    return rows


def _key(item) -> tuple[str,int]:
    if isinstance(item,dict):
        return (str(item['run_id']),int(item['time_index']))
    return (str(item.run_id),int(item.time_index))


def _strict_row_compare(rows: list[dict], expected: list[object], horizon: int,
                        origin: str, label: str) -> dict[tuple[str,int],dict]:
    seen={}
    for row in rows:
        key=_key(row)
        _assert(key not in seen,f'Duplicate row key: {label}: {key}')
        seen[key]=row
    expected_map={_key(s):s for s in expected}
    _assert(len(expected_map)==len(expected),f'Duplicate expected snapshot: {label}')
    _assert(set(seen)==set(expected_map),f'Row identity mismatch: {label}; missing={len(set(expected_map)-set(seen))}, extra={len(set(seen)-set(expected_map))}')
    for key,row in seen.items():
        snap=expected_map[key]
        _assert(row['split_group_id']==str(snap.split_group_id),f'Group mismatch {label}/{key}')
        for col,v in [('future_time_index',int(snap.future_time_index)),('frag_at_horizon',int(snap.frag_at_horizon))]:
            _assert(int(row[col])==v,f'{col} mismatch {label}/{key}')
        _assert(int(row['future_time_index'])-int(row['time_index'])==horizon,f'Future horizon mismatch: {key}')
        for col,v in [('beta_current',float(snap.beta_current)),('beta_target',float(snap.beta_target))]:
            _assert(math.isclose(float(row[col]),v,rel_tol=0,abs_tol=1e-10),f'{col} mismatch {label}/{key}')
        _assert(row['score_origin']==origin,f'Score origin mismatch: {label}/{key}')
        if label=='fit':
            _assert(row['risk_probability'].strip()=='',f'Fit set should not have risk probabilities: {key}')
        else:
            p=float(row['risk_probability'])
            _assert(math.isfinite(p) and 0 <= p <= 1,f'Invalid probability: {label}/{key}')
        _assert(math.isfinite(float(row['predicted_beta'])),f'Invalid predicted beta {label}/{key}')
    return seen


def _pilot_views(root:Path,seed:int):
    from fanet.dataset import build_dataset_views,train_val_test_split
    from fanet.r4_evidence import split_early_stopping_validation,_validation_partition
    base=json.loads((root/'configs'/'publication_compact.json').read_text(encoding='utf-8'))
    sim=dict(base['sim'])
    sim.update(time_steps=30,swarm_sizes=[10],runs_per_setting=9,
               mobility_models=['rwp','gm','mission'],pi_resolution=8,
               forecast_horizon_steps=6)
    views=build_dataset_views(sim,seed=seed)
    train,val,test,_=train_val_test_split(
        views.forecast_eligible,split_seed=2026,
        stratify_by=('mobility','n_nodes','link_model','graph_policy','radio_scenario'),
        return_mapping=True)
    early,remaining=split_early_stopping_validation(val,seed)
    fit_mask,select_mask=_validation_partition(remaining,seed)
    fit=[s for s,ok in zip(remaining,fit_mask) if ok]
    select=[s for s,ok in zip(remaining,select_mask) if ok]
    return views,train,early,fit,select,test


def _select_threshold_independent(scores: np.ndarray, labels: np.ndarray) -> tuple[float,dict]:
    candidates=[]
    for i in range(1,20):
        threshold=round(0.05*i,2)
        yp=scores>=threshold
        tp=int(np.sum(yp & (labels==1)))
        fp=int(np.sum(yp & (labels==0)))
        fn=int(np.sum((~yp)&(labels==1)))
        f1=2*tp/max(2*tp+fp+fn,1)
        candidates.append(dict(threshold=threshold,risk_f1=f1,fp=fp,tp=tp,fn=fn))
    c=sorted(candidates,key=lambda x:(-x['risk_f1'],x['fp'],-x['threshold']))[0]
    return c['threshold'],c


def audit(*,root:Path,evidence:Path,source_report:Path|None=None) -> dict:
    root=Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    import torch
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    from fanet.models import GCNEncoder,GraphRegressor
    from fanet.evaluation import predict_torch

    root=Path(root).resolve()
    evidence=Path(evidence).resolve()
    manifest=json.loads((evidence/'MANIFEST.json').read_text(encoding='utf-8'))
    _assert(manifest['schema']=='fanet_r4_neural_evidence_v2','Wrong evidence schema')
    _assert(manifest['task_name']==EXPECTED_TASK,'Only R5 GCN pilot is supported by this R6 source replay')
    _assert(manifest['status']=='RAW_EVIDENCE_EXPORTED_NOT_SCIENTIFICALLY_REPLAYED','Unexpected historical evidence status')
    _assert(manifest['training_config'].get('type')=='small_engineering_pilot_not_paper','Not R5 engineering pilot')
    _assert(manifest['training_config']['epochs']==1,'R5 pilot should have one epoch')
    seed=int(manifest['seed'])
    _assert(seed==101,'R6 frozen R5 pilot seed must be 101')
    _assert(int(manifest['horizon_steps'])==6,'Unexpected horizon')
    _assert(set(manifest['artifact_sha256'])==REQUIRED,'Evidence artifact inventory mismatch')
    for filename,recorded in manifest['artifact_sha256'].items():
        _assert(sha256(evidence/filename)==recorded,f'SHA256 mismatch: {filename}')
    for filename,recorded in manifest['source_sha256'].items():
        _assert(sha256(root/'fanet'/filename)==recorded,f'Source SHA256 drift: {filename}')

    torch.set_num_threads(1)
    views,train,early,fit,select,test=_pilot_views(root,seed)
    _assert(len(views.full_trajectory)==3240 and len(views.forecast_eligible)==2592,'Unexpected source trajectory counts')
    full_truth={_key(s):s for s in views.full_trajectory}
    _assert(len(full_truth)==len(views.full_trajectory),'Full-truth key duplicate')
    horizon=int(manifest['horizon_steps'])
    for snap in views.forecast_eligible:
        future=full_truth.get((str(snap.run_id),int(snap.time_index)+horizon))
        _assert(future is not None,'Missing future truth frame')
        _assert(float(snap.beta_target)==float(future.beta_current),'Replayed target differs from future truth')

    fit_rows=_read_csv(evidence/'validation_calibration_fit.csv')
    select_rows=_read_csv(evidence/'validation_threshold_selection.csv')
    test_rows=_read_csv(evidence/'test.csv')
    fit_map=_strict_row_compare(fit_rows,fit,horizon,'uncalibrated_beta_prediction_do_not_use_as_risk_probability','fit')
    select_map=_strict_row_compare(select_rows,select,horizon,'calibrated_probability','selection')
    test_map=_strict_row_compare(test_rows,test,horizon,'calibrated_probability','test')
    roles={'train':train,'early':early,'fit':fit,'select':select,'test':test}
    groups={k:{str(s.split_group_id) for s in snaps} for k,snaps in roles.items()}
    for name,g in groups.items():
        _assert(g,f'Empty group role {name}')
        for other,gg in groups.items():
            if name!=other:_assert(not(g&gg),f'Split group leakage: {name} / {other}')
    _assert(set(manifest['early_stopping_groups'])==groups['early'],'Early stopping group mismatch')
    cal=json.loads((evidence/'calibration.json').read_text(encoding='utf-8'))
    _assert(set(cal['groups_fit'])==groups['fit'] and set(cal['groups_threshold'])==groups['select'],'Calibration-group metadata mismatch')
    _assert(cal['classes']==[0,1],'Wrong class interpretation')

    # Instantiate known architecture directly: independent of the saved training
    # object and without any new model fitting. Only *weights_only* torch.load.
    model=GraphRegressor(GCNEncoder(train[0].node_features.shape[1],8,0.0),8,0.0)
    state=torch.load(evidence/'checkpoint_state_dict.pt',map_location='cpu',weights_only=True)
    _assert(isinstance(state,dict) and bool(state),'Missing state dict')
    model.load_state_dict(state,strict=True)
    model.eval()
    max_beta_diff=0.0
    max_prob_diff=0.0
    predicted_by_role={}
    for label,snapshots,rows in [('fit',fit,fit_map),('selection',select,select_map),('test',test,test_map)]:
        pred=predict_torch(model,snapshots)
        predicted_by_role[label]=np.asarray(pred,dtype=float)
        for s,score in zip(snapshots,pred):
            err=abs(float(rows[_key(s)]['predicted_beta'])-float(score))
            max_beta_diff=max(max_beta_diff,err)
            _assert(err<=2e-6,f'Checkpoint replay mismatch for {label}/{_key(s)}: {err}')

    fit_y=np.asarray([int(s.frag_at_horizon) for s in fit])
    _assert(set(fit_y.tolist())=={0,1},'Invalid calibration classes')
    logit=LogisticRegression(C=1.0,max_iter=1000,random_state=seed%(2**32-1))
    logit.fit(predicted_by_role['fit'].reshape(-1,1),fit_y)
    _assert(abs(float(logit.coef_[0,0])-float(cal['coefficient']))<2e-6,'Independent logistic coefficient mismatch')
    _assert(abs(float(logit.intercept_[0])-float(cal['intercept']))<2e-6,'Independent logistic intercept mismatch')
    for label,snaps,rows in [('selection',select,select_map),('test',test,test_map)]:
        probabilities=expit(float(cal['coefficient'])*predicted_by_role[label]+float(cal['intercept']))
        for snap,prob in zip(snaps,probabilities):
            err=abs(float(rows[_key(snap)]['risk_probability'])-float(prob))
            max_prob_diff=max(max_prob_diff,err)
            _assert(err<=1e-8,f'Calibrated probability replay mismatch {label}/{_key(snap)}: {err}')

    selection_ordered=np.asarray([float(select_map[_key(s)]['risk_probability']) for s in select])
    selection_y=np.asarray([int(s.frag_at_horizon) for s in select])
    threshold,decision=_select_threshold_independent(selection_ordered,selection_y)
    saved_selection=json.loads((evidence/'threshold_selection.json').read_text(encoding='utf-8'))
    _assert(manifest['threshold_selection']==saved_selection,'Manifest threshold differs from artifact')
    _assert(abs(float(saved_selection['selected_threshold'])-threshold)<1e-12,'Threshold selection replay mismatch')
    for col,value in decision.items():
        _assert(math.isclose(float(saved_selection['selection_metrics'][col]),float(value),rel_tol=0,abs_tol=1e-9),f'Threshold metric mismatch {col}')

    full_events=0
    from collections import defaultdict
    seqs=defaultdict(list)
    for s in views.full_trajectory:seqs[str(s.run_id)].append(s)
    for seq in seqs.values():
        seq=sorted(seq,key=lambda x:int(x.time_index))
        full_events+=sum(a.beta_current<=1 and b.beta_current>1 for a,b in zip(seq,seq[1:]))
    _assert(full_events==163,'Full truth event count differs')
    test_ps=np.asarray([float(test_map[_key(s)]['risk_probability']) for s in test])
    test_ys=np.asarray([int(s.frag_at_horizon) for s in test])
    pred=test_ps>=threshold
    tp=int(np.sum(pred&(test_ys==1)))
    fp=int(np.sum(pred&(test_ys==0)))
    fn=int(np.sum((~pred)&(test_ys==1)))
    tn=int(np.sum((~pred)&(test_ys==0)))
    f1=float(2*tp/max(2*tp+fp+fn,1))
    brier=float(np.mean((test_ps-test_ys)**2))
    if source_report is not None:
        report=json.loads(Path(source_report).read_text(encoding='utf-8'))
        _assert(report['full_snapshots']==len(views.full_trajectory),'R5 report snapshot mismatch')
        _assert(report['valid_forecast_snapshots']==len(views.forecast_eligible),'R5 report eligible snapshot mismatch')
        _assert(report['full_trajectory_fragmentation_events']==full_events,'R5 report event mismatch')
        _assert(report['test_rows_exported']==len(test),'R5 report test rows mismatch')
        _assert(report['checkpoint_sha256']==sha256(evidence/'checkpoint_state_dict.pt'),'R5 report checkpoint mismatch')
    return {
        'schema':'fanet_r6_independent_checkpoint_replay_v1','status':'ENGINEERING_PILOT_REPLAY_PASS',
        'seed':seed,'model':'GCN','epochs':1,'torch_checkpoint_reloaded':True,
        'full_snapshots':len(views.full_trajectory),'strict_forecast_rows':len(views.forecast_eligible),
        'full_truth_events':full_events,
        'roles':{k:{'groups':len(v),'rows':len(roles[k])} for k,v in groups.items()},
        'replayed_prediction_rows':len(fit)+len(select)+len(test),
        'test_rows':len(test),'max_checkpoint_prediction_absolute_error':max_beta_diff,
        'max_calibrated_probability_absolute_error':max_prob_diff,
        'selected_threshold_validation_only':threshold,
        'engineering_test_sample_risk_f1':f1,'engineering_test_sample_brier':brier,
        'engineering_test_confusion':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},
        'important_limits':['Not 20-seed or nine-model R4 experiment',
                            'Not SimPy packet simulation',
                            'Replayed sample Risk-F1 is NOT event-level Event-F1',
                            'Single engineering seed: not evidence of model superiority'],
        'scientific_submission':'HOLD'
    }


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description='Replay R5 GCN evidence using independent source and checkpoint')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--r5-report',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:
        result=audit(root=args.root,evidence=args.evidence,source_report=args.r5_report)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            if args.output.exists():
                raise ReplayFailure('Refusing to overwrite existing replay audit')
            args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        return 0
    except (ReplayFailure,OSError,ValueError,RuntimeError,KeyError) as exc:
        print(json.dumps({'status':'ENGINEERING_REPLAY_FAIL','reason':str(exc),'scientific_submission':'HOLD'},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
