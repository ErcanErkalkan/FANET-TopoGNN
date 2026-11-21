"""Independent frozen checkpoint+calibration replay for all nine PyTorch models.

No model fitting except independent LogisticRegression calibration verification;
no test data is used for fitting or decision thresholds.
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
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

NEURAL = ["GCN", "GAT", "GraphSAGE", "PI+MLP", "FANET-TopoGNN",
          "FANET-TopoGNN (concat)", "tgcn:5", "stgcn:5", "tgn:5"]


def _by_key(path: Path) -> dict[tuple[str,int],dict]:
    with path.open(newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    d={}
    for row in rows:
        key=(str(row['run_id']),int(row['time_index']))
        if key in d:raise ValueError('Duplicate recorded output key '+str(key))
        d[key]=row
    return d


def _make_model(task:str, train, cfg):
    import torch
    from fanet.models import (
        GCNEncoder,GATEncoder,SAGEEncoder,GraphRegressor,PIRegressor,
        FANETTopoGNN,FANETTopoGNNConcat,TemporalRegressor,
    )
    inp=int(train[0].node_features.shape[1]);pi=int(train[0].pi.shape[0]);h=int(cfg['hidden_dim']);drop=float(cfg['dropout'])
    if task=='GCN':return GraphRegressor(GCNEncoder(inp,h,drop),h,drop)
    if task=='GAT':return GraphRegressor(GATEncoder(inp,h,drop),h,drop)
    if task=='GraphSAGE':return GraphRegressor(SAGEEncoder(inp,h,drop),h,drop)
    if task=='PI+MLP':return PIRegressor(pi,h,drop)
    if task=='FANET-TopoGNN':return FANETTopoGNN(inp,pi,h,drop)
    if task=='FANET-TopoGNN (concat)':return FANETTopoGNNConcat(inp,pi,h,drop)
    kind, window = task.split(':')
    if int(window)!=5 or kind not in ('tgcn','stgcn','tgn'):raise ValueError('Unrecognized temporal task')
    return TemporalRegressor(GCNEncoder(inp,h,drop),h,kind,drop)


def _predict(model, task:str, snaps:list):
    from fanet.evaluation import predict_torch
    from fanet.dataset import split_by_run
    from fanet.r4_evidence import restrict_to_frozen_temporal_support
    window=int(task.split(':')[1]) if ':' in task else None
    pred=predict_torch(model,snaps,temporal_window=window)
    if window is None:
        aligned=snaps
    else:
        grouped=split_by_run(snaps)
        aligned=[]
        for _,seq in sorted(grouped.items()):
            aligned.extend(sorted(seq,key=lambda s:s.time_index)[window-1:])
    aligned,pred,_=restrict_to_frozen_temporal_support(
        snaps,aligned,pred,np.zeros(len(pred)),window=5)
    return aligned,np.asarray(pred,dtype=float)


def _verify_rows(rows, snapshots, preds, label:str, *, tolerance:float=2e-6):
    if len(rows)!=len(snapshots):raise ValueError(f'{label} row count mismatch')
    maxerr=0.
    keys=set()
    for s,p in zip(snapshots,preds):
        key=(str(s.run_id),int(s.time_index));keys.add(key)
        r=rows.get(key)
        if r is None:raise ValueError(f'{label} missing key {key}')
        e=abs(float(r['predicted_beta'])-float(p));maxerr=max(maxerr,e)
        if e>tolerance:raise ValueError(f'{label} checkpoint score mismatch {key}: {e}')
        for col,truth in [('split_group_id',str(s.split_group_id)),
                          ('future_time_index',str(int(s.future_time_index))),
                          ('frag_at_horizon',str(int(s.frag_at_horizon)))]:
            if str(r[col])!=truth:raise ValueError(f'{label} identity/target mismatch {col}/{key}')
        if float(r['beta_target'])!=float(s.beta_target):
            raise ValueError(f'{label} beta target mismatch {key}')
    if len(keys)!=len(rows):raise ValueError(f'{label} extra keys')
    return maxerr


def audit(root:Path,config_path:Path,seed:int, *, output:Path|None=None) -> dict:
    import torch
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from fanet.dataset import build_dataset_views,train_val_test_split
    from fanet.r4_evidence import (_validation_partition,split_early_stopping_validation,
                                   verify_evidence,evidence_location,select_probability_threshold)
    from fanet.pipeline import _config_signature
    from fanet.config import load_config
    root=Path(root).resolve();c=load_config(config_path)
    if seed not in c.training['seed_list']:raise ValueError('Seed not in config')
    if list(c.training['selected_models'])!=list(__import__('json').loads((root/'configs/revision_neural_20seed_full.json').read_text())['selected_models']):
        raise ValueError('Audit requires all fourteen frozen model families')
    torch.set_num_threads(1)
    views=build_dataset_views(c.sim,seed=seed)
    tr,val,te= train_val_test_split(views.forecast_eligible,
                    split_seed=int(c.sim.get('split_seed',seed)),
                    stratify_by=tuple(list(c.sim.get('split_stratify_by',['mobility']))+['graph_policy','radio_scenario']))
    early,remain=split_early_stopping_validation(val,seed)
    sig=_config_signature(c)
    seed_dir=root/c.raw['output_dir']/'per_seed'/f'seed_{seed}'
    summary=[]
    role_groups={'train':{str(s.split_group_id) for s in tr},'early':{str(s.split_group_id) for s in early},
                 'test':{str(s.split_group_id) for s in te}}
    max_prediction_error=0.;max_probability_error=0.;max_coef_error=0.
    for task in NEURAL:
        if not verify_evidence(seed_dir,task,sig,require_role_isolation=True):
            raise ValueError('Recorded evidence SHA/role provenance failed '+task)
        path=evidence_location(seed_dir,task)
        manifest=json.loads((path/'MANIFEST.json').read_text(encoding='utf-8'))
        model=_make_model(task,tr,manifest['training_config'])
        state=torch.load(path/'checkpoint_state_dict.pt',map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not state or not all(bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError('Invalid model state dict')
        model.load_state_dict(state,strict=True);model.eval()
        va, va_pred=_predict(model,task,remain)
        tst,t_pred=_predict(model,task,te)
        fit_mask,sel_mask=_validation_partition(va,seed)
        fit=[s for s,ok in zip(va,fit_mask) if ok]
        sel=[s for s,ok in zip(va,sel_mask) if ok]
        role_groups['calibration']={str(s.split_group_id) for s in fit}
        role_groups['threshold']={str(s.split_group_id) for s in sel}
        fit_rows=_by_key(path/'validation_calibration_fit.csv')
        sel_rows=_by_key(path/'validation_threshold_selection.csv')
        test_rows=_by_key(path/'test.csv')
        e1=_verify_rows(fit_rows,fit,va_pred[fit_mask],task+' calibration')
        e2=_verify_rows(sel_rows,sel,va_pred[sel_mask],task+' selection')
        e3=_verify_rows(test_rows,tst,t_pred,task+' test')
        max_prediction_error=max(max_prediction_error,e1,e2,e3)
        calibration=json.loads((path/'calibration.json').read_text())
        coef=float(calibration['coefficient']);intercept=float(calibration['intercept'])
        y=np.asarray([int(s.frag_at_horizon) for s in fit])
        if set(y.tolist())!={0,1}:raise ValueError('Single class calibration labels')
        fitted=LogisticRegression(C=1.,max_iter=1000,random_state=seed%(2**32-1)).fit(va_pred[fit_mask,None],y)
        coef_error=abs(float(fitted.coef_[0,0])-coef)
        intercept_error=abs(float(fitted.intercept_[0])-intercept)
        max_coef_error=max(max_coef_error,coef_error,intercept_error)
        if coef_error>2e-6 or intercept_error>2e-6:
            raise ValueError(f'Calibration coefficient mismatch: {task}')
        for label,snaps,preds,rows in [('validation',sel,va_pred[sel_mask],sel_rows),('test',tst,t_pred,test_rows)]:
            probs=expit(coef*preds+intercept)
            for s,p in zip(snaps,probs):
                key=(str(s.run_id),int(s.time_index))
                error=abs(float(rows[key]['risk_probability'])-float(p))
                max_probability_error=max(max_probability_error,error)
                if error>1e-8:raise ValueError(f'{task} {label} probability mismatch: {error}')
        th=select_probability_threshold(expit(coef*va_pred[sel_mask]+intercept),
                                        np.asarray([int(s.frag_at_horizon) for s in sel]))
        if th!=manifest['threshold_selection']:
            raise ValueError(f'Validation-only alarm threshold replay mismatch: {task}')
        group_sets=[set(role_groups[k]) for k in ('train','early','calibration','threshold','test')]
        if any(not g for g in group_sets) or any(a&b for i,a in enumerate(group_sets) for b in group_sets[i+1:]):
            raise ValueError('Validation split group leakage')
        if set(manifest['early_stopping_groups'])!=role_groups['early']:
            raise ValueError('Early stopping role mismatch')
        summary.append({'task':task,'checkpoint_sha256':hashlib.sha256((path/'checkpoint_state_dict.pt').read_bytes()).hexdigest(),
                        'validation_calibration_fit_rows':len(fit),'validation_threshold_rows':len(sel),
                        'test_rows':len(tst),'selected_threshold':th['selected_threshold'],
                        'max_beta_prediction_abs_error':max(e1,e2,e3),'max_calibration_coef_abs_error':max(coef_error,intercept_error)})
    outcome={'schema':'fanet_r7_all_nine_neural_independent_replay_v1','status':'PASS',
             'scientific_claim':'ENGINEERING_ONLY_ONE_SEED_ONE_EPOCH',
             'seed':seed,'neural_checkpoint_count':len(summary),
             'total_test_predictions':sum(r['test_rows'] for r in summary),
             'shared_test_time_support':len({(s.run_id,s.time_index)for s in tst}),
             'max_beta_prediction_abs_error':max_prediction_error,
             'max_probability_abs_error':max_probability_error,
             'max_calibration_coef_abs_error':max_coef_error,
             'models':summary,'role_groups':{k:sorted(v) for k,v in role_groups.items()},
             'full_20_seed_experiment':'NOT_EXECUTED', 'packet_simpy':'NOT_EXECUTED',
             'submission':'HOLD'}
    if output:
        output=Path(output)
        if output.exists():raise FileExistsError('Would overwrite an audit')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(outcome,indent=2,ensure_ascii=False)+'\n')
    return outcome


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--config',type=Path,default=ROOT/'configs/r7_one_seed_fourteen_models.json')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report=audit(args.root,args.config,args.seed,output=args.output)
    print(json.dumps({key:report[key] for key in ['status','neural_checkpoint_count','total_test_predictions','max_beta_prediction_abs_error','max_probability_abs_error','max_calibration_coef_abs_error']}))

if __name__=='__main__':main()
