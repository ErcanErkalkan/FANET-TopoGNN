"""Post-hoc, development-only comparison to a four-frame fragmentation-history heuristic.

WARNING: The heuristic and this analysis were specified after inspecting R9.10
results. This is NOT a prospective/blind or superiority experiment. No model is
trained. The base R9.10 model, its calibration, scores and 0.15 threshold are
never modified. The auxiliary heuristic's 19 threshold candidates are chosen
on R9.9 threshold-selection block 1, then frozen for development blocks 2-6.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = tuple(round(0.05*i, 2) for i in range(1,20))
HISTORY = 4
HORIZON = 6
COOLDOWN = 6
STEPS = 200
BUDGET = 2.0
ORIGINAL_THRESHOLD = .15
FROZEN_DEVELOPMENT_SHA = 'f65af11c033ff62d40d219e825e5e6cd6a357331af9e90d073d1bfe0385a9894'
OUT = ROOT/'data_derived/wn_simple_baseline'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path,records):
    if not records:
        raise ValueError(f'No rows: {path}')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys(),lineterminator='\n')
        w.writeheader();w.writerows(records)


def heuristic(truth):
    """Use only four preceding observed component counts, not a future label."""
    return {t:sum(int(truth[t-k]>1) for k in range(1, HISTORY+1))/HISTORY
            for t in range(HISTORY,STEPS-HORIZON)}


def evaluate(truth, scores,threshold):
    if set(truth)!=set(range(STEPS)) or set(scores)!=set(range(HISTORY,STEPS-HORIZON)):
        raise ValueError('Mismatched full truth or decision support')
    if not all(0<=p<=1 and math.isfinite(p) for p in scores.values()):
        raise ValueError('Invalid risk')
    onsets=[t for t in range(1,STEPS) if truth[t-1]<=1 <truth[t]]
    emitted=[];last=-10**12;positive_before=False
    for t in sorted(scores):
        if truth[t]>1:
            positive_before=False
            continue
        positive=scores[t]>=threshold
        if positive and not positive_before and t-last>=COOLDOWN:
            emitted.append(t);last=t
        positive_before=positive
    unmatched=set(onsets);lead=[]
    for t in emitted:
        matches=[e for e in range(t+1,t+HORIZON+1) if e in unmatched]
        if matches:
            e=min(matches); unmatched.remove(e);lead.append(e-t)
    tp=len(lead);fp=len(emitted)-tp;ev=len(onsets)
    return dict(events=ev,true_alerts=tp,false_alerts=fp,alerts=len(emitted),
                observed_frames=STEPS,lead_steps=lead)


def summarize(results):
    ev=sum(r['events'] for r in results)
    tp=sum(r['true_alerts'] for r in results)
    fp=sum(r['false_alerts'] for r in results)
    minutes=sum(r['observed_frames'] for r in results)*.1/60
    f1=2*tp/(ev+tp+fp) if (ev+tp+fp) else None
    return dict(runs=len(results),events=ev,true_alerts=tp,false_alerts=fp,
                alerts=tp+fp,minutes=minutes,recall=tp/ev if ev else None,
                precision=tp/(tp+fp) if tp+fp else None,event_f1=f1,
                false_alerts_per_minute=fp/minutes,within_budget=fp/minutes<=BUDGET)


def build_selection_truth():
    path=ROOT/'frozen/plan/R9_5_TRAIN_VALIDATION_MATRIX.csv'
    if sha(path)!=FROZEN_DEVELOPMENT_SHA:
        raise ValueError('Source plan hash mismatch')
    plan=read_csv(path)
    selection=[r for r in plan if r['role']=='threshold_selection' and int(r['replicate'])==1]
    assert len(selection)==36 and len({r['split_group_id'] for r in selection})==9
    # Read old simulator source via pinned helper; do not access separate blind matrix.
    path=ROOT/'frozen/r9_6_cost_pilot.py'
    spec=importlib.util.spec_from_file_location('r96_baseline',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checks=mod.prepare(ROOT/'core', ROOT/'frozen/plan')[1]
    ds=mod.simulator(ROOT/'core')
    sim=json.loads((ROOT/'core/configs/r7_full_20seed_isolated.json').read_text())['sim']
    radios=dict(ds._radio_scenarios(sim))
    kinematic={}; truth={}; betas={}
    for row in selection:
        k=(row['mobility'],int(row['n_nodes']),int(row['run_seed']))
        if k not in kinematic:kinematic[k]=ds._kinematic_run(sim,*k)
        cfg=copy.deepcopy(radios[row['radio_scenario']]);cfg['graph_policy']=row['graph_policy']
        snaps=ds.simulate_run(cfg,*k,kinematic[k],include_unlabeled_tail=True)
        mod.validate_run(snaps,row)
        truth[row['run_id']]={s.time_index:int(s.beta_current) for s in snaps}
        betas[row['run_id']]={s.time_index:float(s.beta_target) for s in snaps if s.target_available}
    anchor=ROOT/'frozen/anchor/Linux/fanet_topognn_development_beta.csv'
    line=read_csv(anchor)
    if len(line)!=36*190:raise ValueError('Selection frozen score size differs')
    anchor_by_run=defaultdict(dict)
    for r in line:
        run=r['run_id'];t=int(r['time_index']);future=int(r['future_time_index'])
        if run not in truth or t not in range(HISTORY,STEPS-HORIZON):
            raise ValueError('Extra anchor selection row')
        if future!=t+HORIZON or not math.isclose(float(r['beta_target']),truth[run][t+HORIZON],abs_tol=0,rel_tol=0):
            raise ValueError('Selection target does not agree with independent simulator')
        if int(r['frag_at_horizon']) != int(truth[run][t+HORIZON]>1):
            raise ValueError('Selection horizon label mismatch')
        if t in anchor_by_run[run]:raise ValueError('Duplicate anchor row')
        anchor_by_run[run][t]=float(r['predicted_beta'])
    if set(anchor_by_run)!=set(truth) or any(len(s)!=190 for s in anchor_by_run.values()):
        raise ValueError('Missing anchor selection rows')
    return selection,truth,anchor_by_run,checks


def get_frozen_evaluation():
    folder=next((ROOT/'audit/reference_linux').iterdir())
    # This is a complete, independently checked development transfer, NOT blind holdout.
    import sys
    sys.path.insert(0,str(ROOT/'audit'))
    from verify_r9_10_strict import verify
    verify(folder)
    plan_by_run={r['run_id']:r for r in read_csv(ROOT/'frozen/plan/R9_5_TRAIN_VALIDATION_MATRIX.csv')
                 if r['role']=='threshold_selection' and int(r['replicate']) in (2,3,4,5,6)}
    truth=defaultdict(dict);scores=defaultdict(dict)
    for line in read_csv(folder/'evaluation_full_truth.csv'):
        truth[line['run_id']][int(line['time_index'])]=int(float(line['beta_current']))
    for line in read_csv(folder/'evaluation_scores.csv'):
        scores[line['run_id']][int(line['time_index'])]=float(line['p_fragment'])
    assert set(truth)==set(scores)==set(plan_by_run) and len(truth)==180
    return plan_by_run,truth,scores,folder


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    select, sel_truth, sel_beta, plan_sha=build_selection_truth()
    anchor=json.loads((ROOT/'frozen/anchor/Linux/R9_9_CALIBRATION_ALERT_PILOT.json').read_text())
    cal=anchor['models']['fanet_topognn']['calibrator'];slope=cal['slope'];intercept=cal['intercept']
    top_sel=[];sel_baselines={x:[] for x in GRID}
    for row in select:
        run=row['run_id'];truth=sel_truth[run]
        prob={t:1/(1+math.exp(-(slope*b+intercept))) for t,b in sel_beta[run].items()}
        top_sel.append(evaluate(truth,prob,ORIGINAL_THRESHOLD))
        simple=heuristic(truth)
        for x in GRID:
            sel_baselines[x].append(evaluate(truth,simple,x))
    topo_sel=summarize(top_sel)
    # Independent anchor: verify the frozen TopoGNN selection summary.
    anchor_perf=anchor['models']['fanet_topognn']['selection_diagnostic_only']
    for src,k in [('events','Ground_Truth_Fragmentation_Events'),('true_alerts','True_Alert_Events'),
                  ('false_alerts','False_Alert_Events')]:
        if topo_sel[src]!=anchor_perf[k]:
            raise ValueError(f'R9.9 anchor selection mismatch {src}: {topo_sel[src]} versus {anchor_perf[k]}')
    grid=[]
    for threshold in GRID:
        s=summarize(sel_baselines[threshold]);grid.append({'threshold':threshold,**s})
    feasible=[x for x in grid if x['within_budget']]
    if not feasible:
        raise ValueError('No threshold among 19 candidates fits selection budget: baseline undefined')
    chosen=min(feasible,key=lambda x:(-x['event_f1'],x['false_alerts_per_minute'],
                                       -x['recall'],x['threshold']))
    selected=chosen['threshold']
    write_csv(OUT/'threshold_selection_grid.csv',grid)
    plan_eval, truth_eval, topo_eval,ref=get_frozen_evaluation()
    rows=[]
    for run in sorted(plan_eval):
        original=evaluate(truth_eval[run],topo_eval[run],ORIGINAL_THRESHOLD)
        baseline=evaluate(truth_eval[run],heuristic(truth_eval[run]),selected)
        for name,record,threshold in [('TopoGNN',original,ORIGINAL_THRESHOLD),('History-4',baseline,selected)]:
            rows.append({'model':name,'threshold':threshold,'run_id':run,
                        'split_group_id':plan_eval[run]['split_group_id'],
                        'block':plan_eval[run]['replicate'],'mobility':plan_eval[run]['mobility'],
                        'n_nodes':plan_eval[run]['n_nodes'],'graph_policy':plan_eval[run]['graph_policy'],
                        'radio_scenario':plan_eval[run]['radio_scenario'],
                        **{k:v for k,v in record.items() if k!='lead_steps'}})
    original_counts={r['run_id']:r for r in read_csv(ref/'run_counts.csv')}
    for row in rows:
        if row['model']=='TopoGNN':
            check=original_counts[row['run_id']]
            for key in ('events','true_alerts','false_alerts'):
                if int(row[key])!=int(check[key]):
                    raise ValueError(f'Original score replay differs {row["run_id"]}/{key}')
    write_csv(OUT/'paired_evaluation_by_run.csv',rows)
    summary=[]
    for model in ('TopoGNN','History-4'):
        part=[r for r in rows if r['model']==model]
        for scope,field in [('pooled',None),('mobility','mobility'),('node_count','n_nodes')]:
            cats=[('',part)] if field is None else [(k,[x for x in part if x[field]==k])
                 for k in sorted({x[field] for x in part})]
            for cat,p in cats:
                summary.append({'model':model,'scope':scope,'stratum':cat,**summarize(p)})
    write_csv(OUT/'paired_summary.csv',summary)
    # Paired resampling at the independent trajectory-group level, stratified by mobility x N.
    group_by_model={}
    for model in ('TopoGNN','History-4'):
        gr=defaultdict(list)
        for x in rows:
            if x['model']==model:gr[x['split_group_id']].append(x)
        group_by_model[model]={k:summarize(v) for k,v in gr.items()}
    strata=defaultdict(list)
    for gid,items in group_by_model['TopoGNN'].items():
        pp=next(r for r in rows if r['split_group_id']==gid)
        strata[(pp['mobility'],pp['n_nodes'])].append(gid)
    assert len(strata)==9 and all(len(v)==5 for v in strata.values())
    np_rng=np.random.default_rng(20261121)
    boots=[]
    for _ in range(10000):
        sample=[gid for key in sorted(strata) for gid in np_rng.choice(strata[key],size=5,replace=True)]
        vector={}
        for model in ('TopoGNN','History-4'):
            ee=sum(group_by_model[model][gid]['events'] for gid in sample)
            tp=sum(group_by_model[model][gid]['true_alerts'] for gid in sample)
            fp=sum(group_by_model[model][gid]['false_alerts'] for gid in sample)
            mins=sum(group_by_model[model][gid]['minutes'] for gid in sample)
            vector[model]={'f1':2*tp/(ee+tp+fp),'fp_rate':fp/mins,'recall':tp/ee}
        boots.append((vector['TopoGNN']['f1']-vector['History-4']['f1'],
                      vector['TopoGNN']['fp_rate']-vector['History-4']['fp_rate']))
    bootstrap={'resamples':10000,'resampling_unit':'45 paired trajectory groups, 5 per mobility x node count cell',
               'seed':20261121,'event_f1_topognn_minus_baseline_95pct':
               np.quantile(np.asarray(boots)[:,0],[.025,.975]).tolist(),
               'fp_per_min_topognn_minus_baseline_95pct':
               np.quantile(np.asarray(boots)[:,1],[.025,.975]).tolist(),
               'status':'DESCRIPTIVE_POST_HOC_NOT_CONFIRMATORY'}
    receipt={'status':'POST_HOC_DEVELOPMENT_ONLY_COMPARISON_PUBLICATION_HOLD',
             'comparison':'TopoGNN 0.15 versus History-4 rule, 19 grid points on block 1 only',
             'baseline_formula':'score(t) = (sum_{k=1..4} 1[beta_current(t-k)>1])/4; current connectivity is used by shared alert gate',
             'prospectively_preregistered':False,'blind_holdout_read':False,'superiority_test':False,
             'baseline_selected_threshold':selected,'baseline_selection':chosen,
             'frozen_topognn_selection':topo_sel,
             'evaluation':{m:summarize([x for x in rows if x['model']==m]) for m in ('TopoGNN','History-4')},
             'selection_provenance':{'frozen_plan_sha256':FROZEN_DEVELOPMENT_SHA,
                                      'selection_anchor_sha256':sha(ROOT/'frozen/anchor/Linux/fanet_topognn_development_beta.csv'),
                                      'topognn_calibrator_sha256':sha(ROOT/'frozen/anchor/Linux/R9_9_CALIBRATION_ALERT_PILOT.json'),
                                      'evaluation_reference_sha256':sha(ref/'evaluation_scores.csv'),
                                      'simulator_source_sha256':plan_sha},
             'bootstrap':bootstrap,
             'constraints':['One fixed heuristic, chosen for simplicity after viewing original R9.10 outcomes.',
                            'Threshold chosen only on development selection block 1; never tuned on blocks 2-6.',
                            'Full truth is simulated for threshold selection and matches frozen target labels.',
                            'No retraining or change to TopoGNN; same 190 decision instants, event matching, horizon and cooldown.',
                            'Historical R9.10 evaluation blocks are NOT blind for this newly constructed heuristic.',
                            'Simple history-only reference cannot establish general model superiority or practical usefulness.']}
    (OUT/'COMPARISON_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'baseline_selected_threshold':selected,'selection':{'TopoGNN':topo_sel,'History-4':chosen},
                      'evaluation':receipt['evaluation'],'bootstrap':bootstrap},indent=2))

if __name__=='__main__':main()
