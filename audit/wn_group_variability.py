"""Audit R9.10 blocks and trajectory-cluster resampling (descriptive only).

Experimental evidence is frozen. A trajectory seed/group, not its four paired
radio/graph runs or frames, is the smallest resampling unit. The nine balanced
mobility-by-size strata each contain five groups (one per development block).
This is not a confirmatory confidence interval or an independent test set.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from verify_r9_10_strict import ROOT, verify

BUDGET = 2.0
BLOCKS = tuple(range(2, 7))
MOBILITIES = ("rwp", "gm", "mission")
NODES = (10, 20, 30)
SEED = 20261121
RESAMPLES = 12000
NUMERIC = ('events', 'true_alerts', 'false_alerts', 'observed_frames')


def load_groups(p: Path):
    groups = defaultdict(list)
    with (p / 'run_counts.csv').open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 180 or len({r['run_id'] for r in rows}) != 180:
        raise ValueError('Expected exactly 180 distinct runs')
    for r in rows:
        key = (int(r['replicate']), r['mobility'], int(r['n_nodes']), r['split_group_id'])
        groups[key].append(r)
    if len(groups) != 45:
        raise ValueError('Expected exactly 45 trajectory clusters')
    out = []
    for key, variants in sorted(groups.items()):
        b, mob, n, g = key
        if b not in BLOCKS or mob not in MOBILITIES or n not in NODES:
            raise ValueError(f'Unexpected stratum: {key}')
        if len(variants) != 4 or {(r['radio_scenario'], r['graph_policy']) for r in variants} != {
           ('nominal','fixed'), ('nominal','adaptive'), ('degraded','fixed'), ('degraded','adaptive') }:
            raise ValueError(f'Incomplete matched quartet for {key}')
        vals = {name:sum(int(r[name]) for r in variants) for name in NUMERIC}
        if vals['observed_frames'] != 800:
            raise ValueError(f'Unexpected trajectory exposure: {key}')
        out.append(dict(block=b,mobility=mob,n_nodes=n,split_group_id=g,minutes=800/600,**vals))
    if {(r['block'],r['mobility'],r['n_nodes']) for r in out} != set(itertools.product(BLOCKS,MOBILITIES,NODES)):
        raise ValueError('Incomplete block-by-mobility-by-size lattice')
    return out


def totals(rows):
    ev = sum(r['events'] for r in rows)
    tp = sum(r['true_alerts'] for r in rows)
    fp = sum(r['false_alerts'] for r in rows)
    minute = sum(r['observed_frames'] for r in rows) / 600
    return dict(trajectory_groups=len(rows), runs=4*len(rows),events=ev, true_alerts=tp,
         false_alerts=fp, minutes=minute, event_recall=tp/ev if ev else None,
         false_alerts_per_minute=fp/minute, within_budget=fp/minute<=BUDGET)


def percentile(vals, prob):
    vals = sorted(vals)
    pos=(len(vals)-1)*prob
    lower=int(pos); frac=pos-lower
    return vals[lower]*(1-frac)+vals[min(lower+1,len(vals)-1)]*frac


def bootstrap(groups, *, seed=SEED, resamples=RESAMPLES):
    """Balanced conditional cluster resampling: five groups per each of 9 cells."""
    if resamples < 100:
        raise ValueError('Require >=100 resamples')
    strata = {(m,n):[g for g in groups if g['mobility']==m and g['n_nodes']==n]
              for m in MOBILITIES for n in NODES}
    if any(len(g)!=5 for g in strata.values()):
        raise ValueError('Expected 5 trajectory groups in every sampling stratum')
    rng=random.Random(seed)
    samples={m:[] for m in ('all',)+MOBILITIES}
    for _ in range(resamples):
        sample=[]
        for m in MOBILITIES:
            for n in NODES:
                cell=strata[m,n]
                sample.extend(cell[rng.randrange(5)] for _ in range(5))
        for m in samples:
            rows=sample if m=='all' else [r for r in sample if r['mobility']==m]
            x=totals(rows)
            samples[m].append((x['false_alerts_per_minute'],x['event_recall']))
    rows=[]
    for m in ('all',)+MOBILITIES:
        observed=totals(groups if m=='all' else [x for x in groups if x['mobility']==m])
        rr=[v[0] for v in samples[m]]
        recall=[v[1] for v in samples[m] if v[1] is not None]
        rows.append(dict(mobility=m,trajectory_groups=observed['trajectory_groups'],
            runs=observed['runs'], false_alarms=observed['false_alerts'],
            minutes=observed['minutes'],false_alarm_rate=observed['false_alerts_per_minute'],
            rate_percentile_2_5=percentile(rr,.025),rate_percentile_97_5=percentile(rr,.975),
            event_recall=observed['event_recall'],recall_percentile_2_5=percentile(recall,.025),
            recall_percentile_97_5=percentile(recall,.975),resamples=resamples,random_seed=seed))
    return rows


def analyze(groups):
    blocks=[dict(block=b,**totals([g for g in groups if g['block']==b])) for b in BLOCKS]
    dropped=[dict(removed_block=b,**totals([g for g in groups if g['block']!=b])) for b in BLOCKS]
    dropped_gm=[dict(removed_block=b,**totals([g for g in groups if g['block']!=b and g['mobility']=='gm'])) for b in BLOCKS]
    return blocks,dropped,dropped_gm


def export(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def refresh(tex: Path, identifier: str, rows):
    old=tex.read_text(encoding='utf8')
    a='% BEGIN VERIFIED '+identifier
    b='% END VERIFIED '+identifier
    if old.count(a)!=1 or old.count(b)!=1:
        raise ValueError('Missing table markers: '+identifier)
    start=old.index(a)+len(a);end=old.index(b)
    tex.write_text(old[:start]+'\n'+''.join(rows)+old[end:],encoding='utf8')


def latex_rows_blocks(blocks):
    for b in blocks:
        v=f"{b['false_alerts_per_minute']:.2f}"
        if not b['within_budget']:v=r'\textbf{'+v+'}'
        yield f"{b['block']} & {b['events']} & {b['true_alerts']} & {b['false_alerts']} & 12 & {b['event_recall']*100:.1f} & {v} \\\\\n"


def latex_rows_boot(boot):
    names={'all':'All runs','rwp':'Random waypoint','gm':'Gauss--Markov','mission':'Mission-based'}
    for r in boot:
        yield (f"{names[r['mobility']]} & {r['trajectory_groups']} & {r['false_alarm_rate']:.2f} & "
               f"[{r['rate_percentile_2_5']:.2f}, {r['rate_percentile_97_5']:.2f}] & "
               f"{r['event_recall']*100:.1f} & "
               f"[{r['recall_percentile_2_5']*100:.1f}, {r['recall_percentile_97_5']*100:.1f}] \\\\\n")


def latex_rows_dropped(overall, gm):
    for allrow, gmrow in zip(overall,gm):
        if allrow['removed_block']!=gmrow['removed_block']:
            raise ValueError('Block exclusion mismatch')
        yield (f"{allrow['removed_block']} & {allrow['trajectory_groups']} & "
               f"{allrow['minutes']:.0f} & {allrow['false_alerts_per_minute']:.2f} & "
               f"{gmrow['false_alerts_per_minute']:.2f} " + chr(92)*2 + "\n")

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--sync-tex',action='store_true')
    a.add_argument('--resamples',type=int,default=RESAMPLES)
    args=a.parse_args()
    refs=[next((ROOT/'audit'/f'reference_{platform}').iterdir()) for platform in ('linux','windows')]
    refs.append(ROOT/'evidence/pilot_linux')
    checked=[verify(p) for p in refs]
    groups=[load_groups(p) for p in refs]
    if groups[0]!=groups[1] or groups[0]!=groups[2]:
        raise ValueError('Trajectory group counts differ between reference platforms or pilot')
    total=totals(groups[0])
    if (total['events'],total['true_alerts'],total['false_alerts'],total['minutes'])!=(1114,183,88,60):
        raise ValueError('Unexpected pooled event evidence')
    if any(total[k]!=checked[0]['strata'][0][k] for k in ('events','true_alerts','false_alerts','minutes')):
        raise ValueError('Verifier disagrees with trajectory-group totals')
    blocks,dropped,dropped_gm=analyze(groups[0]);boot=bootstrap(groups[0],resamples=args.resamples)
    assert sum(b['false_alerts'] for b in blocks)==88
    d=ROOT/'data_derived'
    export(blocks,d/'wn_r9_10_block_variability.csv')
    export(dropped,d/'wn_r9_10_leave_one_block_out.csv')
    export(dropped_gm,d/'wn_r9_10_leave_one_block_out_gm.csv')
    export(boot,d/'wn_r9_10_cluster_bootstrap.csv')
    if args.sync_tex:
        refresh(ROOT/'paper/manuscript/main.tex','BLOCK VARIABILITY',latex_rows_blocks(blocks))
        refresh(ROOT/'paper/manuscript/ESM_1.tex','CLUSTER BOOTSTRAP',latex_rows_boot(boot))
        refresh(ROOT/'paper/manuscript/ESM_1.tex','BLOCK LEAVE ONE OUT',latex_rows_dropped(dropped,dropped_gm))
    print(json.dumps(dict(verified_sources=len(refs),independent_trajectory_group_units=45,
        paired_variants_per_group=4,blocks=blocks,bootstrap=boot,
        group_resamples=args.resamples,seed=SEED,exploratory_only=True),indent=2))

if __name__=='__main__': main()
