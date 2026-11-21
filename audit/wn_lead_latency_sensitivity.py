"""Frozen-alarm lead-time vs hypothetical effective-action latency; descriptive only.

No action/recovery is simulated. The upper-bound timely-hit counts assume that a
post-alert action becomes effective in <= L seconds, with no sensing/communications
extra delay unless already included in L. For each matched alert, a potential
on-time action exists if onset_time-alert_time >= L.

Both procedures are post-hoc sensitivity summaries of development data, not
controller experiments or new validation cohorts. No thresholds are retuned.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit'))
from verify_r9_10_strict import verify,rows,event_matches,THRESHOLD,HORIZON,COOLDOWN
import wn_simple_baseline_comparison as h4

OUT=ROOT/'data_derived'
TICKS=tuple(range(0,7))  # 0.0..0.6 seconds; exact tick comparisons avoid roundoff.


def match_pairs(truth,score,cutoff):
    above={t for t,p in score.items() if truth[t]==1 and p>=cutoff}
    emitted=[t for t in sorted(above) if t-1 not in above]
    effective=[];last=-10**9
    for t in emitted:
        if t-last>=COOLDOWN:
            effective.append(t);last=t
    events={t for t in range(1,200) if truth[t-1]==1 and truth[t]>1}
    unmatched=set(events);pairs=[]
    for alarm in effective:
        options=unmatched.intersection(range(alarm+1,alarm+HORIZON+1))
        if options:
            e=min(options); unmatched.remove(e)
            pairs.append((alarm,e))
    return pairs,len(effective),len(events)


def read_runs(folder):
    ref=verify(folder)
    truth=defaultdict(dict);scores=defaultdict(dict)
    for row in rows(folder/'evaluation_full_truth.csv'):
        truth[row['run_id']][int(row['time_index'])]=int(float(row['beta_current']))
    for row in rows(folder/'evaluation_scores.csv'):
        scores[row['run_id']][int(row['time_index'])]=float(row['p_fragment'])
    info={r['run_id']:r for r in rows(folder/'run_counts.csv')}
    totals={kind:Counter() for kind in ('TopoGNN','History-4')}
    distributions={kind:Counter() for kind in totals}
    by_mobility={kind:defaultdict(Counter) for kind in totals}
    for run in sorted(truth):
        t,p=truth[run],scores[run]
        for kind in totals:
            q=p if kind=='TopoGNN' else h4.heuristic(t)
            cutoff=THRESHOLD if kind=='TopoGNN' else 0.30
            matched,alerts,events=match_pairs(t,q,cutoff)
            if kind=='TopoGNN':
                strict=event_matches(t,p)
                assert len(matched)==strict['true_alerts']
                assert alerts-len(matched)==strict['false_alerts']
                assert Counter(e-a for a,e in matched)==Counter(strict['lead_steps'])
            else:
                check=h4.evaluate(t,q,0.30)
                assert check['true_alerts']==len(matched)
                assert check['false_alerts']==alerts-len(matched)
            obs=totals[kind];obs['events']+=events;obs['alerts']+=alerts;obs['matched']+=len(matched)
            m=by_mobility[kind][info[run]['mobility']]
            m['events']+=events;m['alerts']+=alerts;m['matched']+=len(matched)
            for alarm,onset in matched:
                lead=onset-alarm
                assert 1<=lead<=6
                distributions[kind][lead]+=1
                m[lead]+=1
    assert totals['TopoGNN']==Counter(events=1114,alerts=271,matched=183)
    assert totals['History-4']==Counter(events=1114,alerts=516,matched=381)
    assert sum(distributions['TopoGNN'].values())==183
    assert ref['median_lead_seconds']==0.2
    return totals,distributions,by_mobility


def result(folder):
    totals,distribution,mobility=read_runs(folder)
    tab=[];dists=[];regimes=[]
    for kind in ('TopoGNN','History-4'):
        n=totals[kind]['events'];hits=totals[kind]['matched']
        for lead in range(1,7):
            dists.append({'method':kind,'lead_ticks':lead,'lead_seconds':f'{lead/10:.1f}',
                          'matched_alerts':distribution[kind][lead]})
        for t in TICKS:
            available=sum(nhit for lead,nhit in distribution[kind].items() if lead>=t)
            tab.append({'method':kind,'assumed_effective_action_latency_seconds':f'{t/10:.1f}',
                        'matched_alerts':hits,'potentially_timely_matched_alerts':available,
                        'fraction_of_matched_pct':f'{100*available/hits:.1f}',
                        'fraction_of_all_events_pct':f'{100*available/n:.1f}',
                        'events_total':n,'false_alerts_unchanged':totals[kind]['alerts']-hits})
        for mob in ('rwp','gm','mission'):
            v=mobility[kind][mob]
            for t in (1,2,3):
                timely=sum(v[lead] for lead in range(t,7))
                regimes.append({'method':kind,'mobility':mob,'latency_seconds':f'{t/10:.1f}',
                                'events':v['events'],'matched':v['matched'],'potentially_timely':timely,
                                'all_event_pct':f'{100*timely/v["events"]:.1f}'})
    return tab,dists,regimes


def folders():
    return (next((ROOT/'audit/reference_linux').glob('fixed_threshold_*')),
            next((ROOT/'audit/reference_windows').glob('fixed_threshold_*')),
            ROOT/'evidence/pilot_linux')


def consistent():
    out=[result(p) for p in folders()]
    if any(item !=out[0] for item in out[1:]):
        raise AssertionError('Platform records do not agree on lead-time or latency sensitivity')
    return out[0]


def write(path,data):
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,data[0].keys(),lineterminator='\n');wr.writeheader();wr.writerows(data)


def main():
    tab,dist,mob=consistent()
    write(OUT/'wn_r9_10_action_latency_sensitivity.csv',tab)
    write(OUT/'wn_r9_10_lead_distribution.csv',dist)
    write(OUT/'wn_r9_10_action_latency_by_mobility.csv',mob)
    print(json.dumps({'records':3,'lead_distribution':dist,'sensitivity':tab,'mobility':mob},indent=2))

if __name__=='__main__':main()
