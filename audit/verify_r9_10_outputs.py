"""Independent standard-library event/count/aggregate verification of one R9.10 run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def require(test, msg):
    if not test:
        raise ValueError(msg)


def verify(root):
    r = json.loads((root/'R9_10_FIXED_THRESHOLD_AUDIT.json').read_bytes())
    require(r['evaluation_replicates'] == [2, 3, 4, 5, 6] and r['threshold'] == .15 and
            r['false_alert_budget_per_minute'] == 2.0 and r['blind_holdout_read'] is False and
            r['threshold_search'] is False and r['calibration_refit'] is False, 'Scope or protocol changed')
    require(set(r['output_sha256']) == {'evaluation_scores.csv', 'evaluation_full_truth.csv',
                                       'run_counts.csv', 'budget_summary.csv'}, 'Unexpected output set')
    for n, digest in r['output_sha256'].items():
        require(hashlib.sha256((root/n).read_bytes()).hexdigest() == digest, f'Output SHA mismatch {n}')
    counts = rows(root/'run_counts.csv')
    ids = {line['run_id']: line for line in counts}
    require(len(counts) == len(ids) == 180, 'Expected 180 run counts')
    full, decisions = defaultdict(dict), defaultdict(dict)
    for line in rows(root/'evaluation_full_truth.csv'):
        key, t = line['run_id'], int(line['time_index'])
        require(t not in full[key] and line['replicate'] == ids[key]['replicate'], 'Bad trajectory key')
        full[key][t] = float(line['beta_current'])
    for line in rows(root/'evaluation_scores.csv'):
        key, t = line['run_id'], int(line['time_index'])
        raw = float(line['predicted_beta'])
        p = float(line['p_fragment'])
        z = r['calibrator']['slope'] * raw + r['calibrator']['intercept']
        expected_p = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
        require(t not in decisions[key] and line['replicate'] == ids[key]['replicate'] and
                line['split_group_id'] == ids[key]['split_group_id'] and
                int(line['future_time_index']) == t+6 and float(line['beta_current']) == full[key][t] and
                float(line['beta_target']) == full[key][t+6] and
                int(line['frag_at_horizon']) == int(full[key][t+6] > 1) and
                0 <= p <= 1 and math.isclose(p, expected_p, abs_tol=1e-14, rel_tol=0), 'Bad decision data')
        decisions[key][t] = p
    require(full.keys() == decisions.keys() == ids.keys(), 'Unmatched run')
    for key in ids:
        ts, p = full[key], decisions[key]
        require(set(ts) == set(range(200)) and set(p) == set(range(4,194)), 'Bad temporal support')
        events = [t for t in range(1,200) if ts[t-1] <= 1 < ts[t]]
        eligible = {t for t in p if ts[t] <= 1}
        warnable = sum(any(t in eligible for t in range(event-6, event)) for event in events)
        active = {t for t in eligible if p[t] >= .15}
        emitted = []
        for t in sorted(active):
            if t-1 not in active and (not emitted or t-emitted[-1] >= 6):
                emitted.append(t)
        pool = set(events)
        tp = 0
        for t in emitted:
            match = min((e for e in pool if t < e <= t+6), default=None)
            if match is not None:
                pool.remove(match)
                tp += 1
        expected = (len(events), warnable, tp, len(emitted)-tp, len(ts), len(p))
        fields = ('events','warnable_events','true_alerts','false_alerts','observed_frames','decision_rows')
        require(tuple(int(ids[key][field]) for field in fields) == expected, f'Alert count mismatch {key}')
    aggregates = {(None,None), *((None,m) for m in ('rwp','gm','mission')),
                  *((b,m) for b in range(2,7) for m in (None,'rwp','gm','mission'))}
    observed = {(item['replicate'],item['mobility']) for item in r['evaluation_strata']}
    require(aggregates == observed and len(r['evaluation_strata']) == 24, 'Missing or repeated stratum')
    csv_aggregates = rows(root/'budget_summary.csv')
    require(len(csv_aggregates) == 24, 'Wrong summary CSV length')
    for item, exported in zip(r['evaluation_strata'], csv_aggregates):
        require(exported == {k: '' if v is None else str(v) for k,v in item.items()}, 'CSV/JSON differences')
        part = [row for row in counts if (item['replicate'] is None or int(row['replicate']) == item['replicate'])
                and (item['mobility'] is None or row['mobility'] == item['mobility'])]
        for field in ('events','warnable_events','true_alerts','false_alerts'):
            require(sum(int(row[field]) for row in part) == item[field], f'Stratum aggregation mismatch {field}')
        minutes = sum(int(row['observed_frames']) for row in part) / 600
        require(item['runs'] == len(part) and item['minutes'] == minutes and
                item['trajectory_groups'] == len({row['split_group_id'] for row in part}) and
                item['false_alerts_per_minute'] == item['false_alerts']/minutes and
                item['within_budget'] == (item['false_alerts'] <= 2*minutes), 'Exposure/budget mismatch')
    require(r['all_audited_budgets_met'] == all(item['within_budget'] for item in r['evaluation_strata']),
            'Overall budget mismatch')
    return {'verified_runs': len(ids), 'verified_strata': len(aggregates), 'output_files_sha256_verified': 4}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report_directory', type=Path)
    print(json.dumps(verify(ap.parse_args().report_directory), indent=2))
