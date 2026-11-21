"""Development-only transfer audit of the pinned R9.9 TopoGNN operating point."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PHASE = 'R9_10'
BLOCKS = (2, 3, 4, 5, 6)
THRESHOLD = .15
BUDGET = 2.0
R99_SOURCE = 'a9cae766b82dd8e8c493205efdd167641ad9d49beef6454818938f15513cf250'
R99_RECEIPTS = {
    'Windows': 'c85864ec96cbb9ef4254ca35527096a9ba57e89e4eb4e8edc9c2d0f96389a857',
    'Linux': '6bc5b36c1221f493a532510137a0492abc2894408e0c63c5474dadb70e6d68ef',
}
R99_NAME = 'R9_9_CALIBRATION_ALERT_PILOT.json'
BETA_FILE = 'fanet_topognn_development_beta.csv'
PROTOCOL_SHA = '31452a9772903b6686d3a88d11778d07157204f8619715c201241544f0b80fa4'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preflight(a):
    protocol = Path(__file__).with_name('R9_10_PROTOCOL.json')
    if sha(protocol) != PROTOCOL_SHA or sha(a.r99_code) != R99_SOURCE:
        raise ValueError('Protocol or R9.9 source mismatch')
    spec = spec_from_file_location('r99_frozen', a.r99_code)
    r99 = module_from_spec(spec)
    sys.modules[spec.name] = r99
    spec.loader.exec_module(r99)
    shared = r99.load_verified(a.r98_code, r99.R98_SHA, 'r98_frozen')
    old = r99.load_verified(a.r97_code, shared.R97_SHA, 'r97_frozen')
    cost, _, provenance = old.input_check(a)
    if provenance['manifest_sha256'] != r99.MANIFEST_SHA:
        raise ValueError('Development manifest changed')
    ref = a.reference / R99_NAME
    if sha(ref) != R99_RECEIPTS.get(platform.system()):
        raise ValueError('Incorrect pinned same-platform R9.9 report')
    report = json.loads(ref.read_bytes())
    model = report['models']['fanet_topognn']
    if (report['status'] != r99.STATUS or report['runner_sha256'] != R99_SOURCE or
            report['blind_holdout_read'] is not False or report['external_freeze_complete'] is not False or
            report['model_seed'] != old.SEED or report['topognn_epochs'] != old.EPOCHS or
            model['selection_diagnostic_only']['threshold'] != THRESHOLD or
            report['false_alert_budget_per_full_trajectory_minute'] != BUDGET or
            sha(a.reference / BETA_FILE) != report['output_sha256'][BETA_FILE]):
        raise ValueError('R9.9 calibration or anchor reference mismatch')
    with (a.plan / cost.PLAN_CSV).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    used = {name: [r for r in rows if r['role'] == name and int(r['replicate']) == 1]
            for name in (*old.ROLES, r99.CAL)}
    r99.group_audit(used, cost, (*old.ROLES, r99.CAL))
    blocks = {b: [r for r in rows if r['role'] == r99.SEL and int(r['replicate']) == b] for b in BLOCKS}
    group_overlap_check(used, blocks, cost, r99)
    return r99, shared, old, cost, used, blocks, report, provenance


def group_overlap_check(used, blocks, cost, r99):
    cells = {(m, n, radio, graph) for m in cost.MOBILITIES for n in cost.NODES
             for radio in cost.RADIOS for graph in cost.GRAPHS}
    fields = ('split_group_id', 'run_seed', 'run_id')
    seen = {f: {r[f] for rows in used.values() for r in rows} for f in fields}
    for b, part in blocks.items():
        group_count = Counter(r['split_group_id'] for r in part)
        actual = {(r['mobility'], int(r['n_nodes']), r['radio_scenario'], r['graph_policy']) for r in part}
        pair = {(r['split_group_id'], r['run_seed'], r['mobility'], r['n_nodes']) for r in part}
        sets = {f: {r[f] for r in part} for f in fields}
        if (len(part) != 36 or actual != cells or len(group_count) != 9 or set(group_count.values()) != {4}
                or len(pair) != 9 or len(sets['run_seed']) != 9 or len(sets['run_id']) != 36
                or any(r['role'] != r99.SEL or int(r['replicate']) != b or
                       int(r['simulator_steps']) != r99.LENGTH or int(r['horizon_steps']) != r99.H or
                       float(r['dt']) != r99.DT for r in part)
                or any(seen[f] & sets[f] for f in fields)):
            raise ValueError(f'Invalid or overlapping development block {b}')
        for f in fields:
            seen[f].update(sets[f])


def result_row(scope, replicate, mobility, rows):
    tp = sum(r['true_alerts'] for r in rows)
    fp = sum(r['false_alerts'] for r in rows)
    ev = sum(r['events'] for r in rows)
    frames = sum(r['observed_frames'] for r in rows)
    minutes = frames * .1 / 60
    return {
        'scope': scope, 'replicate': replicate, 'mobility': mobility, 'runs': len(rows),
        'trajectory_groups': len({r['split_group_id'] for r in rows}), 'minutes': minutes,
        'events': ev, 'warnable_events': sum(r['warnable_events'] for r in rows),
        'true_alerts': tp, 'false_alerts': fp, 'false_alerts_per_minute': fp / minutes,
        'within_budget': fp <= BUDGET * minutes, 'event_recall': tp / ev if ev else None,
        'event_precision': tp / (tp + fp) if tp + fp else None,
        'event_f1': 2 * tp / (tp + fp + ev) if tp + fp + ev else None,
    }


def score_runs(r99, old, evaluator, result, snapshots, truth, part, beta_keys, beta, prob, score_writer, truth_writer):
    rmap = {r['run_id']: r for r in part}
    indices, labels = {}, {}
    for i, s in enumerate(snapshots):
        indices.setdefault(s.run_id, []).append(i)
    for s in truth:
        labels.setdefault(s.run_id, []).append(s)
    if set(indices) != set(rmap) or set(labels) != set(rmap):
        raise ValueError('Incomplete per-run score/truth support')
    records = []
    for run_id, ix in indices.items():
        r = rmap[run_id]
        snaps = [snapshots[i] for i in ix]
        complete = labels[run_id]
        if len(snaps) != 190 or len(complete) != 200 or any(s.mobility != r['mobility'] for s in snaps):
            raise ValueError('Run support or mobility changed')
        scores = prob[ix]
        metrics = evaluator(snaps, scores, complete, THRESHOLD, r99.DT, r99.H, r99.H)
        events, warnable = r99.warnable_events(complete, snaps)
        if events != metrics['Ground_Truth_Fragmentation_Events']:
            raise ValueError('Event count mismatch')
        records.append({**{k: r[k] for k in ('replicate', 'run_id', 'split_group_id', 'mobility',
                                              'n_nodes', 'graph_policy', 'radio_scenario')},
                        'observed_frames': len(complete), 'decision_rows': len(snaps), 'events': events,
                        'warnable_events': warnable, 'true_alerts': int(metrics['True_Alert_Events']),
                        'false_alerts': int(metrics['False_Alert_Events'])})
        score_writer.writerows((r['replicate'], *beta_keys[i], float(snapshots[i].beta_current),
                                float(beta[i]), float(prob[i])) for i in ix)
        truth_writer.writerows((r['replicate'], s.run_id, s.time_index, s.beta_current) for s in complete)
    return records


def run(a):
    r99, shared, old, cost, used, blocks, ref, provenance = preflight(a)
    meta = {**provenance, 'protocol_sha256': PROTOCOL_SHA,
            'r99_report_sha256': sha(a.reference / R99_NAME), 'r99_runner_sha256': R99_SOURCE,
            'evaluation_replicates': BLOCKS, 'replayed_evaluation_runs': sum(map(len, blocks.values())),
            'threshold': THRESHOLD, 'false_alert_budget_per_minute': BUDGET,
            'calibrator': ref['models']['fanet_topognn']['calibrator'],
            'scientific_submission': 'HOLD', 'blind_holdout_read': False, 'external_freeze_complete': False}
    if a.check:
        return {'status': 'R9_10_CHECK_PASS_NO_SIMULATION', **meta}
    for p in (a.workspace, a.plan, a.r96_code.parent, a.r96_windows_report.parent, a.r97_code.parent,
              a.r98_code.parent, a.reference, a.r99_code.parent):
        if p == a.output or p in a.output.parents or a.output in p.parents:
            raise ValueError('Output directory overlaps an earlier source or report')
    start = time.perf_counter()
    import numpy as np
    import scipy
    import sklearn
    import torch
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    ds = cost.simulator(a.workspace)
    from fanet import evaluation, training
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': ds.pd.__version__, 'scikit_learn': sklearn.__version__,
                'scipy': scipy.__version__, 'torch': str(torch.__version__)}
    if versions != ref['versions'] or not training.TORCH_AVAILABLE or torch.version.cuda is not None:
        raise RuntimeError(f'Use the pinned R9.9 CPU environment: {versions}')
    torch.set_num_threads(old.THREADS)
    torch.set_num_interop_threads(1)
    cfg = json.loads((a.workspace / 'configs/r7_full_20seed_isolated.json').read_bytes())
    out = a.output / datetime.now(timezone.utc).strftime('fixed_threshold_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('R9.10: reconstructing R9.9 fitted model and auditing its anchor...', file=sys.stderr, flush=True)
    fit_roles = {r: used[r] for r in old.ROLES}
    anchor_data, anchor_truth = r99.replay(shared, old, cost, ds, cfg['sim'], fit_roles, np)
    fingerprints = shared.fingerprint(anchor_data, np)
    if fingerprints != {r: ref['input_fingerprints'][r] for r in old.ROLES}:
        raise ValueError('Reconstructed model inputs differ from R9.9')
    train, select, anchor = (anchor_data[r] for r in old.ROLES)
    fit_cfg = {**cfg['training'], 'epochs': old.EPOCHS, 'patience': old.EPOCHS}
    cal = meta['calibrator']

    def probabilities(beta):
        if not np.isfinite(beta).all():
            raise ValueError('Nonfinite beta')
        return expit(cal['slope'] * beta + cal['intercept'])

    with threadpool_limits(limits=old.THREADS):
        fitted = training.train_torch_model('FANET-TopoGNN', train, select, fit_cfg, train[0].pi.size, old.SEED)
        if not isinstance(fitted.model, torch.nn.Module) or shared.fingerprint(anchor_data, np) != fingerprints:
            raise ValueError('Model or frozen snapshot verification failed')
        keys, beta = r99.beta_scores(old, evaluation, fitted, anchor, np)
        expected_keys, expected_beta = shared.scores(a.reference / BETA_FILE, old.SUPPORT)
        max_delta = float(np.max(np.abs(beta - expected_beta)))
        anchor_metrics = evaluation.alert_event_metrics_full_trajectory(
            anchor, probabilities(beta), anchor_truth, THRESHOLD, r99.DT, r99.H, r99.H)
        ref_metrics = ref['models']['fanet_topognn']['selection_diagnostic_only']
        if ([tuple(str(x) for x in k) for k in keys] != expected_keys or max_delta > shared.ATOL['fanet_topognn']
                or any(v != ref_metrics[k] for k, v in anchor_metrics.items())):
            raise ValueError('R9.9 anchor decision support, beta parity, or event metrics changed')
        del train, select, anchor, anchor_data, anchor_truth
        runs = []
        scores_path = out / 'evaluation_scores.csv'
        truth_path = out / 'evaluation_full_truth.csv'
        with scores_path.open('x', encoding='utf-8', newline='') as s, truth_path.open('x', encoding='utf-8', newline='') as t:
            sw, tw = (csv.writer(f, lineterminator='\n') for f in (s, t))
            sw.writerow(('replicate', *old.SUPPORT, 'beta_current', 'predicted_beta', 'p_fragment'))
            tw.writerow(('replicate', 'run_id', 'time_index', 'beta_current'))
            for b, part in blocks.items():
                print(f'R9.10: evaluating frozen threshold on development block {b}...', file=sys.stderr, flush=True)
                dscores, truth = r99.replay(shared, old, cost, ds, cfg['sim'], {r99.SEL: part}, np)
                scored = dscores[r99.SEL]
                keys, beta = r99.beta_scores(old, evaluation, fitted, scored, np)
                prob = probabilities(beta)
                runs.extend(score_runs(r99, old, evaluation.alert_event_metrics_full_trajectory,
                                       fitted, scored, truth, part, keys, beta, prob, sw, tw))
                del truth, dscores, scored
    summary = [result_row('pooled', None, None, runs)]
    summary.extend(result_row('pooled_mobility', None, m, [r for r in runs if r['mobility'] == m])
                   for m in cost.MOBILITIES)
    for b in BLOCKS:
        subset = [r for r in runs if int(r['replicate']) == b]
        summary.append(result_row('replicate', b, None, subset))
        summary.extend(result_row('replicate_mobility', b, m, [r for r in subset if r['mobility'] == m])
                       for m in cost.MOBILITIES)
    outputs = {p.name: sha(p) for p in (scores_path, truth_path)}
    for n, rr in (('run_counts.csv', runs), ('budget_summary.csv', summary)):
        fields = tuple(rr[0])
        outputs[n] = r99.write_csv(out / n, fields, ([r[k] for k in fields] for r in rr))
    peak, method = cost.memory()
    report = {'status': 'R9_10_DEVELOPMENT_TRANSFER_COMPLETE_PUBLICATION_HOLD', **meta,
              'runner_sha256': sha(Path(__file__)), 'versions': versions, 'platform': platform.platform(),
              'model_reconstructed_from_replicate_1': True, 'calibration_refit': False,
              'threshold_search': False, 'anchor_regression_max_abs_delta': max_delta,
              'anchor_event_metrics': anchor_metrics, 'evaluation_strata': summary,
              'all_audited_budgets_met': all(r['within_budget'] for r in summary),
              'output_sha256': outputs, 'elapsed_seconds': time.perf_counter() - start,
              'peak_memory_bytes': peak, 'memory_measurement': method,
              'limits': ['A single fitted operating point, evaluated on five development-only blocks.',
                         'No external test, generalization guarantee, interval, or approved deployment threshold.',
                         'R9.9 anchor is excluded. Four graph/radio variants are paired, not independent repeats.']}
    p = out / 'R9_10_FIXED_THRESHOLD_AUDIT.json'
    old.fs_json(p, report)
    return {**report, 'report_path': str(p)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ('workspace', 'plan', 'r96-code', 'r96-windows-report', 'r97-code', 'r98-code', 'r99-code', 'reference'):
        ap.add_argument('--' + n, type=lambda x: Path(x).resolve(), required=True)
    ap.add_argument('--output', type=lambda x: Path(x).resolve())
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if not a.check and a.output is None:
        ap.error('--output is required for the audit')
    for n in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[n] = '' if n == 'CUDA_VISIBLE_DEVICES' else '2'
    print(json.dumps(run(a), indent=2, ensure_ascii=True, allow_nan=False))
