"""Development-only replay-once benchmark. No external risk calibration or test read."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import MappingProxyType

R97_SHA = 'aa777da9bda63f2e16eb71c072fa74d3e7eb93ef54471378e578d1c7d844a31d'
BASELINE_REPORT = 'R9_7_TRAINING_WRITE_PILOT.json'
EXPECTED_REPORT_SHA = {
    'Windows': 'db5bd6b9940d09b1942f3bb32dcf180ce66409746cc8a85e095e1a024a79cc71',
    'Linux': 'ba9935db525b920aadc052e62614bdc240b28f5c11699a47843c78cb900a99a0',
}
REFERENCE_SUPPORT = '90ea6ebfa156bda5452fd22f7fec6dfbdf745184b147648240fb0de2b3f823cb'
ATOL = {'extra_trees': 1e-10, 'fanet_topognn': 1e-6}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def input_check(a):
    if sha(a.r97_code) != R97_SHA:
        raise ValueError('R9.7 code SHA-256 differs from the audited source')
    spec = importlib.util.spec_from_file_location('r97_source', a.r97_code)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cost, rows, provenance = old.input_check(a)
    report_path = a.reference / BASELINE_REPORT
    old_report = json.loads(report_path.read_text(encoding='utf-8'))
    baseline_platform = old_report['platform'].split('-', 1)[0]
    if baseline_platform != platform.system() or sha(report_path) != EXPECTED_REPORT_SHA.get(baseline_platform):
        raise ValueError('Use the pinned R9.7 reference report from this operating system')
    if (old_report['status'] != old.STATUS or old_report['runner_sha256'] != R97_SHA
            or old_report['support_sha256'] != REFERENCE_SUPPORT or old_report['blind_holdout_read'] is not False):
        raise ValueError('Wrong R9.7 evidence identity')
    for model in old.MODELS:
        item = old_report['models'][model]
        if item['csv_name'] != f'{model}_development_beta.csv' or sha(a.reference / item['csv_name']) != item['csv_sha256']:
            raise ValueError(f'Wrong R9.7 reference CSV identity: {model}')
    return old, cost, rows, provenance, old_report, sha(report_path)


def protect(data, np):
    for part in data.values():
        for snapshot in part:
            for field in fields(snapshot):
                val = getattr(snapshot, field.name)
                if isinstance(val, np.ndarray):
                    val.flags.writeable = False
    return MappingProxyType({role: tuple(items) for role, items in data.items()})


def fingerprint(data, np):
    result = {}
    for role, items in data.items():
        h = hashlib.sha256()
        for snapshot in items:
            scalar = []
            for field in fields(snapshot):
                value = getattr(snapshot, field.name)
                if isinstance(value, np.ndarray):
                    if value.flags.writeable:
                        raise ValueError('A shared array is writable')
                    h.update(json.dumps((field.name, value.dtype.str, value.shape)).encode())
                    h.update(value.tobytes(order='C'))
                else:
                    scalar.append(value.item() if isinstance(value, np.generic) else value)
            h.update(json.dumps(scalar, ensure_ascii=True, allow_nan=False).encode())
        result[role] = h.hexdigest()
    return result


def scores(p, support):
    keys, vals = [], []
    with p.open(encoding='utf-8', newline='') as stream:
        rd = csv.DictReader(stream)
        if tuple(rd.fieldnames or ()) != (*support, 'predicted_beta'):
            raise ValueError(f'Wrong CSV header: {p.name}')
        for row in rd:
            keys.append(tuple(row[k] for k in support))
            val = float(row['predicted_beta'])
            if not math.isfinite(val):
                raise ValueError('Non-finite beta prediction')
            vals.append(val)
    if len(keys) != 6840 or len({k[:2] for k in keys}) != len(keys):
        raise ValueError('Incomplete or duplicate score support')
    return keys, vals


def compare_score_files(left, right, support, limit):
    k1, v1 = scores(left, support)
    k2, v2 = scores(right, support)
    if k1 != k2:
        raise ValueError(f'Decision keys or true labels changed: {right.name}')
    differences = sorted(abs(x - y) for x, y in zip(v1, v2))
    n = len(differences)
    return k2, {
        'n': n, 'exactly_equal_predictions': sum(x == 0 for x in differences),
        'mean_absolute_difference': math.fsum(differences) / n,
        'p95_absolute_difference_nearest_rank': differences[math.ceil(.95 * n) - 1],
        'max_absolute_difference': differences[-1],
        'soft_cut_beta_gt_one_disagreements': sum((x > 1) != (y > 1) for x, y in zip(v1, v2)),
        'byte_identical': sha(left) == sha(right), 'fixed_absolute_tolerance': limit,
        'within_fixed_absolute_tolerance': differences[-1] <= limit,
    }


def fit_and_write(old, model, data, cfg, ds, out):
    import numpy as np
    import torch
    from fanet import evaluation, source_gated, training

    train, select, scored = (data[r] for r in old.ROLES)
    fit_cfg = {**cfg['training'], 'epochs': old.EPOCHS, 'patience': old.EPOCHS}
    fit_start = time.perf_counter()
    if model == 'extra_trees':
        result = source_gated.fit_current_state_extratrees(train, select, 6, 0.1, old.SEED)
        info = {'residual_scale': result.model.residual_scale, 'n_estimators_per_head': result.model.n_estimators}
    else:
        result = training.train_torch_model('FANET-TopoGNN', train, select, fit_cfg, train[0].pi.size, old.SEED)
        if not isinstance(result.model, torch.nn.Module):
            raise RuntimeError('Real PyTorch model required')
        info = {'epochs': old.EPOCHS, 'batch_size': fit_cfg['batch_size']}
    fit_time = time.perf_counter() - fit_start
    t = time.perf_counter()
    beta, _, _, aligned, _ = evaluation.predict_generic(result, scored)
    prediction_time = time.perf_counter() - t
    keys = [old.target_key(s) for s in aligned]
    if keys != [old.target_key(s) for s in scored] or len(beta) != len(keys) or not np.isfinite(beta).all():
        raise ValueError(f'Changed or invalid prediction support: {model}')
    path = out / f'{model}_development_beta.csv'
    t = time.perf_counter()
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow((*old.SUPPORT, 'predicted_beta'))
        w.writerows((*key, float(pred)) for key, pred in zip(keys, beta))
        f.flush()
        os.fsync(f.fileno())
    return {'fit_seconds': fit_time, 'prediction_seconds': prediction_time,
            'csv_flush_and_fsync_seconds': time.perf_counter() - t,
            'csv_name': path.name, 'csv_bytes': path.stat().st_size,
            'csv_sha256': sha(path), 'details': info}


def run(a):
    start = time.perf_counter()
    old, cost, rows, provenance, baseline, baseline_sha = input_check(a)
    if a.check:
        return {'status': 'R9_8_INPUT_CHECK_PASS_NO_REPLAY', 'r97_report_sha256': baseline_sha,
                'unique_scenario_runs': sum(map(len, rows.values())), **provenance}
    out_root = a.output
    for source in (a.workspace, a.plan, a.r96_code.parent, a.r96_windows_report.parent,
                   a.r97_code.parent, a.reference):
        if out_root == source or out_root in source.parents or source in out_root.parents:
            raise ValueError('R9.8 output cannot overlap earlier artifacts')
    ds = cost.simulator(a.workspace)
    import numpy as np
    import scipy
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits
    from fanet import evaluation, source_gated, training  # preload both stacks before replay
    if not training.TORCH_AVAILABLE or torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('Expected actual CPU-only PyTorch')
    torch.set_num_threads(old.THREADS)
    torch.set_num_interop_threads(1)
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': ds.pd.__version__,
                'scikit_learn': sklearn.__version__, 'torch': str(torch.__version__)}
    if any(baseline['models'][m]['versions'] != versions for m in old.MODELS):
        raise ValueError(f'Use the original R9.7 software versions for parity: {versions}')
    cfg = json.loads((a.workspace / 'configs/r7_full_20seed_isolated.json').read_bytes())
    out = out_root / datetime.now(timezone.utc).strftime('shared_replay_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    setup_seconds = time.perf_counter() - start
    print('R9.8 replaying 108 development scenarios once...', file=sys.stderr, flush=True)
    t = time.perf_counter()
    data = old.generate(cost, ds, cfg['sim'], rows)
    replay_seconds = time.perf_counter() - t
    t = time.perf_counter()
    data = protect(data, np)
    original_fingerprint = fingerprint(data, np)
    initial_audit_seconds = time.perf_counter() - t
    models = {}
    both_keys = None
    with threadpool_limits(limits=old.THREADS):
        for m in old.MODELS:
            print(f'R9.8 fitting {m} with shared read-only data...', file=sys.stderr, flush=True)
            item = fit_and_write(old, m, data, cfg, ds, out)
            t = time.perf_counter()
            if fingerprint(data, np) != original_fingerprint:
                raise ValueError(f'Shared input changed during {m}')
            filename = item['csv_name']
            keys, parity = compare_score_files(a.reference / filename, out / filename, old.SUPPORT, ATOL[m])
            if not parity['within_fixed_absolute_tolerance']:
                raise ValueError(f'R9.7 beta parity outside the fixed tolerance: {m}: {parity}')
            if both_keys is not None and both_keys != keys:
                raise ValueError('Models do not share identical targets and ordering')
            both_keys = keys
            models[m] = {**item, 'parity': parity, 'input_and_output_audit_seconds': time.perf_counter() - t}
    support_sha = hashlib.sha256(json.dumps(both_keys, separators=(',', ':')).encode()).hexdigest()
    if support_sha != REFERENCE_SUPPORT:
        raise ValueError('Shared decision support digest changed')
    peak, memory_measurement = cost.memory()
    baseline_replay = sum(x['development_replay_seconds'] for x in baseline['models'].values())
    baseline_wall = sum(x['worker_wall_seconds'] for x in baseline['models'].values())
    wall = time.perf_counter() - start
    report = {
        'status': 'R9_8_SHARED_REPLAY_PARITY_PASS_PUBLICATION_HOLD', 'schema': 'fanet_r9_8_shared_replay_v1',
        'scientific_submission': 'HOLD', 'blind_holdout_read': False, 'external_freeze_complete': False,
        'calibration_fit_performed': False, 'external_alert_threshold_selected': False,
        'platform': platform.platform(), 'versions': {**versions, 'scipy': scipy.__version__},
        'r98_runner_sha256': sha(Path(__file__)), 'r97_runner_sha256': R97_SHA,
        'r97_report_sha256': baseline_sha, **provenance,
        'replayed_scenario_runs': sum(map(len, rows.values())), 'r97_replayed_scenario_runs': 216,
        'r97_replay_seconds': baseline_replay, 'r98_replay_seconds': replay_seconds,
        'replay_time_reduction_fraction': (baseline_replay - replay_seconds) / baseline_replay,
        'r97_total_worker_wall_seconds': baseline_wall, 'r98_process_wall_seconds_before_report_write': wall,
        'setup_seconds': setup_seconds, 'initial_input_freeze_and_audit_seconds': initial_audit_seconds,
        'shared_peak_process_memory_bytes': peak, 'memory_measurement': memory_measurement,
        'decision_rows': len(both_keys), 'support_sha256': support_sha,
        'all_snapshot_fields_fingerprinted_before_and_after_each_model': True,
        'shared_input_sha256_by_role': original_fingerprint, 'models': models,
        'limits': ['Historical R9.7 timing is not a simultaneous controlled benchmark.',
                   'R9.8 peak is for one shared process, R9.7 peaks are separate workers.',
                   'One development block per role, two neural CPU epochs, no blind test.',
                   'No inference about power, fairness in final protocol, or superiority.'],
    }
    p = out / 'R9_8_SHARED_REPLAY_PILOT.json'
    old.fs_json(p, report)
    return {**report, 'report_path': str(p)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('workspace', 'plan', 'r96-code', 'r96-windows-report', 'r97-code', 'reference'):
        p.add_argument('--' + n, required=True, type=lambda v: Path(v).resolve())
    p.add_argument('--output', type=lambda v: Path(v).resolve())
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if not a.check and a.output is None:
        p.error('--output is required for the pilot')
    for n in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[n] = '' if n == 'CUDA_VISIBLE_DEVICES' else '2'
    print(json.dumps(run(a), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
