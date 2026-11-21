"""Bounded development-only training and writing benchmark of the frozen R8 models."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPLICATE = 1
EPOCHS = 2
SEED = 97114
THREADS = 2
ROLES = ('training', 'model_selection_or_early_stopping', 'threshold_selection')
MODELS = ('extra_trees', 'fanet_topognn')
SUPPORT = ('run_id', 'time_index', 'split_group_id', 'future_time_index', 'frag_at_horizon', 'beta_target')
STATUS = 'R9_7_TRAINING_WRITE_PILOT_COMPLETE_PUBLICATION_HOLD'
R96_CODE_SHA = '71134ab16b82cd0e62fc760d242df736687fe81b88f1e7282a7acef279dc1e4d'
R96_WINDOWS_REPORT_SHA = '6e4faa3986d7eb0d5f486dd21e512058dc324c8dfcf713c14220925e33004b51'
MODELING_SHA = {
    'fanet/training.py': 'ec15e1b4efddd8d3c3f4fe72f98d4c1690e1ae3ec033c8c8d79f82bd32ae9d2d',
    'fanet/models.py': '981c1e8c5d6babe2b1e7e467a43ddf661f77880848103bb0e235eb7bc3739c78',
    'fanet/source_gated.py': '325bc015a45ae3a2a18da395f289b0948d9d55e69ff72e52482d75a7a7ada008',
    'fanet/evaluation.py': '41bbc56d9db1e8efaa3b9181b596aca980633da0758f251a685936c95c0110c6',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fs_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(value, output, indent=2, ensure_ascii=True, allow_nan=False)
        output.write('\n')


def input_check(a):
    if sha(a.r96_code) != R96_CODE_SHA or sha(a.r96_windows_report) != R96_WINDOWS_REPORT_SHA:
        raise ValueError('R9.6 source or Windows receipt SHA-256 mismatch')
    spec = importlib.util.spec_from_file_location('cost_support', a.r96_code)
    cost = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cost)
    _, provenance = cost.prepare(a.workspace, a.plan)
    receipt = json.loads(a.r96_windows_report.read_text(encoding='utf-8'))
    if (receipt.get('status') != cost.PASS or receipt.get('simulation_runs') != 72
            or receipt.get('runner_sha256') != R96_CODE_SHA
            or receipt.get('development_csv_sha256') != provenance['development_csv_sha256']
            or receipt.get('simulator_sha256') != provenance['simulator_sha256']
            or receipt.get('holdout_file_read') is not False):
        raise ValueError('R9.6 Windows evidence does not match the pilot inputs')
    for name, expected in MODELING_SHA.items():
        if sha(a.workspace / name) != expected:
            raise ValueError(f'R8 model source SHA-256 mismatch: {name}')
    with (a.plan / cost.PLAN_CSV).open(newline='', encoding='utf-8') as f:
        rows = {r: [] for r in ROLES}
        for row in csv.DictReader(f):
            if row['role'] in rows and int(row['replicate']) == REPLICATE:
                rows[row['role']].append(row)
    cells = {(m, n, c, g) for m in cost.MOBILITIES for n in cost.NODES
             for c in cost.RADIOS for g in cost.GRAPHS}
    previous = set()
    for role, part in rows.items():
        got = {(r['mobility'], int(r['n_nodes']), r['radio_scenario'], r['graph_policy']) for r in part}
        groups = {r['split_group_id'] for r in part}
        seeds = {int(r['run_seed']) for r in part}
        if len(part) != 36 or got != cells or len(groups) != 9 or len(seeds) != 9 or previous & seeds:
            raise ValueError(f'Invalid coverage or cross-role seed overlap: {role}')
        previous |= seeds
    return cost, rows, {**provenance, 'model_source_sha256': MODELING_SHA,
                         'r96_code_sha256': R96_CODE_SHA, 'r96_windows_report_sha256': R96_WINDOWS_REPORT_SHA,
                         'r96_windows_simulator_only_projection_seconds': receipt['serial_development_only_projection_seconds']}


def generate(cost, ds, config, rows):
    radios = dict(ds._radio_scenarios(config))
    snaps = {}
    for role in ROLES:
        kinematic = {}
        selected = []
        for r in rows[role]:
            key = r['mobility'], int(r['n_nodes']), int(r['run_seed'])
            if key not in kinematic:
                kinematic[key] = ds._kinematic_run(config, *key)
            cfg = copy.deepcopy(radios[r['radio_scenario']])
            cfg['graph_policy'] = r['graph_policy']
            path = ds.simulate_run(cfg, *key, kinematic[key], include_unlabeled_tail=True)
            cost.validate_run(path, r)
            # Common support starting after the four-frame warm-up.
            selected.extend(path[4:194])
        if len(selected) != 6840:
            raise ValueError(f'Unexpected support: {role}')
        snaps[role] = sorted(selected, key=lambda s: (s.run_id, s.time_index))
    return snaps


def target_key(s):
    return s.run_id, s.time_index, s.split_group_id, s.future_time_index, s.frag_at_horizon, float(s.beta_target)


def read_scores(path):
    import math
    keys = []
    with path.open(encoding='utf-8', newline='') as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != (*SUPPORT, 'predicted_beta'):
            raise ValueError(f'Score schema mismatch: {path.name}')
        for row in rd:
            if not math.isfinite(float(row['predicted_beta'])):
                raise ValueError('Non-finite beta prediction')
            keys.append(tuple(row[k] for k in SUPPORT))
    if len(keys) != 6840 or len({k[:2] for k in keys}) != len(keys):
        raise ValueError(f'Wrong or duplicate decision support: {path.name}')
    return keys


def worker(a):
    t = time.perf_counter()
    cost, rows, _ = input_check(a)
    ds = cost.simulator(a.workspace)
    import numpy as np
    import sklearn
    import torch
    from threadpoolctl import threadpool_limits
    from fanet import evaluation, source_gated, training
    if not training.TORCH_AVAILABLE or torch.cuda.is_available():
        raise RuntimeError('Expected real PyTorch using CPU only')
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    cfg = json.loads((a.workspace / 'configs/r7_full_20seed_isolated.json').read_text(encoding='utf-8'))
    setup_seconds = time.perf_counter() - t
    t = time.perf_counter()
    snaps = generate(cost, ds, cfg['sim'], rows)
    data_seconds = time.perf_counter() - t
    train, select, scored = (snaps[r] for r in ROLES)
    fit_cfg = {**cfg['training'], 'epochs': EPOCHS, 'patience': EPOCHS}
    with threadpool_limits(limits=THREADS):
        t = time.perf_counter()
        if a.worker == 'extra_trees':
            result = source_gated.fit_current_state_extratrees(train, select, 6, 0.1, SEED)
            extra = {'residual_scale': result.model.residual_scale, 'n_estimators_per_head': result.model.n_estimators,
                     'native_risk_head_fitted_but_not_exported': True}
        else:
            result = training.train_torch_model('FANET-TopoGNN', train, select, fit_cfg, train[0].pi.size, SEED)
            if not isinstance(result.model, torch.nn.Module):
                raise RuntimeError('A PyTorch surrogate is not permitted')
            extra = {'epochs': EPOCHS, 'batch_size': fit_cfg['batch_size']}
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        beta, _, _, aligned, _ = evaluation.predict_generic(result, scored)
        prediction_seconds = time.perf_counter() - t
    supports = [target_key(s) for s in aligned]
    if supports != [target_key(s) for s in scored] or len(beta) != len(scored) or not np.isfinite(beta).all():
        raise ValueError(f'Prediction support mismatch: {a.worker}')
    path = a.output / f'{a.worker}_development_beta.csv'
    t = time.perf_counter()
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow((*SUPPORT, 'predicted_beta'))
        w.writerows((*key, float(pred)) for key, pred in zip(supports, beta))
        f.flush()
        os.fsync(f.fileno())
    write_seconds = time.perf_counter() - t
    peak, metric = cost.memory()
    return {'model': result.model_name, 'setup_seconds': setup_seconds, 'development_replay_seconds': data_seconds,
            'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
            'csv_flush_and_fsync_seconds': write_seconds, 'csv_bytes': path.stat().st_size,
            'csv_sha256': sha(path), 'csv_name': path.name, 'common_decision_rows': len(scored),
            'peak_process_memory_bytes': peak, 'memory_measurement': metric, 'details': extra,
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': ds.pd.__version__,
                         'scikit_learn': sklearn.__version__, 'torch': str(torch.__version__)}}


def run(a):
    _, rows, provenance = input_check(a)
    if a.check:
        return {'status': 'R9_7_INPUT_CHECK_PASS_NO_TRAINING', 'replicate': REPLICATE,
                'scenario_runs_by_role': {k: len(v) for k, v in rows.items()}, **provenance}
    loc = a.output.resolve()
    for source in (a.workspace, a.plan, a.r96_code.parent, a.r96_windows_report.parent):
        if loc == source or loc in source.parents or source in loc.parents:
            raise ValueError('R9.7 output cannot overlap frozen source inputs')
    dest = loc / datetime.now(timezone.utc).strftime('training_write_%Y%m%dT%H%M%S_%fZ')
    dest.mkdir(parents=True, exist_ok=False)
    records = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
               OPENBLAS_NUM_THREADS=str(THREADS))
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
               '--workspace', str(a.workspace), '--plan', str(a.plan),
               '--r96-code', str(a.r96_code), '--r96-windows-report', str(a.r96_windows_report),
               '--output', str(dest)]
    for model in MODELS:
        print(f'R9.7 replaying 108 development runs and fitting {model}...', file=sys.stderr, flush=True)
        t = time.perf_counter()
        r = subprocess.run([*command, '--worker', model], check=True, capture_output=True, encoding='utf-8', env=env)
        item = json.loads(r.stdout)
        item['worker_wall_seconds'] = time.perf_counter() - t
        records[model] = item
    left, right = (read_scores(dest / records[m]['csv_name']) for m in MODELS)
    if left != right:
        raise ValueError('The model CSVs do not share identical decision keys and reference labels')
    support_sha = hashlib.sha256(json.dumps(left, separators=(',', ':')).encode()).hexdigest()
    report = {
        'schema': 'fanet_r9_7_training_write_pilot_v1', 'status': STATUS,
        'scientific_submission': 'HOLD', 'blind_holdout_read': False, 'external_freeze_complete': False,
        'replicate': REPLICATE, 'fit_seed': SEED, 'cpu_threads_per_worker': THREADS,
        'platform': platform.platform(), 'roles': {r: {'scenario_runs': len(items), 'trajectory_groups': 9,
                                                       'common_h6_w5_rows': 6840} for r, items in rows.items()},
        'unique_scenario_runs': sum(map(len, rows.values())), 'scenario_replays_across_both_workers': 216,
        'scored_role': 'threshold_selection', 'common_decision_rows': len(left), 'support_sha256': support_sha,
        'calibration_fit_performed': False, 'external_alert_threshold_selected': False,
        'runner_sha256': sha(Path(__file__)), **provenance, 'models': records,
        'limits': ['Training and scoring costs are for the frozen model trainers on this one development block per role.',
                   'ExtraTrees fit includes its built-in native risk head and its internal validation threshold; neither is used here.',
                   'Only regression-score CSVs were written. No checkpoints or deployment metrics.',
                   'Memory is total peak for each isolated worker including imports, replay, fit, scoring and export.',
                   'CSV time excludes Google Drive sync. These timings do not validate sample size, scientific claims or full-budget runtime.'],
    }
    path = dest / 'R9_7_TRAINING_WRITE_PILOT.json'
    fs_json(path, report)
    return {**report, 'report_path': str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'plan', 'r96-code', 'r96-windows-report'):
        p.add_argument('--' + name, required=True, type=lambda v: Path(v).resolve())
    p.add_argument('--output', type=lambda v: Path(v).resolve())
    g = p.add_mutually_exclusive_group()
    g.add_argument('--check', action='store_true')
    g.add_argument('--worker', choices=MODELS, help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.check and a.output is None:
        p.error('--output is required for fitting')
    try:
        out = worker(a) if a.worker else run(a)
    except subprocess.CalledProcessError as e:
        print(e.stderr or e.stdout or str(e), file=sys.stderr)
        raise
    print(json.dumps(out, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
