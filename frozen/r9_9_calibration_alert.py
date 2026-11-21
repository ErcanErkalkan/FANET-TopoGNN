"""One pre-specified R9.5 development block per role. Never opens blind data."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

R98_SHA = 'b8dcd73a54fd1a158620440a1e8adce378abfc16ba9ceab0d84160fca5514f05'
R98_REPORT = 'R9_8_SHARED_REPLAY_PILOT.json'
REPORT_SHA = {
    'Linux': 'ff67410a2bda086caa42f9230bd4b6d5ed1e84e4d601b7597fe0414f7f434076',
    'Windows': '82907acf3f8b92c3725564a4139908c06b2aefd00dfad2b4267d6b2a069eeefa',
}
MANIFEST_SHA = 'a16b709b39ee2c43e03522c98d57f321c7ab652feb6bf915c216898674483ef1'
CAL = 'calibration_fit'
SEL = 'threshold_selection'
H, DT, W, LENGTH = 6, 0.1, 5, 200
GRID = tuple(n / 100 for n in range(5, 100, 5))
BUDGET = 2.0
STATUS = 'R9_9_DEVELOPMENT_CALIBRATION_ALERT_PILOT_COMPLETE_PUBLICATION_HOLD'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_verified(path, digest, name):
    if sha(path) != digest:
        raise ValueError(f'Input SHA-256 mismatch: {path.name}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_audit(rows, cost, roles):
    cells = {(m, n, r, g) for m in cost.MOBILITIES for n in cost.NODES
             for r in cost.RADIOS for g in cost.GRAPHS}
    previous_groups, previous_seeds, previous_runs = set(), set(), set()
    if set(rows) != set(roles):
        raise ValueError('Wrong development roles')
    for role, part in rows.items():
        if any(r['role'] != role or int(r['replicate']) != 1 or
               int(r['horizon_steps']) != H or int(r['simulator_steps']) != LENGTH or
               float(r['dt']) != DT for r in part):
            raise ValueError(f'Role or temporal protocol mismatch: {role}')
        coverage = {(r['mobility'], int(r['n_nodes']), r['radio_scenario'], r['graph_policy']) for r in part}
        groups = Counter(r['split_group_id'] for r in part)
        seeds = {int(r['run_seed']) for r in part}
        runs = {r['run_id'] for r in part}
        mapping = {(r['split_group_id'], int(r['run_seed']), r['mobility'], int(r['n_nodes'])) for r in part}
        if (len(part) != 36 or coverage != cells or len(groups) != 9 or set(groups.values()) != {4} or
                len(seeds) != 9 or len(mapping) != 9 or len(runs) != 36 or
                previous_groups.intersection(groups) or previous_seeds & seeds or previous_runs & runs):
            raise ValueError(f'Invalid scenario coverage or cross-role overlap: {role}')
        previous_groups.update(groups)
        previous_seeds.update(seeds)
        previous_runs.update(runs)


def input_check(a):
    shared = load_verified(a.r98_code, R98_SHA, 'r98_verified')
    old = load_verified(a.r97_code, shared.R97_SHA, 'r97_verified')
    cost, rows, provenance = old.input_check(a)
    if provenance['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('R9.5 manifest changed')
    rows[CAL] = []
    with (a.plan / cost.PLAN_CSV).open(encoding='utf-8', newline='') as f:
        rows[CAL].extend(r for r in csv.DictReader(f) if r['role'] == CAL and int(r['replicate']) == 1)
    group_audit(rows, cost, (*old.ROLES, CAL))
    ref_path = a.reference / R98_REPORT
    ref_sha = sha(ref_path)
    if ref_sha != REPORT_SHA.get(platform.system()):
        raise ValueError('Use the pinned same-platform R9.8 report and beta CSVs')
    ref = json.loads(ref_path.read_bytes())
    if (ref['status'] != 'R9_8_SHARED_REPLAY_PARITY_PASS_PUBLICATION_HOLD' or
            ref['r98_runner_sha256'] != R98_SHA or ref['support_sha256'] != shared.REFERENCE_SUPPORT or
            ref['blind_holdout_read'] is not False):
        raise ValueError('R9.8 provenance mismatch')
    for m in old.MODELS:
        f = f'{m}_development_beta.csv'
        if ref['models'][m]['csv_name'] != f or sha(a.reference / f) != ref['models'][m]['csv_sha256']:
            raise ValueError(f'R9.8 reference beta changed: {f}')
    return shared, old, cost, rows, ref, {**provenance, 'r98_code_sha256': R98_SHA,
                                         'r98_reference_report_sha256': ref_sha}


@dataclass(frozen=True)
class Truth:
    run_id: str
    time_index: int
    beta_current: float


def replay(shared, old, cost, ds, sim, rows, np):
    import copy
    radios = dict(ds._radio_scenarios(sim))
    data, truth = {}, []
    for role, part in rows.items():
        kinematic, eligible = {}, []
        for r in part:
            key = (r['mobility'], int(r['n_nodes']), int(r['run_seed']))
            if key not in kinematic:
                kinematic[key] = ds._kinematic_run(sim, *key)
            cfg = copy.deepcopy(radios[r['radio_scenario']])
            cfg['graph_policy'] = r['graph_policy']
            path = ds.simulate_run(cfg, *key, kinematic[key], include_unlabeled_tail=True)
            cost.validate_run(path, r)
            if role == SEL:
                truth.extend(Truth(s.run_id, s.time_index, float(s.beta_current)) for s in path)
            eligible.extend(path[W - 1:LENGTH - H])
        if len(eligible) != 6840:
            raise ValueError(f'Incomplete decision support: {role}')
        data[role] = sorted(eligible, key=lambda s: (s.run_id, s.time_index))
    return shared.protect(data, np), tuple(truth)


def operating_point(candidates):
    valid = [item for item in candidates if item['within_budget']]
    if valid:
        winner = min(valid, key=lambda x: (-x['Alert_Event_F1'], x['False_Alert_Events_per_minute'],
                                            -x['Alert_Event_Recall'], x['threshold']))
        if not winner['True_Alert_Events']:
            return winner, 'NO_TRUE_ALERT_UNDER_BUDGET'
        return winner, 'DESCRIPTIVE_ONLY_IN_SAMPLE_OPERATING_POINT'
    winner = min(candidates, key=lambda x: (x['False_Alert_Events_per_minute'], -x['Alert_Event_F1'],
                                            x['threshold']))
    return winner, 'NO_THRESHOLD_UNDER_BUDGET'


def write_csv(path, header, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)
    return sha(path)


def beta_scores(old, evaluator, result, scored, np):
    beta, _, _, aligned, _ = evaluator.predict_generic(result, scored)
    keys = [old.target_key(s) for s in aligned]
    if keys != [old.target_key(s) for s in scored] or beta.shape != (len(keys),) or not np.isfinite(beta).all():
        raise ValueError('Prediction decision support or numerical validity changed')
    return keys, beta


def warnable_events(full, scored):
    # Full event denominator matches R8. This extra count is diagnostic only.
    lookup = {(s.run_id, s.time_index) for s in scored if s.beta_current <= 1}
    prev, events, possible = None, 0, 0
    for s in full:
        if prev and prev.run_id == s.run_id and prev.beta_current <= 1 < s.beta_current:
            events += 1
            possible += any((s.run_id, t) in lookup for t in range(s.time_index - H, s.time_index))
        prev = s
    return events, possible


def calibration_fit(beta, labels, seed, np, sklearn):
    from sklearn.linear_model import LogisticRegression
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError('Calibration role has one class; cannot fit logistic calibration')
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error', sklearn.exceptions.ConvergenceWarning)
        clf.fit(beta.reshape(-1, 1), labels)
    return clf


def eval_grid(evaluate, ds, scores, truth, np):
    if scores.shape != (len(ds),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid calibrated probabilities')
    candidates = []
    for threshold in GRID:
        ms = evaluate(ds, scores, truth, threshold, DT, H, H)
        candidates.append({'threshold': threshold,
                           'within_budget': ms['False_Alert_Events_per_minute'] <= BUDGET, **ms})
    return candidates


def run(a):
    shared, old, cost, rows, ref, provenance = input_check(a)
    roles = {r: {'runs': len(part), 'trajectory_groups': 9, 'decision_rows': 6840} for r, part in rows.items()}
    if a.check:
        return {'status': 'R9_9_CHECK_PASS_NO_SIMULATION', 'roles': roles, **provenance}
    for source in (a.workspace, a.plan, a.r96_code.parent, a.r96_windows_report.parent,
                   a.r97_code.parent, a.r98_code.parent, a.reference):
        if a.output == source or a.output in source.parents or source in a.output.parents:
            raise ValueError('Output must not overlap any frozen source input or prior phase')
    start = time.perf_counter()
    ds = cost.simulator(a.workspace)
    import numpy as np
    import scipy
    import sklearn
    import torch
    from sklearn import exceptions  # ensure sklearn.exceptions is loaded
    from threadpoolctl import threadpool_limits
    from fanet import evaluation, source_gated, training
    if not training.TORCH_AVAILABLE or torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('The frozen pilot uses real CPU-only PyTorch')
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': ds.pd.__version__, 'scikit_learn': sklearn.__version__,
                'scipy': scipy.__version__, 'torch': str(torch.__version__)}
    if versions != ref['versions']:
        raise ValueError(f'Use the pinned same-platform R9.8 dependency versions: {versions}')
    torch.set_num_threads(old.THREADS)
    torch.set_num_interop_threads(1)
    cfg = json.loads((a.workspace / 'configs/r7_full_20seed_isolated.json').read_bytes())
    out = a.output / datetime.now(timezone.utc).strftime('calibration_alert_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('R9.9 replaying 144 development scenarios. No blind data.', file=sys.stderr, flush=True)
    t = time.perf_counter()
    data, truth = replay(shared, old, cost, ds, cfg['sim'], rows, np)
    replay_seconds = time.perf_counter() - t
    fingerprints = shared.fingerprint(data, np)
    for role in old.ROLES:
        if fingerprints[role] != ref['shared_input_sha256_by_role'][role]:
            raise ValueError(f'R9.8 input fingerprint changed: {role}')
    train, choose, select = (data[r] for r in old.ROLES)
    cal = data[CAL]
    config = {**cfg['training'], 'epochs': old.EPOCHS, 'patience': old.EPOCHS}
    y_cal, y_sel = (np.asarray([s.frag_at_horizon for s in part], dtype=int) for part in (cal, select))
    dev_prev = float(y_cal.mean())
    binary_counts = {r: {str(k): int((y == k).sum()) for k in (0, 1)}
                     for r, y in ((CAL, y_cal), (SEL, y_sel))}
    events, possible = warnable_events(truth, select)
    models, output_digests = {}, {}
    with threadpool_limits(limits=old.THREADS):
        for m in old.MODELS:
            print(f'R9.9 fitting, calibrating, selecting: {m}', file=sys.stderr, flush=True)
            t = time.perf_counter()
            if m == 'extra_trees':
                result = source_gated.fit_current_state_extratrees(train, choose, H, DT, old.SEED)
            else:
                result = training.train_torch_model('FANET-TopoGNN', train, choose, config, train[0].pi.size, old.SEED)
                if not isinstance(result.model, torch.nn.Module):
                    raise RuntimeError('No surrogate model allowed')
            selection_keys, sel_beta = beta_scores(old, evaluation, result, select, np)
            cal_keys, cal_beta = beta_scores(old, evaluation, result, cal, np)
            bname = f'{m}_development_beta.csv'
            output_digests[bname] = write_csv(out / bname, (*old.SUPPORT, 'predicted_beta'),
                                             ((*key, float(b)) for key, b in zip(selection_keys, sel_beta)))
            ref_keys, parity = shared.compare_score_files(a.reference / bname, out / bname,
                                                          old.SUPPORT, shared.ATOL[m])
            if (not parity['within_fixed_absolute_tolerance'] or
                    hashlib.sha256(json.dumps(ref_keys, separators=(',', ':')).encode()).hexdigest()
                    != shared.REFERENCE_SUPPORT):
                raise ValueError(f'R9.8 regression parity failed: {m}')
            lr = calibration_fit(cal_beta, y_cal, old.SEED, np, sklearn)
            probabilities = lr.predict_proba(sel_beta.reshape(-1, 1))[:, 1]
            candidates = eval_grid(evaluation.alert_event_metrics_full_trajectory, select, probabilities, truth, np)
            selected, interpretation = operating_point(candidates)
            if selected['Ground_Truth_Fragmentation_Events'] != events:
                raise ValueError('Unexpected full-trajectory event denominator')
            fname = f'{m}_threshold_grid.csv'
            header = (*candidates[0], 'diagnostic_selected')
            output_digests[fname] = write_csv(out / fname, header,
                                             ((*[row[k] for k in candidates[0]], row is selected)
                                              for row in candidates))
            for role, part, keys, betas, probs in ((CAL, cal, cal_keys, cal_beta, None),
                                                   (SEL, select, selection_keys, sel_beta, probabilities)):
                name = f'{m}_{role}_inputs.csv'
                more = ('calibrated_probability',) if probs is not None else ()
                output_digests[name] = write_csv(
                    out / name, (*old.SUPPORT, 'beta_current', 'predicted_beta', *more),
                    ((*k, float(s.beta_current), float(betas[i]), *(() if probs is None else (float(probs[i]),)))
                     for i, (s, k) in enumerate(zip(part, keys))))
            if shared.fingerprint(data, np) != fingerprints:
                raise ValueError(f'Shared snapshot fields changed during {m}')
            connected = np.asarray([s.beta_current <= 1 for s in select])
            models[m] = {
                'runtime_seconds': time.perf_counter() - t,
                'r98_regression_parity': parity,
                'calibrator': {'slope': float(lr.coef_[0, 0]), 'intercept': float(lr.intercept_[0]),
                               'unique_calibration_beta_count': int(len(np.unique(cal_beta)))},
                'selection_probability_range': [float(probabilities.min()), float(probabilities.max())],
                'selection_role_brier_all_rows': float(np.mean((probabilities - y_sel) ** 2)),
                'selection_role_brier_connected_rows': float(np.mean((probabilities[connected] - y_sel[connected]) ** 2))
                if connected.any() else None,
                'feasible_grid_points': sum(c['within_budget'] for c in candidates),
                'feasible_grid_points_with_true_alert': sum(c['within_budget'] and c['True_Alert_Events'] > 0
                                                           for c in candidates),
                'selection_diagnostic_only': selected, 'operating_point_status': interpretation,
                'deployment_threshold': None,
            }
    name = 'threshold_role_full_truth.csv'
    output_digests[name] = write_csv(out / name, ('run_id', 'time_index', 'beta_current'),
                                     ((s.run_id, s.time_index, s.beta_current) for s in truth))
    peak, memory_method = cost.memory()
    p = out / 'R9_9_CALIBRATION_ALERT_PILOT.json'
    report = {'schema': 'fanet_r9_9_development_calibration_alert_pilot_v1', 'status': STATUS,
              'scientific_submission': 'HOLD', 'blind_holdout_read': False, 'external_freeze_complete': False,
              'platform': platform.platform(), 'versions': versions, 'runner_sha256': sha(Path(__file__)),
              **provenance, 'replicate': 1, 'model_seed': old.SEED, 'topognn_epochs': old.EPOCHS,
              'roles': roles, 'replayed_runs': sum(map(len, rows.values())),
              'calibration_and_selection_label_counts': binary_counts,
              'selection_prevalence_baseline_brier_all_rows': float(np.mean((dev_prev - y_sel) ** 2)),
              'threshold_role_full_truth_frames': len(truth), 'threshold_role_exposure_minutes': len(truth) * DT / 60,
              'threshold_role_total_events': events, 'threshold_role_warnable_events_before_cooldown_and_matching': possible,
              'horizon_steps': H, 'dt_seconds': DT, 'common_warmup_steps': W - 1, 'cooldown_steps': H,
              'calibrator_input': 'regression_beta_only', 'calibrator_C': 1.0,
              'false_alert_budget_per_full_trajectory_minute': BUDGET, 'threshold_grid': list(GRID),
              'event_policy': 'frozen R8 alert_event_metrics_full_trajectory',
              'input_fingerprints': fingerprints, 'output_sha256': output_digests, 'models': models,
              'simulation_seconds': replay_seconds, 'elapsed_seconds_before_final_report': time.perf_counter() - start,
              'peak_process_memory_bytes': peak, 'memory_measurement': memory_method,
              'limits': ['One block per role and two neural CPU epochs.',
                         'Threshold-role event metrics use the data used to choose the threshold.',
                         'No multi-block stability, deployment threshold, superiority, power or publication claim.',
                         'Reported Brier is descriptive for the threshold-selection role, not confirmatory evidence.']}
    old.fs_json(p, report)
    return {**report, 'report_path': str(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'plan', 'r96-code', 'r96-windows-report', 'r97-code', 'r98-code', 'reference'):
        ap.add_argument('--' + name, type=lambda s: Path(s).resolve(), required=True)
    ap.add_argument('--output', type=lambda s: Path(s).resolve())
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if not a.check and a.output is None:
        ap.error('--output is required for a pilot')
    for name in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '' if name == 'CUDA_VISIBLE_DEVICES' else '2'
    print(json.dumps(run(a), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
