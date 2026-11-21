"""Measure two R9.5 development blocks without training or opening holdout files."""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

BLOCKS = (1, 2)
MOBILITIES = ('rwp', 'gm', 'mission')
NODES = (10, 20, 30)
RADIOS = ('nominal', 'degraded')
GRAPHS = ('fixed', 'adaptive')
DEV_ROLES = ('training', 'model_selection_or_early_stopping', 'calibration_fit', 'threshold_selection')
PLAN_DIGEST = 'f65af11c033ff62d40d219e825e5e6cd6a357331af9e90d073d1bfe0385a9894'
SOURCE_HASHES = {
    'configs/r7_full_20seed_isolated.json': '739e0de7da426400e0ddac75dc86317afdf8bbbe412de38104b374ad9e857264',
    'fanet/dataset.py': '8ebf481ad20f056afb3bc0fb991168de61c9c347075fb8fe83e4d1f963edaffb',
    'fanet/geometry.py': 'ac1e3aadc47a52c56020511c602f9f38cfa903822fd668889eed5271fd8ee147',
    'fanet/graph_utils.py': '277a671650a137c5c858cfc4b51689704884450ff2be5a872c69634850d18613',
    'fanet/mobility.py': 'a4a5cd333176bf3b761cc57420bdefcbf1f541db86d7caa3c44a54c91ea949a3',
    'fanet/radio.py': '464370b5bca839877b52077f15d4cbc42edeb21b39d92aee4762456a93fdc453',
    'fanet/topology.py': 'e0bf829aac8163e71ed1ed747334eb9452aa2a79160cb20df0c1d6c0170487e0',
}
PLAN_CSV = 'R9_5_TRAIN_VALIDATION_MATRIX.csv'
REPORT = 'R9_6_COST_PILOT.json'
PASS = 'R9_6_COST_PILOT_COMPLETE_PUBLICATION_HOLD'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace, plan):
    actual = {name: digest(workspace / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise ValueError('R8 simulator/config SHA-256 mismatch: ' +
                         ', '.join(k for k in actual if actual[k] != SOURCE_HASHES[k]))
    manifest_path = plan / 'R9_5_PLAN_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    matrix_path = plan / PLAN_CSV
    if (manifest.get('status') != 'R9_5_MATRIX_GENERATED_AWAITING_EXTERNAL_FREEZE_PUBLICATION_HOLD'
            or manifest.get('r8_config_sha256') != actual['configs/r7_full_20seed_isolated.json']
            or manifest.get('files_sha256', {}).get(PLAN_CSV) != PLAN_DIGEST
            or digest(matrix_path) != PLAN_DIGEST):
        raise ValueError('Pinned R9.5 development plan SHA-256/status mismatch')
    with matrix_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2880 or {r['role'] for r in rows} != set(DEV_ROLES):
        raise ValueError('Expected exactly 2880 development rows and no other roles')
    selected = {i: [r for r in rows if r['role'] == 'training' and int(r['replicate']) == i] for i in BLOCKS}
    expected = {(m, n, radio, graph) for m in MOBILITIES for n in NODES for radio in RADIOS for graph in GRAPHS}
    seeds = set()
    for block in selected.values():
        observed = {(r['mobility'], int(r['n_nodes']), r['radio_scenario'], r['graph_policy']) for r in block}
        group_seeds = {int(r['run_seed']) for r in block}
        if len(block) != 36 or observed != expected or len(group_seeds) != 9 or group_seeds & seeds:
            raise ValueError('Invalid pilot coverage or overlapping trajectory seeds')
        seeds |= group_seeds
    return selected, {'development_csv_sha256': PLAN_DIGEST, 'manifest_sha256': digest(manifest_path),
                      'simulator_sha256': actual}


def memory():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes as w

        class Counters(ctypes.Structure):
            _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                 'PagefileUsage', 'PeakPagefileUsage')]
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = w.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [w.HANDLE, ctypes.POINTER(Counters), w.DWORD]
        get_info.restype = w.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(get_process(), ctypes.byref(counters), counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return int(counters.PeakWorkingSetSize), 'Windows process peak working set'
    import resource
    factor = 1 if sys.platform == 'darwin' else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor), 'process peak RSS'


def simulator(workspace):
    # Load only the original dataset and its siblings. The R8 package initializer
    # eagerly imports the full experiment/training pipeline, which is not needed.
    package = types.ModuleType('fanet')
    package.__path__ = [str(workspace / 'fanet')]
    sys.modules['fanet'] = package
    return importlib.import_module('fanet.dataset')


def validate_run(items, row):
    if len(items) != 200:
        raise ValueError('Expected 200 observed frames')
    for t, s in enumerate(items):
        if s.run_id != row['run_id'] or s.split_group_id != row['split_group_id'] or s.time_index != t:
            raise ValueError('Run identity or time support mismatch')
        if t < 194:
            beta = items[t + 6].beta_current
            if (not s.target_available or s.future_time_index != t + 6
                    or s.beta_target != beta or s.frag_at_horizon != int(beta > 1)):
                raise ValueError('R9.6 h=6 target mismatch')
        elif (s.target_available or s.future_time_index != -1 or s.frag_at_horizon != -1
              or not math.isnan(s.beta_target)):
            raise ValueError('Unobserved target must be unavailable')


def worker(workspace, plan, rep):
    start = time.perf_counter()
    blocks, _ = prepare(workspace, plan)
    data = simulator(workspace)
    config = json.loads((workspace / 'configs/r7_full_20seed_isolated.json').read_text(encoding='utf-8'))['sim']
    setup_s = time.perf_counter() - start
    rows = blocks[rep]
    radio_configs = dict(data._radio_scenarios(config))
    kinematic = {}
    out = []
    sample_start = time.perf_counter()
    for row in rows:
        key = (row['mobility'], int(row['n_nodes']), int(row['run_seed']))
        if key not in kinematic:
            kinematic[key] = data._kinematic_run(config, *key)
    kin_s = time.perf_counter() - sample_start
    sim_s = 0.0
    verify_s = 0.0
    for row in rows:
        key = (row['mobility'], int(row['n_nodes']), int(row['run_seed']))
        cfg = copy.deepcopy(radio_configs[row['radio_scenario']])
        cfg['graph_policy'] = row['graph_policy']
        t = time.perf_counter()
        snapshots = data.simulate_run(cfg, *key, kinematic[key], include_unlabeled_tail=True)
        sim_s += time.perf_counter() - t
        t = time.perf_counter()
        validate_run(snapshots, row)
        verify_s += time.perf_counter() - t
        out.append(snapshots)
    peak, method = memory()
    return {'replicate': rep, 'role': 'training', 'setup_seconds': setup_s,
            'kinematic_seconds': kin_s, 'simulation_seconds': sim_s, 'frame_check_seconds': verify_s,
            'trajectory_groups': len(kinematic), 'scenario_runs': len(out),
            'observed_frames': sum(map(len, out)), 'h6_target_frames': 194 * len(out),
            'shared_w5_h6_decision_frames': 190 * len(out),
            'peak_process_memory_bytes': peak, 'memory_measurement': method,
            'python': platform.python_version(), 'numpy': data.np.__version__, 'pandas': data.pd.__version__}


def execute(args):
    workspace, plan = args.workspace.resolve(), args.plan.resolve()
    blocks, inputs = prepare(workspace, plan)
    if args.worker is not None:
        return worker(workspace, plan, args.worker)
    if args.check:
        return {'status': 'R9_6_INPUT_CHECK_PASS_NO_SIMULATION',
                'pilot_blocks': list(blocks), 'planned_pilot_runs': 72, **inputs}
    output = args.output.resolve()
    if any(output == p or p in output.parents or output in p.parents for p in (workspace, plan)):
        raise ValueError('Output directory cannot overlap R8 or R9.5 inputs')
    results = []
    for rep in BLOCKS:
        print(f'R9.6 measuring training block {rep} of {len(BLOCKS)}...', file=sys.stderr, flush=True)
        t = time.perf_counter()
        result = subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                                 '--workspace', str(workspace), '--plan', str(plan),
                                 '--worker', str(rep)], check=True, stdout=subprocess.PIPE,
                                encoding='utf-8')
        item = json.loads(result.stdout)
        item['worker_wall_seconds'] = time.perf_counter() - t
        results.append(item)
    times = [x['worker_wall_seconds'] for x in results]
    report = {
        'schema': 'fanet_r9_6_development_simulator_cost_pilot_v1', 'status': PASS,
        'scientific_submission': 'HOLD', 'simulation_runs': sum(x['scenario_runs'] for x in results),
        'trajectory_groups': sum(x['trajectory_groups'] for x in results),
        'training_performed': False, 'holdout_file_read': False, 'external_freeze_complete': False,
        'platform': platform.platform(), 'machine_architecture': platform.machine(),
        'logical_cpu_count': os.cpu_count(), 'runner_sha256': digest(Path(__file__)), **inputs,
        'blocks': results,
        'serial_development_only_projection_seconds': {
            'n_blocks': 80, 'low_from_observed_min': min(times) * 80,
            'mid_from_observed_median': statistics.median(times) * 80,
            'high_from_observed_max': max(times) * 80},
        'limits': ['Two training blocks; projection is not an uncertainty interval.',
                   'No model training, export, parallel execution, or blind-test cost measured.',
                   'Memory is a per-worker peak with one block retained, not an all-data estimate.',
                   'No simulated labels or performance scores are exported.'],
    }
    dest = output / datetime.now(timezone.utc).strftime('cost_pilot_%Y%m%dT%H%M%S_%fZ')
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / REPORT).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {**report, 'report_path': str(dest / REPORT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--worker', choices=BLOCKS, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.check and args.worker is None and args.output is None:
        parser.error('--output is required for the pilot')
    print(json.dumps(execute(args), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
