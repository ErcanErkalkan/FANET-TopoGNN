"""R7 full-trajectory event accounting and strict per-model score provenance.

Scientific scope: independent post-review protocol.  Never retrofit the
published/locked Event-F1 from this newer definition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluation import alert_event_metrics_full_trajectory

SCORE_FIELDS = (
    'run_id', 'time_index', 'split_group_id', 'future_time_index',
    'beta_current', 'beta_target', 'frag_at_horizon', 'predicted_beta',
    'risk_score', 'selected_threshold', 'score_kind', 'model',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_full_test_trajectory(full_snapshots: Sequence, test_snapshots: Sequence) -> list:
    """Select the complete original test runs using frozen, grouped split IDs."""
    groups = {str(s.split_group_id) for s in test_snapshots}
    if not groups:
        raise ValueError('Empty test partition')
    expected_runs = {str(s.run_id) for s in test_snapshots}
    full = [s for s in full_snapshots if str(s.split_group_id) in groups]
    actual_runs = {str(s.run_id) for s in full}
    if actual_runs != expected_runs:
        raise ValueError('Complete test trajectory run inventory differs from scored split')
    if any(not getattr(s, 'target_available', True) for s in test_snapshots):
        raise ValueError('Unlabeled frame in supervised test partition')
    if len(full) <= len(test_snapshots):
        raise ValueError('Missing terminal full-trajectory frames (expected T > T-h)')
    return full


def score_and_audit(*, scored_snapshots: Sequence, predictions: Sequence[float],
                    risks: Sequence[float], full_test_trajectory: Sequence,
                    dt: float, horizon_steps: int, threshold: float, model: str,
                    score_kind: str, destination: Path | None = None) -> dict:
    """Recompute events on observed full truth; optionally export immutable CSV."""
    scored = list(scored_snapshots)
    prediction = np.asarray(predictions, dtype=float)
    risk = np.asarray(risks, dtype=float)
    if not scored or prediction.shape != (len(scored),) or risk.shape != (len(scored),):
        raise ValueError('One-to-one aligned prediction and risk scores required')
    if not np.isfinite(prediction).all() or not np.isfinite(risk).all():
        raise ValueError('Non-finite predicted beta or score')
    if np.any(risk < 0) or np.any(risk > 1):
        raise ValueError('Risk score outside [0,1]')
    if score_kind not in ('validation_calibrated_probability', 'uncalibrated_or_binary_risk_score'):
        raise ValueError('The score provenance must be declared')
    if not model.strip():
        raise ValueError('Model name required')
    full = list(full_test_trajectory)
    if {str(s.run_id) for s in scored} != {str(s.run_id) for s in full}:
        raise ValueError('Scored and full-truth runs do not match')
    metrics = alert_event_metrics_full_trajectory(
        scored, risk, full, threshold=threshold, dt=dt, horizon_steps=horizon_steps,
    )
    output = {
        'schema': 'fanet_r7_full_timeline_event_provenance_v1',
        'model': model, 'score_kind': score_kind,
        'forecast_scored_rows': len(scored), 'full_truth_rows': len(full),
        'horizon_steps': int(horizon_steps), 'dt_s': float(dt),
        'selected_threshold': float(threshold), 'full_timeline_event_metrics': metrics,
        'legacy_event_metrics_replaced': False, 'scientific_submission': 'HOLD',
    }
    if destination is not None:
        path = Path(destination)
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError('Cannot overwrite any R7 prediction or provenance artifact')
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = set()
        with path.open('x', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            for snap, beta, r in zip(scored, prediction, risk):
                key = (str(snap.run_id), int(snap.time_index))
                if key in keys:
                    raise ValueError('Duplicate scored run/time keys')
                keys.add(key)
                writer.writerow({
                    'run_id': key[0], 'time_index': key[1],
                    'split_group_id': str(snap.split_group_id),
                    'future_time_index': int(snap.future_time_index),
                    'beta_current': float(snap.beta_current),
                    'beta_target': float(snap.beta_target),
                    'frag_at_horizon': int(snap.frag_at_horizon),
                    'predicted_beta': float(beta), 'risk_score': float(r),
                    'selected_threshold': float(threshold),
                    'score_kind': score_kind, 'model': model,
                })
        output['score_csv_sha256'] = sha256(path)
        path.with_suffix('.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return output


def validate_support_inventory(csv_paths: Sequence[Path], expected_models: Sequence[str]) -> dict:
    """Fail closed on missing models or nonidentical (run,time) support."""
    paths = [Path(p) for p in csv_paths]
    if len(paths) != len(expected_models) or not paths:
        raise ValueError('Missing model score files')
    expected_models = list(expected_models)
    if len(set(expected_models)) != len(expected_models):
        raise ValueError('Duplicate expected models')
    supports = {}
    reference_truth = None
    for path in paths:
        if not path.is_file() or not path.with_suffix('.json').is_file():
            raise FileNotFoundError(path)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if sha256(path) != metadata['score_csv_sha256']:
            raise ValueError(f'Score file hash mismatch: {path}')
        model = metadata['model']
        if model in supports:
            raise ValueError('Duplicate score model')
        with path.open(newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        keys = [(r['run_id'], int(r['time_index'])) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate score keys: {model}')
        if len(keys) != metadata['forecast_scored_rows']:
            raise ValueError(f'Row-count mismatch: {model}')
        truth = {}
        for key, r in zip(keys, rows):
            if r['model'] != model or r['score_kind'] != metadata['score_kind']:
                raise ValueError(f'Per-row model/provenance mismatch: {model}')
            p = float(r['risk_score'])
            if not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(float(r['predicted_beta'])):
                raise ValueError(f'Invalid predictions for model: {model}')
            threshold = float(r['selected_threshold'])
            if not math.isfinite(threshold) or not math.isclose(threshold, float(metadata['selected_threshold']),abs_tol=1e-12):
                raise ValueError(f'Threshold drift in model: {model}')
            if int(r['future_time_index']) - int(r['time_index']) != metadata['horizon_steps']:
                raise ValueError(f'Horizon mismatch in model: {model}')
            if int(r['frag_at_horizon']) != int(float(r['beta_target']) > 1):
                raise ValueError(f'Inconsistent risk label: {model}')
            truth[key] = (r['split_group_id'], r['future_time_index'],
                          r['beta_current'], r['beta_target'], r['frag_at_horizon'])
        if reference_truth is None:
            reference_truth = truth
        elif truth != reference_truth:
            raise ValueError(f'Nonidentical target truth or test time support: {model}')
        supports[model] = set(keys)
    if set(supports) != set(expected_models):
        raise ValueError('Incomplete or unexpected model inventory')
    reference = supports[expected_models[0]]
    for model, support in supports.items():
        if support != reference:
            raise ValueError(f'Nonidentical test time support: {model}')
    return {'status': 'COMMON_TEST_SUPPORT_PASS', 'models': expected_models,
            'rows_per_model': len(reference), 'model_count': len(supports)}
