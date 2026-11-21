"""R4 neural model evidence export for *future* strictly controlled reruns.

This module never reinterprets old 280/280 metrics as fresh model predictions.
It does not load untrusted pickles or select thresholds on the test set.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np

NEURAL_NAMES = frozenset({
    'GCN', 'GAT', 'GraphSAGE', 'PI+MLP', 'FANET-TopoGNN',
    'FANET-TopoGNN (concat)', 'tgcn:5', 'stgcn:5', 'tgn:5',
})
SCORE_COLUMNS = (
    'run_id', 'time_index', 'split_group_id', 'future_time_index',
    'frag_at_horizon', 'beta_current', 'beta_target', 'predicted_beta',
    'risk_probability', 'score_origin',
)


def is_neural_task(task_name: str) -> bool:
    return task_name in NEURAL_NAMES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _slug(task_name: str) -> str:
    import re
    return re.sub(r'[^A-Za-z0-9._-]+', '_', task_name)


def evidence_location(seed_dir: Path, task_name: str) -> Path:
    return Path(seed_dir) / 'r4_raw_evidence' / _slug(task_name)


def _check_split(snaps: Sequence[object], predictions: Sequence[float], horizon: int, split: str) -> np.ndarray:
    preds = np.asarray(predictions, dtype=np.float64)
    if preds.ndim != 1 or preds.size != len(snaps) or not np.all(np.isfinite(preds)):
        raise ValueError(f'{split}: missing/invalid regression predictions or alignment')
    if not snaps:
        raise ValueError(f'{split}: empty split')
    keys = set()
    for snap in snaps:
        k = (str(snap.run_id), int(snap.time_index))
        if k in keys:
            raise ValueError(f'{split}: duplicate (run_id,time_index) key: {k}')
        keys.add(k)
        if not bool(getattr(snap, 'target_available', True)) or int(snap.frag_at_horizon) not in (0, 1):
            raise ValueError(f'{split}: invalid/unlabeled strict forecast target')
        if int(snap.future_time_index) - int(snap.time_index) != horizon:
            raise ValueError(f'{split}: inconsistent strict future horizon')
        if int(snap.frag_at_horizon) != int(float(snap.beta_target) > 1):
            raise ValueError(f'{split}: inconsistent risk event and beta target')
    return preds


def _groups(snaps: Sequence[object]) -> set[str]:
    return {str(s.split_group_id) for s in snaps}


def _validation_partition(snaps: Sequence[object], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic group-level separation: fit calibration vs select threshold."""
    groups = sorted(_groups(snaps), key=lambda g: hashlib.sha256(f'{seed}|calibration|{g}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('At least two independent validation split_group_id values required')
    first = max(1, min(len(groups) - 1, len(groups) // 2))
    fit_groups = set(groups[:first])
    is_fit = np.asarray([str(s.split_group_id) in fit_groups for s in snaps], dtype=bool)
    if is_fit.sum() == 0 or (~is_fit).sum() == 0:
        raise ValueError('Calibration and threshold selection need nonempty disjoint groups')
    return is_fit, ~is_fit



def split_early_stopping_validation(snaps: Sequence[object], seed: int) -> tuple[list[object], list[object]]:
    """Reserve independent validation groups for neural early stopping.

    The remaining groups are reserved for disjoint calibration and threshold
    selection; all three roles must have at least one group. The partition is
    deterministic and is performed before fitting the neural model.
    """
    groups = sorted(_groups(snaps), key=lambda g: hashlib.sha256(
        f'{seed}|neural-early-stop-v1|{g}'.encode('utf-8')).hexdigest())
    if len(groups) < 3:
        raise ValueError('Nested-role isolation requires >=3 validation groups')
    n_early = max(1, len(groups) // 3)
    early_groups = set(groups[:n_early])
    early = [s for s in snaps if str(s.split_group_id) in early_groups]
    remaining = [s for s in snaps if str(s.split_group_id) not in early_groups]
    if not early or not remaining or len(_groups(remaining)) < 2:
        raise ValueError('Early stopping/calibration/selection groups cannot be empty')
    return early, remaining


def select_probability_threshold(probs: Sequence[float], labels: Sequence[int]) -> dict:
    """Predeclared sample Risk-F1 selection, never touches test samples.

    Event-level alert window optimization is deliberately a separate later gate.
    """
    ps = np.asarray(probs, dtype=float)
    ys = np.asarray(labels, dtype=int)
    if ps.ndim != 1 or ys.shape != ps.shape or len(ps) == 0:
        raise ValueError('Threshold selection requires aligned nonempty vectors')
    if not np.isfinite(ps).all() or np.any((ps < 0) | (ps > 1)) or not np.isin(ys, [0, 1]).all():
        raise ValueError('Invalid threshold-selection probabilities or binary labels')
    rows = []
    for th in np.round(np.linspace(0.05, 0.95, 19), 2):
        pred = ps >= th
        tp = int(np.sum(pred & (ys == 1)))
        fp = int(np.sum(pred & (ys == 0)))
        fn = int(np.sum(~pred & (ys == 1)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append({'threshold':float(th),'risk_f1':float(f1),'fp':fp,'tp':tp,'fn':fn})
    chosen = sorted(rows, key=lambda x:(-x['risk_f1'],x['fp'],-x['threshold']))[0]
    return {'protocol':'validation_threshold_only_sample_risk_f1_v1',
            'selected_threshold':chosen['threshold'],
            'selection_metrics':chosen,
            'grid':[x['threshold'] for x in rows]}


def restrict_to_frozen_temporal_support(full_snapshots: Sequence[object],
                                        aligned_snapshots: Sequence[object],
                                        predictions: Sequence[float],
                                        risks: Sequence[float],
                                        *, window: int = 5):
    """Require exactly the common support implied by the longest temporal window.

    Never use post-hoc intersection of whatever predictions were available.
    Static models are restricted to the same predeclared keys; temporal models
    must already predict those keys. The full truth is retained separately.
    """
    if window < 1:
        raise ValueError('Temporal window must be >=1')
    from collections import defaultdict
    grouped=defaultdict(list)
    for snap in full_snapshots:
        grouped[str(snap.run_id)].append(snap)
    expected=set()
    for run,seq in grouped.items():
        order=sorted(seq, key=lambda s:int(s.time_index))
        if len({int(s.time_index) for s in order}) != len(order):
            raise ValueError(f'Duplicate time index in trajectory: {run}')
        expected.update((str(s.run_id),int(s.time_index)) for s in order[window-1:])
    if not expected:
        raise ValueError('No eligible common temporal support')
    beta=np.asarray(predictions,dtype=float)
    risk=np.asarray(risks,dtype=float)
    if beta.ndim != 1 or risk.ndim != 1 or len(beta) != len(aligned_snapshots) or len(risk) != len(beta):
        raise ValueError('Prediction-score shape mismatch')
    if not np.isfinite(beta).all() or not np.isfinite(risk).all():
        raise ValueError('Predictions contain NaN/inf')
    keys=[(str(s.run_id),int(s.time_index)) for s in aligned_snapshots]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate aligned prediction keys')
    found=set(keys)
    if expected - found:
        raise ValueError(f'Missing common temporal prediction support: {len(expected-found)} keys')
    if found - expected and found != {(str(s.run_id),int(s.time_index)) for s in full_snapshots}:
        raise ValueError('Unexpected partial model prediction support')
    mask=np.asarray([key in expected for key in keys],dtype=bool)
    if int(mask.sum()) != len(expected):
        raise ValueError('Temporal support count mismatch')
    return [s for s,ok in zip(aligned_snapshots,mask) if ok],beta[mask],risk[mask]

def _write_rows(path: Path, snaps: Sequence[object], beta_hat: np.ndarray, probabilities: np.ndarray | None,
                origin: str) -> None:
    if len(beta_hat) != len(snaps) or (probabilities is not None and len(probabilities) != len(snaps)):
        raise ValueError('row length mismatch')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for idx, snap in enumerate(snaps):
            score = '' if probabilities is None else float(probabilities[idx])
            writer.writerow(dict(
                run_id=str(snap.run_id), time_index=int(snap.time_index),
                split_group_id=str(snap.split_group_id), future_time_index=int(snap.future_time_index),
                frag_at_horizon=int(snap.frag_at_horizon), beta_current=float(snap.beta_current),
                beta_target=float(snap.beta_target), predicted_beta=float(beta_hat[idx]),
                risk_probability=score, score_origin=origin,
            ))


def export_torch_evidence(*, seed_dir: Path, seed: int, task_name: str,
                          model: object, train_cfg: dict, config_sha256: str,
                          horizon_steps: int, validation_snapshots: Sequence[object],
                          validation_beta_hat: Sequence[float], test_snapshots: Sequence[object],
                          test_beta_hat: Sequence[float],
                          early_stopping_group_ids: Sequence[str] | None = None) -> dict:
    """Save state_dict, disjoint-group validation calibration, and frozen test outputs.

    Training has already happened. This is a new evidence-generation protocol,
    not a retroactive repair of the R4 280/280 results.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    import sklearn

    if not is_neural_task(task_name):
        raise ValueError('Evidence export is limited to the frozen nine neural architectures')
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Real torch.nn.Module required; sklearn surrogate is rejected')
    if not config_sha256 or len(config_sha256) != 64:
        raise ValueError('A full SHA256 config signature is required')
    if int(horizon_steps) < 1:
        raise ValueError('horizon_steps must be positive')
    val_pred = _check_split(validation_snapshots, validation_beta_hat, int(horizon_steps), 'validation')
    test_pred = _check_split(test_snapshots, test_beta_hat, int(horizon_steps), 'test')
    if _groups(validation_snapshots) & _groups(test_snapshots):
        raise ValueError('Validation/test group leakage')
    early_groups = set(map(str, early_stopping_group_ids or []))
    if early_stopping_group_ids is not None:
        if not early_groups or early_groups & (_groups(validation_snapshots) | _groups(test_snapshots)):
            raise ValueError('Early stopping group leakage or empty group role')
    fit_mask, selection_mask = _validation_partition(validation_snapshots, int(seed))
    y_fit = np.asarray([int(s.frag_at_horizon) for s in validation_snapshots], dtype=int)[fit_mask]
    if set(y_fit.tolist()) != {0, 1}:
        raise ValueError('Calibration fit groups contain only one class; do not fabricate probabilities')
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=int(seed) % (2**32 - 1))
    clf.fit(val_pred[fit_mask].reshape(-1, 1), y_fit)
    selection_prob = np.asarray(clf.predict_proba(val_pred[selection_mask].reshape(-1, 1))[:, 1], dtype=float)
    test_prob = np.asarray(clf.predict_proba(test_pred.reshape(-1, 1))[:, 1], dtype=float)
    if any(not np.all(np.isfinite(q) & (q >= 0) & (q <= 1)) for q in (selection_prob, test_prob)):
        raise ValueError('Invalid calibrated probabilities')
    select_labels = np.asarray([int(s.frag_at_horizon) for s in validation_snapshots], dtype=int)[selection_mask]
    threshold_selection = select_probability_threshold(selection_prob, select_labels)
    out_dir = evidence_location(seed_dir, task_name)
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite R4 neural evidence: {out_dir}')
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.incomplete_', dir=out_dir.parent))
    try:
        # State dict only (rather than arbitrary pickle of whole Python module).
        weights = {str(key): value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(weights, temp / 'checkpoint_state_dict.pt')
        fit_snap = [s for s, selected in zip(validation_snapshots, fit_mask) if selected]
        select_snap = [s for s, selected in zip(validation_snapshots, selection_mask) if selected]
        _write_rows(temp / 'validation_calibration_fit.csv', fit_snap, val_pred[fit_mask], None,
                    'uncalibrated_beta_prediction_do_not_use_as_risk_probability')
        _write_rows(temp / 'validation_threshold_selection.csv', select_snap, val_pred[selection_mask],
                    selection_prob, 'calibrated_probability')
        _write_rows(temp / 'test.csv', test_snapshots, test_pred, test_prob, 'calibrated_probability')
        calibration = dict(method='validation_group_logistic_beta_to_event_v1',
                           fitted_on='validation_calibration_fit_only',
                           threshold_selected_on='validation_threshold_selection_only',
                           tested_on='held_out_test_only',
                           coefficient=float(clf.coef_[0][0]), intercept=float(clf.intercept_[0]),
                           classes=[int(n) for n in clf.classes_],
                           groups_fit=sorted(_groups(fit_snap)),
                           groups_threshold=sorted(_groups(select_snap)))
        (temp / 'calibration.json').write_text(json.dumps(calibration, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        (temp / 'threshold_selection.json').write_text(json.dumps(threshold_selection, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        manifest = dict(
            schema='fanet_r4_neural_evidence_v2', status='RAW_EVIDENCE_EXPORTED_NOT_SCIENTIFICALLY_REPLAYED',
            seed=int(seed), task_name=task_name, architecture=f'{model.__class__.__module__}.{model.__class__.__name__}',
            config_sha256=config_sha256, horizon_steps=int(horizon_steps),
            torch_version=str(torch.__version__), sklearn_version=str(sklearn.__version__),
            training_config=train_cfg,
            early_stopping_groups=sorted(early_groups),
            threshold_selection=threshold_selection,
            source_sha256={name:_sha256(Path(__file__).resolve().parent/name)
                           for name in ('r4_evidence.py','pipeline.py','training.py','models.py','dataset.py','evaluation.py')
                           if (Path(__file__).resolve().parent/name).is_file()},
            samples={
                'validation_calibration_fit':len(fit_snap), 'validation_threshold_selection':len(select_snap),
                'test':len(test_snapshots)},
            score_origin='calibrated_probability', calibration_role='disjoint_validation_groups',
            limitations=['Model architecture code required to load state_dict',
                         'Complete observation truth and end-to-end replay not included',
                         'No direct comparison to previous R4 metrics: protocol differs',
                         'Temporal support may differ; matching must be established before comparisons'],
            artifact_sha256={name:_sha256(temp/name) for name in (
                'checkpoint_state_dict.pt','validation_calibration_fit.csv',
                'validation_threshold_selection.csv','test.csv','calibration.json','threshold_selection.json')},
        )
        (temp / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        temp.rename(out_dir)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def verify_evidence(seed_dir: Path, task_name: str, expected_config_sha256: str | None = None, *, require_role_isolation: bool = False) -> bool:
    """Reject absent, tampered, or mismatched evidence during --resume."""
    path = evidence_location(seed_dir, task_name)
    try:
        data = json.loads((path/'MANIFEST.json').read_text(encoding='utf-8'))
        if data.get('schema') != 'fanet_r4_neural_evidence_v2' or data.get('task_name') != task_name:
            return False
        if expected_config_sha256 and data.get('config_sha256') != expected_config_sha256:
            return False
        files = data.get('artifact_sha256')
        required = {'checkpoint_state_dict.pt','validation_calibration_fit.csv',
                    'validation_threshold_selection.csv','test.csv','calibration.json','threshold_selection.json'}
        if set(files) != required or not all(_sha256(path/name)==digest for name,digest in files.items()):
            return False
        for label, filename in [('validation_calibration_fit','validation_calibration_fit.csv'),
                                ('validation_threshold_selection','validation_threshold_selection.csv'),
                                ('test','test.csv')]:
            with (path/filename).open('r', encoding='utf-8', newline='') as f:
                rows=list(csv.DictReader(f))
            if len(rows)!=int(data['samples'][label]) or not rows:
                return False
        groups = []
        for filename in ('validation_calibration_fit.csv','validation_threshold_selection.csv','test.csv'):
            with (path/filename).open('r', encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            groups.append({row['split_group_id'] for row in rows})
        early = set(data.get('early_stopping_groups', []))
        if require_role_isolation and not early:
            return False
        if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)):
            return False
        if any(early & g for g in groups):
            return False
        selection = json.loads((path/'threshold_selection.json').read_text(encoding='utf-8'))
        if data.get('threshold_selection') != selection:
            return False
        if not np.isfinite(float(selection['selected_threshold'])):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
