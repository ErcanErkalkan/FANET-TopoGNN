"""Preflight for an independent, leakage-controlled R4 re-evaluation.

This module does not re-use the old R4 scores or claim new R4 measurements.
The inputs must be freshly exported risk probabilities and matched snapshot keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .dataset import Snapshot, require_forecast_eligible
from .evaluation import alert_event_metrics_full_trajectory


@dataclass(frozen=True)
class ModelSplitScores:
    """Out-of-sample continuous risk scores with explicit provenance."""

    snapshots: list[Snapshot]
    risk_probabilities: np.ndarray
    score_origin: str  # 'native_probability' or 'calibrated_probability'


def _checked_scores(record: ModelSplitScores, horizon_steps: int) -> dict[tuple[str, int], tuple[float, int, float]]:
    if record.score_origin not in {'native_probability', 'calibrated_probability'}:
        raise ValueError('R4 requires genuine risk probabilities; binary count > 1 proxy is not accepted')
    require_forecast_eligible(record.snapshots)
    values = np.asarray(record.risk_probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(record.snapshots) or not np.isfinite(values).all():
        raise ValueError('scores must be finite 1-D and align 1:1 with snapshot keys')
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError('risk probabilities must be within [0,1]')
    output = {}
    for snap, score in zip(record.snapshots, values):
        if snap.future_time_index - snap.time_index != horizon_steps:
            raise ValueError('inconsistent forecast horizon')
        key = (snap.run_id, snap.time_index)
        if key in output:
            raise ValueError(f'duplicate prediction key: {key}')
        output[key] = float(score), int(snap.frag_at_horizon), float(snap.beta_current)
    return output


def validate_common_support(
    by_model: Mapping[str, ModelSplitScores], horizon_steps: int
) -> tuple[tuple[str, int], ...]:
    """Reject missing temporal predictions instead of silently shrinking test sets."""
    if len(by_model) < 2:
        raise ValueError('at least two models must be compared')
    checked = {name: _checked_scores(row, horizon_steps) for name, row in by_model.items()}
    reference = next(iter(checked.values()))
    if not reference:
        raise ValueError('empty common decision set')
    for name, actual in checked.items():
        if actual.keys() != reference.keys():
            raise ValueError(f'{name}: prediction support differs; no silent intersection allowed')
        for key in reference:
            _, target, beta = actual[key]
            _, expected_target, expected_beta = reference[key]
            if target != expected_target or not np.isclose(beta, expected_beta):
                raise ValueError(f'{name}: inconsistent target/topology truth for {key}')
    return tuple(sorted(reference.keys()))


def select_validation_operating_point(
    validation: ModelSplitScores,
    complete_validation_truth: list[Snapshot],
    dt: float,
    horizon_steps: int,
    false_alert_budget_per_minute: float = float('inf'),
    thresholds: tuple[float, ...] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2)),
) -> dict[str, float | bool]:
    """Threshold is selected on validation data only; never accept test data."""
    _checked_scores(validation, horizon_steps)
    if not thresholds:
        raise ValueError('threshold search cannot be empty')
    if false_alert_budget_per_minute < 0:
        raise ValueError('false alert budget must be nonnegative')
    candidates = []
    for threshold in thresholds:
        metrics = alert_event_metrics_full_trajectory(
            validation.snapshots, validation.risk_probabilities, complete_validation_truth,
            float(threshold), dt, horizon_steps,
        )
        candidates.append((float(threshold), metrics))
    valid = [entry for entry in candidates
             if entry[1]['False_Alert_Events_per_minute'] <= false_alert_budget_per_minute]
    if valid:
        selected = sorted(valid, key=lambda entry: (
            -entry[1]['Alert_Event_F1'],
            entry[1]['False_Alert_Events_per_minute'],
            -entry[1]['Alert_Event_Recall'],
            entry[0],
        ))[0]
    else:
        selected = sorted(candidates, key=lambda entry: (
            entry[1]['False_Alert_Events_per_minute'],
            -entry[1]['Alert_Event_F1'],
            entry[0],
        ))[0]
    return {
        'Selected_Threshold': selected[0],
        'Validation_Constraint_Met': bool(valid),
        **{f'Validation_{k}': float(v) for k, v in selected[1].items()},
    }


def compare_r4_models(
    validation: Mapping[str, ModelSplitScores],
    test: Mapping[str, ModelSplitScores],
    validation_full_truth: list[Snapshot],
    test_full_truth: list[Snapshot],
    *,
    dt: float,
    horizon_steps: int,
    false_alert_budget_per_minute: float = float('inf'),
) -> list[dict[str, float | bool | str]]:
    """Strict matched-support comparison; returns descriptive rows, not p-values."""
    if set(validation) != set(test):
        raise ValueError('validation and test must contain identical model names')
    validate_common_support(validation, horizon_steps)
    validate_common_support(test, horizon_steps)
    val_groups = {s.split_group_id for record in validation.values() for s in record.snapshots}
    test_groups = {s.split_group_id for record in test.values() for s in record.snapshots}
    if val_groups & test_groups:
        raise ValueError('validation/test split_group_id leakage')
    results = []
    for name in sorted(validation):
        operating = select_validation_operating_point(
            validation[name], validation_full_truth, dt, horizon_steps,
            false_alert_budget_per_minute,
        )
        test_metric = alert_event_metrics_full_trajectory(
            test[name].snapshots, test[name].risk_probabilities, test_full_truth,
            float(operating['Selected_Threshold']), dt, horizon_steps,
        )
        results.append({
            'Model': name,
            'Score_Origin': test[name].score_origin,
            'Validation_Decision_Count': float(len(validation[name].snapshots)),
            'Test_Decision_Count': float(len(test[name].snapshots)),
            **operating,
            **{f'Test_{k}': float(v) for k, v in test_metric.items()},
        })
    return results
