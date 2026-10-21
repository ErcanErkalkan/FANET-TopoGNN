"""Regression tests for prospective, right-censored full-horizon labels."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fanet.config import load_config
from fanet.dataset import (
    build_dataset, relabel_forecast_horizon, split_by_run,
    validate_full_horizon_labels,
)


ROOT = Path(__file__).resolve().parents[1]


def _sim() -> dict:
    return dict(load_config(ROOT / "configs/smoke_30s.json").sim)


def test_strict_excludes_censored_tail_and_matches_raw_future() -> None:
    legacy = build_dataset(_sim(), seed=7)
    strict = build_dataset({**_sim(), "horizon_label_policy": "strict_full_horizon"}, seed=7)
    horizon = 2
    legacy_runs = split_by_run(legacy)
    strict_runs = split_by_run(strict)
    assert set(legacy_runs) == set(strict_runs)
    for run_id, original in legacy_runs.items():
        clean = strict_runs[run_id]
        by_time = {s.time_index: s for s in original}
        assert len(original) == 12
        assert len(clean) == 10
        assert [s.time_index for s in clean] == list(range(10))
        for snap in clean:
            actual = by_time[snap.time_index + horizon]
            assert snap.future_time_index - snap.time_index == horizon
            assert snap.beta_target == actual.beta_current
            assert snap.frag_at_horizon == int(actual.beta_current > 1.0)
    assert validate_full_horizon_labels(strict, horizon)["rows_checked"] == 30


def test_strict_relabel_uses_complete_base_and_exact_time() -> None:
    legacy = build_dataset(_sim(), seed=7)
    for horizon in (1, 2, 5):
        relabeled = relabel_forecast_horizon(legacy, horizon, strict_full_horizon=True)
        assert len(relabeled) == 3 * (12 - horizon)
        assert validate_full_horizon_labels(relabeled, horizon)["rows_checked"] == len(relabeled)
    assert len(relabel_forecast_horizon(legacy, 5)) == len(legacy)


def test_invalid_strict_labels_and_policy_fail_closed() -> None:
    strict = build_dataset({**_sim(), "horizon_label_policy": "strict_full_horizon"}, seed=7)
    with pytest.raises(ValueError, match="Clamped or misaligned"):
        validate_full_horizon_labels([replace(strict[0], future_time_index=strict[0].time_index)], 2)
    with pytest.raises(ValueError, match="Fragmentation label"):
        validate_full_horizon_labels([replace(strict[0], frag_at_horizon=1 - strict[0].frag_at_horizon)], 2)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_full_horizon_labels([strict[0], strict[0]], 2)
    with pytest.raises(ValueError, match="Unknown horizon_label_policy"):
        build_dataset({**_sim(), "horizon_label_policy": "typo"}, seed=7)
    with pytest.raises(ValueError, match="requires horizon_steps > 0"):
        build_dataset({**_sim(), "horizon_label_policy": "strict_full_horizon", "forecast_horizon_steps": 0}, seed=7)
