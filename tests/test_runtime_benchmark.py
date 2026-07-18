from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from fanet.runtime_benchmark import (
    STAGES,
    TOTAL_STAGE,
    summarize_latency_samples,
    validate_percentile_order,
    validate_stage_totals,
)


def _samples(timed_iterations: int = 4) -> pd.DataFrame:
    rows = []
    for phase, iterations in [("warmup", 2), ("timed", timed_iterations)]:
        for iteration in range(iterations):
            values = [float(index + iteration + 1) / 100.0 for index in range(len(STAGES))]
            for stage, value in zip(STAGES, values):
                rows.append(
                    {
                        "model": "model",
                        "thread_mode": "single",
                        "n_nodes": 10,
                        "phase": phase,
                        "iteration": iteration,
                        "stage": stage,
                        "latency_ms": value,
                    }
                )
            rows.append(
                {
                    "model": "model",
                    "thread_mode": "single",
                    "n_nodes": 10,
                    "phase": phase,
                    "iteration": iteration,
                    "stage": TOTAL_STAGE,
                    "latency_ms": sum(values),
                }
            )
    return pd.DataFrame(rows)


def test_stage_total_is_sum_of_exclusive_stages() -> None:
    samples = _samples()
    validate_stage_totals(samples, atol_ms=1e-12)
    bad = samples.copy()
    index = bad.index[bad["stage"].eq(TOTAL_STAGE)][0]
    bad.loc[index, "latency_ms"] += 1.0
    try:
        validate_stage_totals(bad, atol_ms=1e-12)
    except ValueError as error:
        assert "stage sum mismatch" in str(error)
    else:
        raise AssertionError("invalid total must be rejected")


def test_summary_excludes_warmup_and_preserves_sample_count() -> None:
    summary = summarize_latency_samples(_samples(timed_iterations=7))
    assert summary["sample_count"].eq(7).all()
    assert len(summary) == len(STAGES) + 1


def test_percentile_ordering_is_validated() -> None:
    summary = summarize_latency_samples(_samples())
    validate_percentile_order(summary)
    summary.loc[0, "p90_ms"] = summary.loc[0, "p50_ms"] - 1.0
    try:
        validate_percentile_order(summary)
    except ValueError as error:
        assert "monotonically ordered" in str(error)
    else:
        raise AssertionError("decreasing percentiles must be rejected")


def test_joblib_reload_prediction_consistency(tmp_path: Path) -> None:
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = x[:, 0] * 0.25 - x[:, 2]
    model = ExtraTreesRegressor(n_estimators=8, random_state=7, n_jobs=1).fit(x, y)
    expected = model.predict(x)
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    assert np.array_equal(expected, restored.predict(x))
