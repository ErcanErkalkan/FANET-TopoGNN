from __future__ import annotations

from types import SimpleNamespace
import builtins

import numpy as np
import pandas as pd
import pytest

from scripts import run_explainability as explain


def test_run_block_permutation_is_deterministic_and_preserves_blocks() -> None:
    run_ids = np.asarray(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    first = explain.block_permutation_indices(run_ids, np.random.default_rng(41))
    second = explain.block_permutation_indices(run_ids, np.random.default_rng(41))
    assert np.array_equal(first, second)
    for start in (0, 3, 6):
        donor = first[start : start + 3]
        assert np.array_equal(donor, np.arange(donor[0], donor[0] + 3))
        assert len(set(run_ids[donor])) == 1


def test_group_permutation_does_not_mutate_model_input_or_predictions() -> None:
    matrix = np.arange(36, dtype=float).reshape(9, 4)
    original = matrix.copy()
    prediction_before = matrix.sum(axis=1)
    changed, _ = explain.permute_columns_by_run_block(
        matrix,
        np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3, dtype=object),
        np.asarray([1, 2]),
        np.random.default_rng(9),
    )
    assert np.array_equal(matrix, original)
    assert np.array_equal(matrix.sum(axis=1), prediction_before)
    assert not np.shares_memory(changed, matrix)


def test_grouped_permutation_row_count_and_determinism() -> None:
    aligned = []
    for run_id in ("a", "b"):
        for time_index in range(5):
            aligned.append(
                SimpleNamespace(
                    run_id=run_id,
                    time_index=time_index,
                    beta_current=1.0 if time_index < 4 else 2.0,
                    beta_target=1.0 if time_index < 4 else 2.0,
                )
            )
    count = np.asarray([item.beta_current for item in aligned], dtype=float)
    risk = np.asarray([0.8 if item.time_index == 2 else 0.1 for item in aligned], dtype=float)

    def predictor(group: str, rng: np.random.Generator):
        offset = 0.01 * (explain.GROUPS.index(group) + 1)
        return count.copy(), np.clip(risk + offset, 0.0, 1.0)

    kwargs = dict(
        seed=7,
        model_name="dummy",
        aligned=aligned,
        baseline_count=count,
        baseline_risk=risk,
        threshold=0.5,
        dt=0.1,
        horizon=3,
        repeats=2,
        permuted_predictor=predictor,
    )
    first = pd.DataFrame(explain.grouped_permutation_rows(**kwargs))
    second = pd.DataFrame(explain.grouped_permutation_rows(**kwargs))
    assert len(first) == len(explain.GROUPS) * len(explain.OUTCOMES)
    pd.testing.assert_frame_equal(first, second)


def test_kinetic_feature_groups_are_disjoint_and_exhaustive() -> None:
    layout = explain.kinetic_group_indices(256)
    flattened = np.concatenate(list(layout.values()))
    assert len(layout) == 9
    assert len(flattened) == len(set(flattened)) == 345
    assert set(flattened) == set(range(345))


def test_partition_run_id_leakage_is_rejected() -> None:
    train = [SimpleNamespace(run_id="train")]
    validation = [SimpleNamespace(run_id="validation")]
    test = [SimpleNamespace(run_id="test")]
    assert all(not values for values in explain.validate_partition_run_ids(train, validation, test).values())
    with pytest.raises(RuntimeError, match="run-level leakage"):
        explain.validate_partition_run_ids(train, validation, [SimpleNamespace(run_id="train")])


def test_missing_shap_is_recorded_without_failing_grouped_permutation(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("optional SHAP missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_shap)
    rows, status = explain.tree_shap_rows(
        7,
        SimpleNamespace(),
        np.zeros((2, 345)),
        SimpleNamespace(),
        {},
        2,
    )
    assert rows == []
    assert status["available"] is False
    assert status["status"] == "not_installed_or_import_failed"
