from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fanet.pipeline import _residual_diagnostic_row
from fanet.training import select_residual_branch_on_validation
from scripts.analyze_residual_branch import make_diagnostic_row, summarize_diagnostics


def test_alpha_selection_uses_validation_only_and_test_changes_cannot_change_alpha() -> None:
    validation = select_residual_branch_on_validation(
        np.asarray([2.0, 3.0]),
        np.asarray([1.0, 1.0]),
        [np.asarray([1.0, 2.0])],
    )
    row_a = make_diagnostic_row(
        seed=7,
        diagnostics={**validation, "selection_split": "validation", "selection_metric": "MAE"},
        y_test=np.asarray([2.0, 2.0]),
        persistence_prediction=np.asarray([1.0, 1.0]),
        selected_prediction=np.asarray([2.0, 2.0]),
    )
    row_b = make_diagnostic_row(
        seed=7,
        diagnostics={**validation, "selection_split": "validation", "selection_metric": "MAE"},
        y_test=np.asarray([50.0, 60.0]),
        persistence_prediction=np.asarray([1.0, 1.0]),
        selected_prediction=np.asarray([2.0, 2.0]),
    )

    assert validation["selected_alpha"] == pytest.approx(1.0)
    assert row_a["selected_alpha"] == row_b["selected_alpha"]
    assert row_a["test_mae_selected_alpha"] != row_b["test_mae_selected_alpha"]


def test_alpha_zero_is_retained_and_selected_when_residual_does_not_help() -> None:
    selection = select_residual_branch_on_validation(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        [np.asarray([10.0, -4.0, 2.0])],
    )

    assert 0.0 in selection["alpha_grid"]
    assert selection["selected_alpha"] == 0.0
    assert selection["alpha_zero"] is True
    assert selection["validation_mae_selected_alpha"] == selection["validation_mae_alpha_0"]


def test_persistence_identical_prediction_has_zero_difference() -> None:
    diagnostics = {
        "selected_alpha": 0.0,
        "validation_mae_alpha_0": 0.25,
        "validation_mae_selected_alpha": 0.25,
        "alpha_zero": True,
        "selection_split": "validation",
        "selection_metric": "MAE",
    }
    row = make_diagnostic_row(
        seed=7,
        diagnostics=diagnostics,
        y_test=np.asarray([1.0, 2.0]),
        persistence_prediction=np.asarray([1.0, 1.0]),
        selected_prediction=np.asarray([1.0, 1.0]),
    )

    assert row["test_mae_alpha_0"] == row["test_mae_selected_alpha"]
    assert row["paired_mae_difference_selected_minus_persistence"] == 0.0
    assert row["percentage_improvement"] == 0.0


def test_pipeline_exports_residual_diagnostics() -> None:
    model = SimpleNamespace(
        residual_diagnostics={
            "selected_alpha": 0.0,
            "validation_mae_alpha_0": 0.5,
            "validation_mae_selected_alpha": 0.5,
            "alpha_zero": True,
            "selection_split": "validation",
            "selection_metric": "MAE",
        }
    )
    result = SimpleNamespace(model_name="Kinetic-TopoGuard", model=model)
    snapshots = [SimpleNamespace(beta_target=2.0, beta_current=1.0)]

    row = _residual_diagnostic_row(17, result, snapshots, np.asarray([1.0]))

    assert row is not None
    assert row["alpha_zero"] is True
    assert row["test_mae_alpha_0"] == row["test_mae_selected_alpha"]


def test_all_zero_paired_differences_are_handled() -> None:
    rows = []
    for seed in (7, 17, 27):
        rows.append(
            {
                "seed": seed,
                "scope_type": "overall",
                "scope_value": "all",
                "selected_alpha": 0.0,
                "validation_mae_alpha_0": 0.2,
                "validation_mae_selected_alpha": 0.2,
                "test_mae_alpha_0": 0.3,
                "test_mae_selected_alpha": 0.3,
                "paired_mae_difference_selected_minus_persistence": 0.0,
                "alpha_zero": True,
            }
        )
    import pandas as pd

    summary = summarize_diagnostics(pd.DataFrame(rows), bootstrap_rounds=50)

    assert summary.iloc[0]["wilcoxon_pvalue"] == 1.0
    assert bool(summary.iloc[0]["count_branch_supported"]) is False
    assert summary.iloc[0]["unchanged_seed_count"] == 3
