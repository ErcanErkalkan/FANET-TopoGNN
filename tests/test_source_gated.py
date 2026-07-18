from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fanet.config import load_config
from fanet.dataset import build_dataset, train_val_test_split
from fanet.source_gated import (
    FEATURE_GROUPS,
    SourceGatedKineticTopoGuard,
    build_source_matrices,
    grouped_oof_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_oof_inputs():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.asarray(["g0", "g1", "g2", "g3"], dtype=object), 8)
    matrices = {name: rng.normal(size=(len(groups), 3)) for name in FEATURE_GROUPS}
    y_count = 1.0 + (matrices["current"][:, 0] > 0.0).astype(float)
    y_risk = (matrices["graph"][:, 0] > 0.0).astype(int)
    aligned = [SimpleNamespace(beta_current=1.0, run_id=f"{group}_run") for group in groups]
    return matrices, y_count, y_risk, groups, aligned


def test_every_oof_row_is_predicted_by_a_model_that_did_not_train_on_it() -> None:
    matrices, y_count, y_risk, groups, aligned = _synthetic_oof_inputs()
    count, risk, folds = grouped_oof_predictions(
        matrices,
        y_count,
        y_risk,
        groups,
        min_samples_leaf=1,
        n_estimators=4,
        n_splits=4,
        seed=7,
        aligned=aligned,
    )

    seen = []
    for fold in folds:
        assert not set(fold["train_row_indices"]) & set(fold["oof_row_indices"])
        seen.extend(fold["oof_row_indices"])
    assert sorted(seen) == list(range(len(groups)))
    assert np.isfinite(count).all()
    assert np.isfinite(risk).all()


def test_split_group_never_appears_in_train_and_oof_sides_of_same_fold() -> None:
    matrices, y_count, y_risk, groups, aligned = _synthetic_oof_inputs()
    _, _, folds = grouped_oof_predictions(
        matrices,
        y_count,
        y_risk,
        groups,
        min_samples_leaf=1,
        n_estimators=4,
        n_splits=4,
        seed=7,
        aligned=aligned,
    )

    for fold in folds:
        assert not set(fold["train_split_group_ids"]) & set(fold["oof_split_group_ids"])
        assert not set(fold["train_run_ids"]) & set(fold["oof_run_ids"])


def test_test_partition_cannot_be_passed_to_model_fit_or_selection() -> None:
    parameters = list(inspect.signature(SourceGatedKineticTopoGuard.fit).parameters)

    assert parameters == ["self", "train_data", "validation_data"]


def test_extended_config_enables_source_gated_without_modifying_parent() -> None:
    development = load_config(ROOT / "configs" / "source_gated_development.json")
    parent = load_config(ROOT / "configs" / "paper_like_submission.json")

    assert "source_gated_kinetic_topoguard" in development.training["selected_models"]
    assert "source_gated_kinetic_topoguard" not in parent.training["selected_models"]
    assert development.sim == parent.sim


def test_serialization_round_trip_preserves_predictions(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs" / "paper_like_submission.json").read_text(encoding="utf-8"))
    sim = deepcopy(raw["sim"])
    sim["time_steps"] = 10
    sim["swarm_sizes"] = [10]
    snapshots = build_dataset(sim, seed=7)
    train, validation, test = train_val_test_split(
        snapshots,
        split_seed=int(sim["split_seed"]),
        stratify_by=tuple(sim.get("split_stratify_by", ["mobility"])),
    )
    model = SourceGatedKineticTopoGuard(
        horizon_steps=int(sim["forecast_horizon_steps"]),
        dt=float(sim["dt"]),
        seed=7,
        parameters={
            "n_splits": 2,
            "n_estimators": 4,
            "frozen_count_min_samples_leaf": 2,
            "frozen_risk_min_samples_leaf": 5,
            "frozen_ridge_alpha": 1.0,
            "frozen_logistic_c": 0.1,
            "frozen_calibration_type": "none",
            "threshold_grid": [0.3, 0.5, 0.7],
        },
    ).fit(train, validation)
    before = model.predict_snapshots(test)
    artifact = tmp_path / "source_gated.pkl"
    model.save(artifact)
    restored = SourceGatedKineticTopoGuard.load(artifact)
    after = restored.predict_snapshots(test)

    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])
    assert [snapshot.run_id for snapshot in before[2]] == [snapshot.run_id for snapshot in after[2]]
    metadata = restored.artifact_metadata()
    assert metadata["feature_group_names"] == list(FEATURE_GROUPS)
    assert metadata["training_folds"]
    assert metadata["selected_parameters"]["count_min_samples_leaf"] == 2
    assert metadata["selected_parameters"]["risk_min_samples_leaf"] == 5
    assert metadata["selected_parameters"]["hyperparameters_frozen_before_confirmatory_test"] is True
