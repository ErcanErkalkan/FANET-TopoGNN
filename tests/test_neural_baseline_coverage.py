from __future__ import annotations

import pandas as pd
import pytest

from fanet.config import load_config
from fanet.pipeline import _backend_metadata, _training_tasks
from scripts import build_neural_seed_extension as builder


def test_expected_baseline_inventory_has_fourteen_models() -> None:
    assert len(builder.EXPECTED_MODELS) == 14
    assert len(set(builder.EXPECTED_MODELS)) == 14


def test_full_rerun_config_schedules_all_fourteen_real_models() -> None:
    config = load_config(builder.ROOT / "configs/publication_neural_full_5seed.json")
    tasks = _training_tasks(config)
    assert len(tasks) == 14
    assert "current_state_extratrees" in tasks
    metadata = _backend_metadata(tasks)
    assert metadata["model_backend"]["current_state_extratrees"] == "scikit_learn"
    assert all(
        metadata["model_backend"][task] == "pytorch"
        for task in tasks
        if task in {"GCN", "GAT", "GraphSAGE", "PI+MLP", "FANET-TopoGNN", "FANET-TopoGNN (concat)", "tgcn:5", "stgcn:5", "tgn:5"}
    )


def test_executed_artifacts_cover_every_model_on_five_shared_seeds() -> None:
    frame, failures = builder.collect_complete_metrics()
    builder.validate_coverage(frame, failures)
    counts = frame.groupby("Model")["seed"].nunique()
    assert set(counts.index) == set(builder.EXPECTED_MODELS)
    assert (counts == 5).all()


def test_surrogate_backend_is_rejected() -> None:
    frame, failures = builder.collect_complete_metrics()
    index = frame.index[frame["Model"] == "GCN"][0]
    frame.loc[index, "Model_Backend"] = "scikit_learn_surrogate"
    with pytest.raises(RuntimeError, match="surrogate/non-PyTorch"):
        builder.validate_coverage(frame, failures)


def test_incomplete_seed_coverage_is_rejected() -> None:
    frame, failures = builder.collect_complete_metrics()
    frame = frame[~((frame["Model"] == "GAT") & (frame["seed"] == 47))]
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        builder.validate_coverage(frame, failures)


def test_neural_parameter_counts_are_exact_positive_counts() -> None:
    counts = builder._torch_parameter_counts()
    assert set(counts) == set(builder.NEURAL_MODELS)
    assert all(total > 0 and trainable == total for total, trainable in counts.values())


def test_aggregate_keeps_all_models_and_event_metrics() -> None:
    frame, failures = builder.collect_complete_metrics()
    builder.validate_coverage(frame, failures)
    aggregate = builder.aggregate_metrics(frame)
    assert aggregate["Model"].tolist() == list(builder.EXPECTED_MODELS)
    assert (aggregate["seed_count"] == 5).all()
    assert {
        "Alert_Event_Precision_mean",
        "Alert_Event_Recall_mean",
        "Alert_Event_F1_mean",
        "False_Alert_Events_per_minute_mean",
    }.issubset(aggregate.columns)
