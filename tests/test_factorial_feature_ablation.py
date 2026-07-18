from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_factorial_feature_ablation as factorial
from fanet.provenance import sha256_file


def test_all_eight_feature_combinations_are_generated() -> None:
    combinations = factorial.feature_combinations()

    assert len(combinations) == 8
    assert {factorial.feature_label(item) for item in combinations} == {
        "current-only",
        "graph",
        "topology",
        "kinematic",
        "graph+topology",
        "graph+kinematic",
        "topology+kinematic",
        "graph+topology+kinematic",
    }


def test_current_only_has_distinct_extratrees_model_name() -> None:
    assert factorial.model_name(()) == "Current-state ExtraTrees"
    assert "persistence" not in factorial.model_name(()).lower()
    assert factorial.model_name(("graph",)) != factorial.model_name(())


def test_run_id_split_leakage_is_rejected() -> None:
    train = [SimpleNamespace(run_id="train")]
    validation = [SimpleNamespace(run_id="validation")]
    test = [SimpleNamespace(run_id="train")]

    with pytest.raises(RuntimeError, match="run_id leakage"):
        factorial.assert_disjoint_run_ids(train, validation, test)


def test_threshold_selection_uses_validation_snapshots_only(monkeypatch: pytest.MonkeyPatch) -> None:
    validation_snapshots = [SimpleNamespace(run_id="validation")]
    test_snapshots = [SimpleNamespace(run_id="test")]
    observed_snapshots: list[object] = []

    def fake_event_metrics(snapshots, scores, threshold, dt, horizon_steps):
        observed_snapshots.extend(snapshots)
        return {
            "Alert_Event_F1": 1.0 if threshold == 0.4 else 0.5,
            "False_Alert_Events_per_minute": 0.0,
        }

    monkeypatch.setattr(factorial, "alert_event_metrics", fake_event_metrics)
    selected = factorial.select_threshold_on_validation(
        labels=np.asarray([1]),
        scores=np.asarray([0.8]),
        snapshots=validation_snapshots,
        dt=0.1,
        horizon_steps=6,
        threshold_grid=(0.3, 0.4),
    )

    assert selected == 0.4
    assert observed_snapshots
    assert all(snapshot in validation_snapshots for snapshot in observed_snapshots)
    assert all(snapshot not in test_snapshots for snapshot in observed_snapshots)


def test_resume_skips_completed_seed_without_invoking_runner(tmp_path: Path) -> None:
    combinations = factorial.feature_combinations()
    seed = 7
    seed_dir = tmp_path / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "seed": [seed] * len(combinations),
            "feature_sources": [factorial.feature_label(item) for item in combinations],
        }
    )
    csv_path = seed_dir / factorial.PER_SEED_FILENAME
    frame.to_csv(csv_path, index=False)
    (seed_dir / factorial.SUCCESS_FILENAME).write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": len(combinations),
                "csv_sha256": sha256_file(csv_path),
            }
        ),
        encoding="utf-8",
    )

    pending, skipped, archived = factorial.prepare_seed_schedule(
        [seed], tmp_path, combinations, resume=True
    )

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("completed seed was rerun")

    completed = factorial.execute_seeds(
        pending,
        sim={},
        combinations=combinations,
        output_dir=tmp_path,
        workers=1,
        runner=forbidden_runner,
    )
    assert pending == []
    assert skipped == [seed]
    assert archived == []
    assert completed == []


def test_summary_matches_per_seed_means_and_seed_count() -> None:
    rows = []
    for seed, mae in [(7, 1.0), (17, 3.0)]:
        row = {
            "seed": seed,
            "Model": "Current-state ExtraTrees",
            "feature_sources": "current-only",
        }
        for metric in factorial.METRIC_COLUMNS:
            row[metric] = mae
        rows.append(row)
    per_seed = pd.DataFrame(rows)

    summary = factorial.summarise_results(per_seed, bootstrap_rounds=50)

    assert int(summary.loc[0, "seeds"]) == 2
    assert float(summary.loc[0, "MAE_mean"]) == 2.0
    assert "MAE_seed_bootstrap_ci95_low" in summary.columns


def test_submission_config_contains_twenty_unique_default_seeds() -> None:
    config = json.loads(factorial.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    seeds = config["training"]["seed_list"]

    assert len(seeds) == 20
    assert len(set(seeds)) == 20
