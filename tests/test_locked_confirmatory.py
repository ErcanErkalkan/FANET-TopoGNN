from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fanet.config import load_config
from fanet.seed_audit import deterministic_confirmatory_seeds
import scripts.run_eaai_locked_confirmatory as locked


ROOT = Path(__file__).resolve().parents[1]


def test_locked_seeds_are_deterministic_unique_and_exclude_every_prior_seed() -> None:
    lock = json.loads((ROOT / "configs" / "eaai_locked_confirmatory.lock.json").read_text())
    config = load_config(ROOT / "configs" / "eaai_locked_confirmatory.json")
    excluded = set(lock["excluded_prior_seeds"])
    selected = list(config.training["seed_list"])

    assert len(selected) == len(set(selected)) == 20
    assert not set(selected) & excluded
    assert selected == deterministic_confirmatory_seeds(excluded, 20)
    assert locked.DRY_RUN_SEED not in selected


def test_committed_lock_verifies_before_execution() -> None:
    verification = locked.verify_lock()

    assert verification["valid"] is True
    assert verification["errors"] == []


def test_config_tampering_is_rejected(monkeypatch, tmp_path: Path) -> None:
    resolved = load_config(ROOT / "configs" / "eaai_locked_confirmatory.json").raw
    resolved["sim"]["forecast_horizon_steps"] = 7
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(resolved), encoding="utf-8")
    monkeypatch.setattr(locked, "CONFIG_PATH", tampered)

    verification = locked.verify_lock()

    assert verification["valid"] is False
    assert any("config SHA-256 mismatch" in error for error in verification["errors"])
    assert any("resolved config SHA-256 mismatch" in error for error in verification["errors"])


def test_primary_paired_analysis_uses_seed_level_pairs() -> None:
    rows = []
    for seed in range(20):
        for model in (locked.PRIMARY_COMPARATOR, locked.PRIMARY_MODEL):
            row = {"seed": seed, "Model": model}
            for metric, higher_is_better in locked.METRICS.items():
                baseline = 0.5 if higher_is_better else 0.2
                row[metric] = baseline + (0.1 if model == locked.PRIMARY_MODEL and higher_is_better else 0.0)
            rows.append(row)
    tests = locked._paired_tests(
        pd.DataFrame(rows), bootstrap_rounds=500, permutation_rounds=2_000
    )
    primary = tests[tests["metric"] == "Alert_Event_F1"].iloc[0]

    assert primary["seed_pairs"] == 20
    assert primary["paired_mean_difference"] > 0.0
    assert primary["benefit_bootstrap_ci95_low"] > 0.0
