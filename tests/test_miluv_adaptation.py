from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fanet.miluv_adaptation import (
    chronological_split,
    fit_probability_calibrator,
    verify_sequence_source_manifest,
)
from scripts.summarize_miluv_adaptation import paired_protocol_deltas


def test_chronological_split_has_guard_and_order() -> None:
    split = chronological_split(np.arange(100, dtype=float) * 0.1, 0.30, 1.0)
    assert split.calibration[-1] < split.guard[0] < split.test[0]
    assert split.test_start_s - split.calibration_end_s == pytest.approx(1.0)
    assert not (set(split.calibration) & set(split.test))


def test_calibration_rejects_test_rows() -> None:
    timestamps = np.arange(100, dtype=float) * 0.1
    split = chronological_split(timestamps, 0.30, 1.0)
    scores = np.linspace(0.0, 1.0, len(timestamps))
    labels = (scores > 0.5).astype(int)
    with pytest.raises(RuntimeError, match="outside the calibration"):
        fit_probability_calibrator("none", scores, labels, split.test, split, seed=7)


def test_calibration_uses_only_chronological_calibration_segment() -> None:
    timestamps = np.arange(100, dtype=float) * 0.1
    split = chronological_split(timestamps, 0.50, 1.0)
    scores = np.tile([0.1, 0.9], 50)
    labels = np.tile([0, 1], 50)
    calibrator = fit_probability_calibrator("sigmoid", scores, labels, split.calibration, split, seed=7)
    assert calibrator.predict(np.asarray([0.1, 0.9])).shape == (2,)


def test_sequence_hash_verification(tmp_path: Path) -> None:
    source = tmp_path / "sequence.csv"
    source.write_bytes(b"verified sequence\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"files": [{"relative_path": "sequence.csv", "bytes": source.stat().st_size, "sha256": digest}]}
    assert verify_sequence_source_manifest(manifest, tmp_path)["valid"]
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="provenance failed"):
        verify_sequence_source_manifest(manifest, tmp_path)


def test_paired_protocol_deltas_require_and_preserve_seed_pairing() -> None:
    rows = []
    for seed in (7, 17):
        for protocol, event_f1, brier in (
            ("zero_shot_frozen", 0.2, 0.30),
            ("chronological_calibration", 0.4, 0.20),
            ("few_shot_prediction_head", 0.1, 0.25),
        ):
            rows.append({"Seed": seed, "Sequence": "seq", "Model": "model", "Protocol": protocol, "primary_fpp_threshold": True, "Alert_Event_F1": event_f1, "False_Alert_Events_per_minute": 1.0, "MAE": 0.5, "Risk_Brier": brier, "Risk_ECE": 0.1})
    result = paired_protocol_deltas(pd.DataFrame(rows))
    chronological = next(row for row in result if row["protocol"] == "chronological_calibration")
    assert chronological["pair_count"] == 2
    assert chronological["event_f1_improvement_mean"] == pytest.approx(0.2)
    assert chronological["brier_improved_pairs"] == 2
