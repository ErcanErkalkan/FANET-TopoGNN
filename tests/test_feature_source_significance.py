from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.analyze_feature_source_significance as significance


def _decision(candidate: np.ndarray, reference: np.ndarray, higher_is_better: bool = True) -> dict:
    result = significance.analyze_paired_arrays(
        candidate,
        reference,
        higher_is_better=higher_is_better,
        bootstrap_rounds=2_000,
        permutation_rounds=5_000,
        random_seed=17,
    )
    return significance.classify_result(result, float(result["paired_permutation_pvalue"]))


def test_synthetic_superiority_is_supported() -> None:
    reference = np.linspace(0.1, 0.3, 12)
    candidate = reference + np.linspace(0.08, 0.12, 12)

    decision = _decision(candidate, reference)

    assert decision["statistically_supported_superiority"] is True
    assert decision["decision"] == "statistically_supported_superiority"


def test_synthetic_equality_is_not_called_equivalent_or_superior() -> None:
    reference = np.linspace(0.1, 0.3, 12)

    decision = _decision(reference.copy(), reference)

    assert decision["no_supported_superiority"] is True
    assert decision["decision"] == "no_supported_superiority"
    assert "equivalent" not in decision["decision"]


def test_synthetic_degradation_is_supported() -> None:
    reference = np.linspace(0.1, 0.3, 12)
    candidate = reference - np.linspace(0.08, 0.12, 12)

    decision = _decision(candidate, reference)

    assert decision["statistically_supported_degradation"] is True
    assert decision["decision"] == "statistically_supported_degradation"


def test_lower_is_better_direction_is_respected() -> None:
    reference = np.linspace(1.0, 1.2, 12)
    candidate = reference - np.linspace(0.08, 0.12, 12)

    decision = _decision(candidate, reference, higher_is_better=False)

    assert decision["statistically_supported_superiority"] is True


def _pairing_frame() -> pd.DataFrame:
    rows = []
    for combination in sorted(significance.EXPECTED_COMBINATIONS):
        model = (
            "Current-state ExtraTrees"
            if combination == significance.REFERENCE
            else f"ExtraTrees ({combination})"
        )
        for seed in (7, 17, 27):
            row = {"seed": seed, "feature_sources": combination, "Model": model}
            for metric in significance.METRICS:
                row[metric] = float(seed)
            rows.append(row)
    return pd.DataFrame(rows)


def test_duplicate_seed_stops_analysis() -> None:
    frame = _pairing_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate seed"):
        significance.validate_pairing(frame)


def test_missing_seed_stops_analysis() -> None:
    frame = _pairing_frame()
    frame = frame.loc[
        ~((frame["feature_sources"] == "graph") & (frame["seed"] == 27))
    ]

    with pytest.raises(ValueError, match="seed pairing mismatch"):
        significance.validate_pairing(frame)


def test_nan_pairs_are_reported_invalid() -> None:
    result = significance.analyze_paired_arrays(
        np.asarray([1.0, np.nan, 2.0]),
        np.asarray([0.0, 0.0, 0.0]),
        higher_is_better=True,
    )

    assert result["analysis_status"] == "invalid_missing_metric_values"
    assert result["n_valid_pairs"] == 2


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    adjusted = significance.holm_adjust([0.001, 0.02, 0.5])

    assert adjusted == pytest.approx([0.003, 0.04, 0.5])
