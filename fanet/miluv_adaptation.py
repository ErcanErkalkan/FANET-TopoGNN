from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

from .evaluation import alert_event_metrics, risk_probability_metrics
from .provenance import verify_manifest


@dataclass(frozen=True)
class ChronologicalSplit:
    calibration: np.ndarray
    guard: np.ndarray
    test: np.ndarray
    calibration_end_s: float
    test_start_s: float


def chronological_split(
    timestamps_s: Iterable[float],
    calibration_fraction: float = 0.30,
    guard_seconds: float = 2.0,
) -> ChronologicalSplit:
    """Split one ordered sequence without randomising temporally adjacent rows."""

    timestamps = np.asarray(list(timestamps_s), dtype=float)
    if timestamps.ndim != 1 or len(timestamps) < 3:
        raise ValueError("chronological splitting requires at least three timestamps")
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if float(guard_seconds) <= 0.0:
        raise ValueError("guard_seconds must be positive")

    cut = max(1, min(int(np.floor(len(timestamps) * calibration_fraction)), len(timestamps) - 2))
    calibration_end = float(timestamps[cut - 1])
    test_start = calibration_end + float(guard_seconds)
    calibration = np.flatnonzero(timestamps <= calibration_end)
    guard = np.flatnonzero((timestamps > calibration_end) & (timestamps < test_start))
    test = np.flatnonzero(timestamps >= test_start)
    if not len(guard):
        raise ValueError("guard interval contains no sampled observations")
    if not len(test):
        raise ValueError("guard interval leaves no independent test observations")
    return ChronologicalSplit(calibration, guard, test, calibration_end, test_start)


def assert_selection_indices(selection_indices: Iterable[int], split: ChronologicalSplit) -> None:
    selected = set(int(value) for value in selection_indices)
    allowed = set(split.calibration.astype(int).tolist())
    forbidden = set(split.test.astype(int).tolist())
    if selected - allowed:
        raise RuntimeError("adaptation/calibration attempted to use rows outside the calibration segment")
    if selected & forbidden:
        raise RuntimeError("test leakage detected during adaptation/calibration")


class IdentityCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


class SigmoidCalibrator:
    def __init__(self, seed: int):
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=int(seed))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        self.model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(scores, dtype=float), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(scores, dtype=float)), dtype=float)


def fit_probability_calibrator(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    fit_indices: Iterable[int],
    split: ChronologicalSplit,
    seed: int,
) -> Any:
    indices = np.asarray(list(fit_indices), dtype=int)
    assert_selection_indices(indices, split)
    x = np.asarray(scores, dtype=float)[indices]
    y = np.asarray(labels, dtype=int)[indices]
    if name == "none":
        return IdentityCalibrator().fit(x, y)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name} calibration requires both outcome classes")
    if name == "sigmoid":
        return SigmoidCalibrator(seed).fit(x, y)
    if name == "isotonic":
        if len(x) < 20:
            raise ValueError("isotonic calibration requires at least 20 calibration labels")
        return IsotonicCalibrator().fit(x, y)
    raise ValueError(f"unknown calibration method: {name}")


def select_calibration_and_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    snapshots: list[Any],
    split: ChronologicalSplit,
    methods: Iterable[str],
    threshold_grid: Iterable[float],
    dt: float,
    horizon_steps: int,
    seed: int,
) -> tuple[Any, float, dict[str, Any], list[dict[str, Any]]]:
    """Select calibration and threshold using calibration rows only."""

    indices = split.calibration
    aligned = [snapshots[int(index)] for index in indices]
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name in methods:
        try:
            calibrator = fit_probability_calibrator(name, scores, labels, indices, split, seed)
        except ValueError as exc:
            candidates.append({"calibration": name, "available": False, "reason": str(exc)})
            continue
        calibrated = np.clip(calibrator.predict(np.asarray(scores)[indices]), 0.0, 1.0)
        probability = risk_probability_metrics(np.asarray(labels)[indices], calibrated)
        ranked = []
        for threshold in threshold_grid:
            event = alert_event_metrics(aligned, calibrated, float(threshold), dt, horizon_steps)
            ranked.append((event["Alert_Event_F1"], -event["False_Alert_Events_per_minute"], float(threshold), event))
        chosen = max(ranked, key=lambda row: (row[0], row[1], row[2]))
        record = {
            "calibration": name,
            "available": True,
            "calibration_brier": float(probability["Risk_Brier"]),
            "calibration_ece": float(probability["Risk_ECE"]),
            "calibration_event_f1": float(chosen[3]["Alert_Event_F1"]),
            "calibration_false_events_per_minute": float(chosen[3]["False_Alert_Events_per_minute"]),
            "selected_threshold": float(chosen[2]),
            "fit_row_count": int(len(indices)),
            "fit_indices": indices.astype(int).tolist(),
        }
        candidates.append(record)
        fitted[name] = calibrator
    available = [row for row in candidates if row["available"]]
    if not available:
        raise RuntimeError("no probability calibration candidate is available")
    selected = min(
        available,
        key=lambda row: (
            row["calibration_brier"], row["calibration_ece"],
            -row["calibration_event_f1"], row["calibration"],
        ),
    )
    return fitted[selected["calibration"]], float(selected["selected_threshold"]), selected, candidates


def fit_few_shot_count_adapter(
    predictions: np.ndarray,
    targets: np.ndarray,
    fit_indices: Iterable[int],
    split: ChronologicalSplit,
    ridge_alpha: float = 1.0,
) -> Ridge:
    indices = np.asarray(list(fit_indices), dtype=int)
    assert_selection_indices(indices, split)
    return Ridge(alpha=float(ridge_alpha)).fit(
        np.asarray(predictions, dtype=float)[indices].reshape(-1, 1),
        np.asarray(targets, dtype=float)[indices],
    )


def verify_sequence_source_manifest(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    verification = verify_manifest(manifest, root)
    if not verification["valid"]:
        raise RuntimeError("MILUV sequence provenance failed: " + "; ".join(verification["errors"]))
    return verification
