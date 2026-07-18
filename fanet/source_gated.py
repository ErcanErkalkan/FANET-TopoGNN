from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from .dataset import Snapshot
from .evaluation import alert_event_metrics, classification_metrics, risk_probability_metrics
from .geometry import pairwise_distances
from .graph_utils import adjacency_from_radius, betti_zero
from .training import TrainResult


MODEL_NAME = "Source-Gated Kinetic-TopoGuard"
FEATURE_GROUPS = ("current", "graph", "topology", "kinematic")
DEFAULT_PARAMETERS = {
    "n_splits": 5,
    "n_estimators": 64,
    "min_samples_leaf_grid": [2, 5],
    "ridge_alpha_grid": [0.1, 1.0],
    "logistic_c_grid": [0.1, 1.0],
    "calibration_options": ["none", "sigmoid", "isotonic"],
    "threshold_grid": [round(value, 2) for value in np.linspace(0.05, 0.95, 19)],
}


def _summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [
            float(values.mean()) if values.size else 0.0,
            float(values.std()) if values.size else 0.0,
            float(np.quantile(values, 0.05)) if values.size else 0.0,
            float(np.quantile(values, 0.50)) if values.size else 0.0,
            float(np.quantile(values, 0.95)) if values.size else 0.0,
        ],
        dtype=np.float32,
    )


def source_feature_groups(
    snapshot: Snapshot,
    previous: Snapshot | None,
    horizon_steps: int,
    dt: float,
) -> dict[str, np.ndarray]:
    """Return non-overlapping source features; only ``current`` contains beta state."""
    current = np.asarray(
        [
            snapshot.beta_current,
            previous.beta_current if previous is not None else snapshot.beta_current,
            snapshot.beta_current - previous.beta_current if previous is not None else 0.0,
            snapshot.n_nodes / 30.0,
            snapshot.radius / 1000.0,
        ],
        dtype=np.float32,
    )
    graph = np.asarray(
        [
            snapshot.stats[4] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[5] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[6] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[7],
            snapshot.stats[8],
            snapshot.stats[9] / max(snapshot.n_nodes - 1, 1),
            snapshot.stats[16],
        ],
        dtype=np.float32,
    )
    topology = np.concatenate([snapshot.pi, _summary(snapshot.pi)]).astype(np.float32)
    tau = float(horizon_steps) * float(dt)
    positions = snapshot.positions.astype(float)
    current_distances = pairwise_distances(positions)
    projected_distances = pairwise_distances(positions + snapshot.velocities.astype(float) * tau)
    triangle = np.triu_indices(snapshot.n_nodes, k=1)
    radius = max(float(snapshot.radius), 1e-6)
    current_pairs = current_distances[triangle] / radius
    projected_pairs = projected_distances[triangle] / radius
    delta = projected_pairs - current_pairs
    projected_adjacency = adjacency_from_radius(projected_distances, radius)
    speeds = np.linalg.norm(snapshot.velocities.astype(float), axis=1) / 30.0
    kinematic = np.concatenate(
        [
            _summary(current_pairs),
            _summary(projected_pairs),
            _summary(delta),
            _summary(speeds),
            np.asarray(
                [
                    float(betti_zero(projected_adjacency)),
                    float(np.mean((current_pairs <= 1.0) & (projected_pairs > 1.0))),
                    float(np.mean((current_pairs > 1.0) & (projected_pairs <= 1.0))),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return {
        "current": np.nan_to_num(current),
        "graph": np.nan_to_num(graph),
        "topology": np.nan_to_num(topology),
        "kinematic": np.nan_to_num(kinematic),
    }


def build_source_matrices(
    snapshots: list[Snapshot], horizon_steps: int, dt: float
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[Snapshot]]:
    rows: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_GROUPS}
    aligned: list[Snapshot] = []
    grouped: dict[str, list[Snapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.run_id, []).append(snapshot)
    for run_id in sorted(grouped):
        previous = None
        for snapshot in sorted(grouped[run_id], key=lambda item: item.time_index):
            features = source_feature_groups(snapshot, previous, horizon_steps, dt)
            for name in FEATURE_GROUPS:
                rows[name].append(features[name])
            aligned.append(snapshot)
            previous = snapshot
    return (
        {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()},
        np.asarray([snapshot.beta_target for snapshot in aligned], dtype=float),
        np.asarray([snapshot.frag_at_horizon for snapshot in aligned], dtype=int),
        np.asarray([snapshot.split_group_id for snapshot in aligned], dtype=object),
        aligned,
    )


def _positive_probability(classifier: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    classes = np.asarray(classifier.classes_)
    positive = np.flatnonzero(classes == 1)
    if positive.size:
        return np.asarray(probabilities[:, int(positive[0])], dtype=float)
    return np.full(len(features), float(classes[0] == 1), dtype=float)


def grouped_oof_predictions(
    matrices: dict[str, np.ndarray],
    y_count: np.ndarray,
    y_risk: np.ndarray,
    groups: np.ndarray,
    *,
    min_samples_leaf: int,
    n_estimators: int,
    n_splits: int,
    seed: int,
    aligned: list[Snapshot] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    unique_groups = np.unique(groups)
    folds = min(int(n_splits), len(unique_groups))
    if folds < 2:
        raise ValueError("at least two distinct split_group_id values are required for cross-fitting")
    splitter = GroupKFold(n_splits=folds)
    count_oof = np.full((len(y_count), len(FEATURE_GROUPS)), np.nan, dtype=float)
    risk_oof = np.full_like(count_oof, np.nan)
    fold_records: list[dict[str, Any]] = []
    base = np.asarray([snapshot.beta_current for snapshot in aligned], dtype=float) if aligned else np.ones(len(y_count))
    residual_target = np.asarray(y_count, dtype=float) - base
    anchor = np.zeros(len(y_count), dtype=np.float32)
    for fold_index, (train_index, holdout_index) in enumerate(splitter.split(anchor, y_count, groups)):
        train_groups = set(groups[train_index].tolist())
        holdout_groups = set(groups[holdout_index].tolist())
        overlap = sorted(train_groups & holdout_groups)
        if overlap:
            raise RuntimeError(f"cross-fitting group leakage in fold {fold_index}: {overlap}")
        for source_index, source in enumerate(FEATURE_GROUPS):
            regressor = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=seed + 1000 * fold_index + 101 + source_index,
                n_jobs=1,
            )
            classifier = ExtraTreesClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=seed + 1000 * fold_index + 211 + source_index,
                n_jobs=1,
            )
            regressor.fit(matrices[source][train_index], residual_target[train_index])
            classifier.fit(matrices[source][train_index], y_risk[train_index])
            count_oof[holdout_index, source_index] = regressor.predict(matrices[source][holdout_index])
            risk_oof[holdout_index, source_index] = _positive_probability(
                classifier, matrices[source][holdout_index]
            )
        fold_records.append(
            {
                "fold": fold_index,
                "train_split_group_ids": sorted(train_groups),
                "oof_split_group_ids": sorted(holdout_groups),
                "train_run_ids": sorted({aligned[index].run_id for index in train_index}) if aligned else [],
                "oof_run_ids": sorted({aligned[index].run_id for index in holdout_index}) if aligned else [],
                "train_row_indices": train_index.tolist(),
                "oof_row_indices": holdout_index.tolist(),
            }
        )
    if np.isnan(count_oof).any() or np.isnan(risk_oof).any():
        raise RuntimeError("cross-fitting failed to produce exactly one OOF prediction per row")
    return count_oof, risk_oof, fold_records


class _SigmoidCalibrator:
    def __init__(self, seed: int):
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "_SigmoidCalibrator":
        self.model.fit(np.asarray(scores).reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


class _IdentityCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "_IdentityCalibrator":
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def _make_calibrator(name: str, seed: int):
    if name == "none":
        return _IdentityCalibrator()
    if name == "sigmoid":
        return _SigmoidCalibrator(seed)
    if name == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    raise ValueError(f"unknown calibration method: {name}")


def _calibrated_predict(calibrator, scores: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict(scores), dtype=float)
    return np.asarray(calibrator.predict(scores), dtype=float)


def _select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    aligned: list[Snapshot],
    threshold_grid: list[float],
    dt: float,
    horizon_steps: int,
) -> tuple[float, dict[str, float]]:
    ranked = []
    for threshold in threshold_grid:
        sample = classification_metrics(labels, scores >= threshold)
        event = alert_event_metrics(aligned, scores, threshold, dt, horizon_steps)
        ranked.append(
            (
                float(event["Alert_Event_F1"]),
                float(sample["Risk_F1"]),
                -float(event["False_Alert_Events_per_minute"]),
                float(threshold),
                {**sample, **event},
            )
        )
    chosen = max(ranked, key=lambda item: item[:4])
    return chosen[3], chosen[4]


class SourceGatedKineticTopoGuard:
    def __init__(self, horizon_steps: int, dt: float, seed: int, parameters: dict[str, Any] | None = None):
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.seed = int(seed)
        self.parameters = {**DEFAULT_PARAMETERS, **(parameters or {})}
        self.count_base_models: dict[str, ExtraTreesRegressor] = {}
        self.risk_base_models: dict[str, ExtraTreesClassifier] = {}
        self.count_meta_model: Ridge | None = None
        self.risk_meta_model: LogisticRegression | None = None
        self.calibrator: Any = _IdentityCalibrator()
        self.calibration_type = "none"
        self.risk_threshold = 0.5
        self.fold_records: list[dict[str, Any]] = []
        self.validation_candidates: list[dict[str, Any]] = []
        self.selected_parameters: dict[str, Any] = {}

    def _fit_full_bases(
        self,
        matrices: dict[str, np.ndarray],
        y_count: np.ndarray,
        y_risk: np.ndarray,
        aligned: list[Snapshot],
        min_leaf: int,
        seed_offset: int,
    ) -> tuple[dict[str, ExtraTreesRegressor], dict[str, ExtraTreesClassifier]]:
        residual = y_count - np.asarray([snapshot.beta_current for snapshot in aligned], dtype=float)
        regressors = {}
        classifiers = {}
        for index, source in enumerate(FEATURE_GROUPS):
            regressors[source] = ExtraTreesRegressor(
                n_estimators=int(self.parameters["n_estimators"]),
                min_samples_leaf=int(min_leaf),
                random_state=self.seed + seed_offset + 101 + index,
                n_jobs=1,
            ).fit(matrices[source], residual)
            classifiers[source] = ExtraTreesClassifier(
                n_estimators=int(self.parameters["n_estimators"]),
                min_samples_leaf=int(min_leaf),
                class_weight="balanced",
                random_state=self.seed + seed_offset + 211 + index,
                n_jobs=1,
            ).fit(matrices[source], y_risk)
        return regressors, classifiers

    @staticmethod
    def _base_predictions(matrices, regressors, classifiers) -> tuple[np.ndarray, np.ndarray]:
        count = np.column_stack([regressors[source].predict(matrices[source]) for source in FEATURE_GROUPS])
        risk = np.column_stack([_positive_probability(classifiers[source], matrices[source]) for source in FEATURE_GROUPS])
        return count, risk

    def fit(self, train_data: list[Snapshot], validation_data: list[Snapshot]) -> "SourceGatedKineticTopoGuard":
        train_x, y_train, risk_train, groups, aligned_train = build_source_matrices(
            train_data, self.horizon_steps, self.dt
        )
        val_x, y_val, risk_val, _, aligned_val = build_source_matrices(
            validation_data, self.horizon_steps, self.dt
        )
        base_val = np.asarray([snapshot.beta_current for snapshot in aligned_val], dtype=float)
        count_candidates = []
        risk_candidates = []
        bases_by_leaf = {}
        frozen_count_leaf = self.parameters.get("frozen_count_min_samples_leaf")
        frozen_risk_leaf = self.parameters.get("frozen_risk_min_samples_leaf")
        leaf_grid = (
            sorted({int(frozen_count_leaf), int(frozen_risk_leaf)})
            if frozen_count_leaf is not None and frozen_risk_leaf is not None
            else [int(value) for value in self.parameters["min_samples_leaf_grid"]]
        )
        ridge_grid = (
            [float(self.parameters["frozen_ridge_alpha"])]
            if "frozen_ridge_alpha" in self.parameters
            else self.parameters["ridge_alpha_grid"]
        )
        logistic_grid = (
            [float(self.parameters["frozen_logistic_c"])]
            if "frozen_logistic_c" in self.parameters
            else self.parameters["logistic_c_grid"]
        )
        calibration_options = (
            [str(self.parameters["frozen_calibration_type"])]
            if "frozen_calibration_type" in self.parameters
            else self.parameters["calibration_options"]
        )
        for leaf_index, min_leaf in enumerate(leaf_grid):
            count_oof, risk_oof, folds = grouped_oof_predictions(
                train_x,
                y_train,
                risk_train,
                groups,
                min_samples_leaf=int(min_leaf),
                n_estimators=int(self.parameters["n_estimators"]),
                n_splits=int(self.parameters["n_splits"]),
                seed=self.seed,
                aligned=aligned_train,
            )
            regressors, classifiers = self._fit_full_bases(
                train_x, y_train, risk_train, aligned_train, int(min_leaf), 10000 * leaf_index
            )
            bases_by_leaf[int(min_leaf)] = (regressors, classifiers, folds)
            count_val_base, risk_val_base = self._base_predictions(val_x, regressors, classifiers)
            residual_train = y_train - np.asarray(
                [snapshot.beta_current for snapshot in aligned_train], dtype=float
            )
            for ridge_alpha in ridge_grid:
                if frozen_count_leaf is not None and int(min_leaf) != int(frozen_count_leaf):
                    continue
                meta = Ridge(alpha=float(ridge_alpha)).fit(count_oof, residual_train)
                prediction = np.clip(base_val + meta.predict(count_val_base), 1.0, None)
                count_candidates.append(
                    {
                        "min_samples_leaf": int(min_leaf),
                        "ridge_alpha": float(ridge_alpha),
                        "validation_mae": float(np.mean(np.abs(y_val - prediction))),
                        "meta_model": meta,
                    }
                )
            for logistic_c in logistic_grid:
                if frozen_risk_leaf is not None and int(min_leaf) != int(frozen_risk_leaf):
                    continue
                meta = LogisticRegression(
                    C=float(logistic_c), max_iter=1000, random_state=self.seed
                ).fit(risk_oof, risk_train)
                raw = meta.predict_proba(risk_val_base)[:, 1]
                for calibration_name in calibration_options:
                    calibrator = _make_calibrator(str(calibration_name), self.seed)
                    calibrator.fit(raw, risk_val)
                    calibrated = _calibrated_predict(calibrator, raw)
                    probability = risk_probability_metrics(risk_val, calibrated)
                    threshold, threshold_metrics = _select_threshold(
                        risk_val,
                        calibrated,
                        aligned_val,
                        [float(value) for value in self.parameters["threshold_grid"]],
                        self.dt,
                        self.horizon_steps,
                    )
                    risk_candidates.append(
                        {
                            "min_samples_leaf": int(min_leaf),
                            "logistic_c": float(logistic_c),
                            "calibration_type": str(calibration_name),
                            "validation_brier": float(probability["Risk_Brier"]),
                            "validation_ece": float(probability["Risk_ECE"]),
                            "validation_event_f1": float(threshold_metrics["Alert_Event_F1"]),
                            "validation_sample_f1": float(threshold_metrics["Risk_F1"]),
                            "validation_false_alert_events_per_minute": float(
                                threshold_metrics["False_Alert_Events_per_minute"]
                            ),
                            "selected_threshold": float(threshold),
                            "meta_model": meta,
                            "calibrator": calibrator,
                        }
                    )
        selected_count = min(
            count_candidates,
            key=lambda row: (row["validation_mae"], row["ridge_alpha"], row["min_samples_leaf"]),
        )
        selected_risk = min(
            risk_candidates,
            key=lambda row: (
                row["validation_brier"],
                row["validation_ece"],
                -row["validation_event_f1"],
                row["logistic_c"],
                row["min_samples_leaf"],
                row["calibration_type"],
            ),
        )
        count_leaf = int(selected_count["min_samples_leaf"])
        risk_leaf = int(selected_risk["min_samples_leaf"])
        self.count_base_models = bases_by_leaf[count_leaf][0]
        self.risk_base_models = bases_by_leaf[risk_leaf][1]
        self.count_meta_model = selected_count["meta_model"]
        self.risk_meta_model = selected_risk["meta_model"]
        self.calibrator = selected_risk["calibrator"]
        self.calibration_type = str(selected_risk["calibration_type"])
        self.risk_threshold = float(selected_risk["selected_threshold"])
        self.fold_records = bases_by_leaf[count_leaf][2]
        self.selected_parameters = {
            "count_min_samples_leaf": count_leaf,
            "risk_min_samples_leaf": risk_leaf,
            "ridge_alpha": float(selected_count["ridge_alpha"]),
            "logistic_c": float(selected_risk["logistic_c"]),
            "calibration_type": self.calibration_type,
            "selected_threshold": self.risk_threshold,
            "selection_split": "validation",
            "hyperparameters_frozen_before_confirmatory_test": bool(
                frozen_count_leaf is not None
                and frozen_risk_leaf is not None
                and "frozen_ridge_alpha" in self.parameters
                and "frozen_logistic_c" in self.parameters
                and "frozen_calibration_type" in self.parameters
            ),
        }
        self.validation_candidates = [
            {key: value for key, value in row.items() if key not in {"meta_model", "calibrator"}}
            for row in risk_candidates
        ]
        return self

    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray, list[Snapshot]]:
        if self.count_meta_model is None or self.risk_meta_model is None:
            raise RuntimeError("model must be fit before prediction")
        matrices, _, _, _, aligned = build_source_matrices(snapshots, self.horizon_steps, self.dt)
        count_base, _ = self._base_predictions(
            matrices, self.count_base_models, self.risk_base_models
        )
        risk_base = np.column_stack(
            [_positive_probability(self.risk_base_models[source], matrices[source]) for source in FEATURE_GROUPS]
        )
        persistence = np.asarray([snapshot.beta_current for snapshot in aligned], dtype=float)
        count = np.clip(persistence + self.count_meta_model.predict(count_base), 1.0, None)
        raw_risk = self.risk_meta_model.predict_proba(risk_base)[:, 1]
        risk = np.clip(_calibrated_predict(self.calibrator, raw_risk), 0.0, 1.0)
        return count, risk, aligned

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "feature_group_names": list(FEATURE_GROUPS),
            "base_model_params": {
                "count": {name: model.get_params() for name, model in self.count_base_models.items()},
                "risk": {name: model.get_params() for name, model in self.risk_base_models.items()},
            },
            "count_meta_coefficients": dict(
                zip(FEATURE_GROUPS, self.count_meta_model.coef_.astype(float).tolist())
            ),
            "count_meta_intercept": float(self.count_meta_model.intercept_),
            "risk_meta_coefficients": dict(
                zip(FEATURE_GROUPS, self.risk_meta_model.coef_[0].astype(float).tolist())
            ),
            "risk_meta_intercept": float(self.risk_meta_model.intercept_[0]),
            "calibration_type": self.calibration_type,
            "selected_threshold": self.risk_threshold,
            "selected_parameters": self.selected_parameters,
            "training_folds": self.fold_records,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SourceGatedKineticTopoGuard":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("serialized artifact is not a SourceGatedKineticTopoGuard")
        return model


def fit_source_gated_kinetic_topoguard(
    train_data: list[Snapshot],
    validation_data: list[Snapshot],
    horizon_steps: int,
    dt: float,
    seed: int,
    parameters: dict[str, Any] | None = None,
) -> TrainResult:
    model = SourceGatedKineticTopoGuard(horizon_steps, dt, seed, parameters).fit(
        train_data, validation_data
    )
    return TrainResult(model=model, model_name=MODEL_NAME, inference_ms=0.0)


class CurrentStateExtraTreesPredictor:
    """Equal-learner current-source comparator, selected only on validation."""

    def __init__(self, horizon_steps: int, dt: float, seed: int, n_estimators: int = 128):
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.regressor: ExtraTreesRegressor | None = None
        self.classifier: ExtraTreesClassifier | None = None
        self.residual_scale = 0.0
        self.risk_threshold = 0.5

    def fit(self, train_data: list[Snapshot], validation_data: list[Snapshot]):
        train_x, y_train, risk_train, _, aligned_train = build_source_matrices(
            train_data, self.horizon_steps, self.dt
        )
        val_x, y_val, risk_val, _, aligned_val = build_source_matrices(
            validation_data, self.horizon_steps, self.dt
        )
        base_train = np.asarray([snapshot.beta_current for snapshot in aligned_train], dtype=float)
        base_val = np.asarray([snapshot.beta_current for snapshot in aligned_val], dtype=float)
        self.regressor = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            random_state=self.seed + 101,
            n_jobs=1,
        ).fit(train_x["current"], y_train - base_train)
        self.classifier = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.seed + 211,
            n_jobs=1,
        ).fit(train_x["current"], risk_train)
        residual_val = self.regressor.predict(val_x["current"])
        self.residual_scale = min(
            (
                (
                    float(np.mean(np.abs(y_val - np.clip(base_val + alpha * residual_val, 1.0, None)))),
                    float(alpha),
                )
                for alpha in np.linspace(0.0, 1.0, 11)
            ),
            key=lambda item: (item[0], item[1]),
        )[1]
        validation_scores = _positive_probability(self.classifier, val_x["current"])
        self.risk_threshold, _ = _select_threshold(
            risk_val,
            validation_scores,
            aligned_val,
            list(DEFAULT_PARAMETERS["threshold_grid"]),
            self.dt,
            self.horizon_steps,
        )
        return self

    def predict_snapshots(self, snapshots: list[Snapshot]):
        matrices, _, _, _, aligned = build_source_matrices(
            snapshots, self.horizon_steps, self.dt
        )
        persistence = np.asarray([snapshot.beta_current for snapshot in aligned], dtype=float)
        residual = self.regressor.predict(matrices["current"])
        prediction = np.clip(persistence + self.residual_scale * residual, 1.0, None)
        risk = _positive_probability(self.classifier, matrices["current"])
        return prediction, risk, aligned


def fit_current_state_extratrees(
    train_data: list[Snapshot], validation_data: list[Snapshot], horizon_steps: int, dt: float, seed: int
) -> TrainResult:
    model = CurrentStateExtraTreesPredictor(horizon_steps, dt, seed).fit(
        train_data, validation_data
    )
    return TrainResult(model=model, model_name="Current-state ExtraTrees", inference_ms=0.0)
