from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
import numpy as np

from .dataset import Snapshot
from .geometry import pairwise_distances
from .graph_utils import adjacency_from_radius, avg_clustering_coefficient, betti_zero, degree_features, largest_component_ratio

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from .models import FANETTopoGNN, FANETTopoGNNConcat, GATEncoder, GCNEncoder, GraphRegressor, PIRegressor, SAGEEncoder, TemporalRegressor
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = object
    DataLoader = object
    Dataset = object
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


class NumpyPipeline:
    def __init__(self, estimator):
        self.scaler = Standardizer()
        self.estimator = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyPipeline":
        self.scaler.fit(x)
        self.estimator.fit(self.scaler.transform(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.scaler.transform(x))


class LinearLeastSquares:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearLeastSquares":
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        self.w, *_ = np.linalg.lstsq(xb, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        return xb @ self.w


class RidgeLeastSquares:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeLeastSquares":
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        eye = np.eye(xb.shape[1], dtype=xb.dtype)
        eye[-1, -1] = 0.0
        self.w = np.linalg.solve(xb.T @ xb + self.alpha * eye, xb.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        return xb @ self.w


class KNNRegressor:
    def __init__(self, k: int = 7):
        self.k = k

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.x = x
        self.y = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        dists = ((x[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argsort(dists, axis=1)[:, : self.k]
        return self.y[idx].mean(axis=1)


class RandomFeatureRegressor:
    def __init__(self, hidden_dim: int = 64, seed: int = 42, alpha: float = 1e-2):
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomFeatureRegressor":
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 0.5, size=(x.shape[1], self.hidden_dim))
        self.b1 = rng.normal(0.0, 0.2, size=(self.hidden_dim,))
        h = np.tanh(x @ self.w1 + self.b1)
        xb = np.concatenate([h, np.ones((len(h), 1), dtype=h.dtype)], axis=1)
        eye = np.eye(xb.shape[1], dtype=xb.dtype)
        eye[-1, -1] = 0.0
        self.w2 = np.linalg.solve(xb.T @ xb + self.alpha * eye, xb.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        xb = np.concatenate([h, np.ones((len(h), 1), dtype=h.dtype)], axis=1)
        return xb @ self.w2


class EnsembleRegressor:
    def __init__(self, estimators: list[object]):
        self.estimators = estimators

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnsembleRegressor":
        for estimator in self.estimators:
            estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = [np.asarray(est.predict(x), dtype=float) for est in self.estimators]
        return np.mean(np.stack(preds), axis=0)


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


if TORCH_AVAILABLE:
    class SnapshotDataset(Dataset):
        def __init__(self, snapshots: list[Snapshot]):
            self.snapshots = snapshots

        def __len__(self) -> int:
            return len(self.snapshots)

        def __getitem__(self, idx: int) -> Snapshot:
            return self.snapshots[idx]


def snapshot_feature_vector(snapshot: Snapshot, mode: str) -> np.ndarray:
    degree = snapshot.adjacency.sum(axis=1)
    degree_hist = np.histogram(degree, bins=5, range=(0, max(snapshot.n_nodes - 1, 1)))[0].astype(np.float32)
    dist = np.linalg.norm(snapshot.positions[:, None, :] - snapshot.positions[None, :, :], axis=-1)
    tri = dist[np.triu_indices_from(dist, k=1)]
    dist_summary = np.array(
        [
            float(tri.mean()) if tri.size else 0.0,
            float(np.quantile(tri, 0.25)) if tri.size else 0.0,
            float(np.quantile(tri, 0.5)) if tri.size else 0.0,
            float(np.quantile(tri, 0.75)) if tri.size else 0.0,
        ],
        dtype=np.float32,
    )
    pi_summary = np.array(
        [
            float(snapshot.pi.mean()),
            float(snapshot.pi.std()),
            float(snapshot.pi.max()),
            float(np.quantile(snapshot.pi, 0.9)),
            float(np.quantile(snapshot.pi, 0.99)),
        ],
        dtype=np.float32,
    )
    if mode == "pi_only":
        feat = np.concatenate([snapshot.pi, pi_summary]).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "hybrid":
        feat = np.concatenate([snapshot.stats, degree_hist, dist_summary, pi_summary, snapshot.pi]).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "graphsage":
        feat = np.concatenate([snapshot.stats, degree_hist, dist_summary]).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "gat":
        feat = np.concatenate([snapshot.stats, degree_hist, pi_summary]).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = np.concatenate([snapshot.stats, degree_hist]).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def shallow_feature_vector(snapshot: Snapshot) -> np.ndarray:
    largest_component_count = snapshot.stats[7] * snapshot.n_nodes
    feat = np.asarray(
        [
            snapshot.stats[0],  # density
            snapshot.stats[1],  # minimum inter-node distance
            snapshot.stats[4],  # mean degree
            snapshot.stats[5],  # maximum degree
            largest_component_count,
            snapshot.stats[8],  # average clustering coefficient
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_array(values: np.ndarray | list[float]) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _current_edge_count(snapshot: Snapshot) -> float:
    return float(snapshot.edge_count_fixed if snapshot.graph_policy == "fixed" else snapshot.edge_count_adaptive)


def _kinetic_projection_features(snapshot: Snapshot, horizon_steps: int, dt: float) -> tuple[np.ndarray, float]:
    """Forward-project UAV positions and summarise imminent topological margins."""
    tau = max(float(horizon_steps) * float(dt), 0.0)
    positions = snapshot.positions.astype(float)
    projected = positions + snapshot.velocities.astype(float) * tau
    current_dist = pairwise_distances(positions)
    projected_dist = pairwise_distances(projected)
    tri_idx = np.triu_indices(snapshot.n_nodes, k=1)
    current_tri = current_dist[tri_idx]
    projected_tri = projected_dist[tri_idx]
    radius = max(float(snapshot.radius), 1e-6)
    current_geom_adj = adjacency_from_radius(current_dist, radius)
    projected_geom_adj = adjacency_from_radius(projected_dist, radius)
    beta_current_geom = float(betti_zero(current_geom_adj))
    beta_projected_geom = float(betti_zero(projected_geom_adj))
    degree_mean, degree_max, degree_min = degree_features(projected_geom_adj)
    pair_count = max(float(len(current_tri)), 1.0)

    current_margin = (radius - current_tri) / radius if current_tri.size else np.asarray([], dtype=float)
    projected_margin = (radius - projected_tri) / radius if projected_tri.size else np.asarray([], dtype=float)
    distance_delta = (projected_tri - current_tri) / radius if current_tri.size else np.asarray([], dtype=float)
    crossing_out = float(np.sum((current_tri <= radius) & (projected_tri > radius))) / pair_count if current_tri.size else 0.0
    crossing_in = float(np.sum((current_tri > radius) & (projected_tri <= radius))) / pair_count if current_tri.size else 0.0
    low_projected_margin = _safe_quantile(projected_margin, 0.05)
    margin_risk = float(_sigmoid(np.asarray([-8.0 * low_projected_margin]))[0])
    projected_risk = 1.0 if beta_projected_geom > 1.0 else 0.0
    crossing_risk = min(1.0, 5.0 * crossing_out)
    kinetic_risk = max(margin_risk, projected_risk, crossing_risk)

    speed = np.linalg.norm(snapshot.velocities.astype(float), axis=1)
    features = [
        float(snapshot.beta_current),
        float(snapshot.beta_fixed),
        float(snapshot.beta_adaptive),
        float(snapshot.beta_adaptive - snapshot.beta_fixed),
        _current_edge_count(snapshot) / pair_count,
        float(snapshot.edge_count_fixed) / pair_count,
        float(snapshot.edge_count_adaptive) / pair_count,
        beta_current_geom,
        beta_projected_geom,
        beta_projected_geom - float(snapshot.beta_current),
        largest_component_ratio(projected_geom_adj),
        avg_clustering_coefficient(projected_geom_adj),
        degree_mean / max(snapshot.n_nodes - 1, 1),
        degree_max / max(snapshot.n_nodes - 1, 1),
        degree_min / max(snapshot.n_nodes - 1, 1),
        _safe_quantile(current_margin, 0.00),
        _safe_quantile(current_margin, 0.05),
        _safe_quantile(current_margin, 0.10),
        _safe_quantile(current_margin, 0.50),
        float(current_margin.mean()) if current_margin.size else 0.0,
        float(current_margin.std()) if current_margin.size else 0.0,
        _safe_quantile(projected_margin, 0.00),
        low_projected_margin,
        _safe_quantile(projected_margin, 0.10),
        _safe_quantile(projected_margin, 0.50),
        float(projected_margin.mean()) if projected_margin.size else 0.0,
        float(projected_margin.std()) if projected_margin.size else 0.0,
        _safe_quantile(distance_delta, 0.05),
        float(distance_delta.mean()) if distance_delta.size else 0.0,
        _safe_quantile(distance_delta, 0.95),
        float(distance_delta.std()) if distance_delta.size else 0.0,
        float(np.mean(distance_delta < 0.0)) if distance_delta.size else 0.0,
        float(np.mean(distance_delta > 0.0)) if distance_delta.size else 0.0,
        crossing_out,
        crossing_in,
        float(speed.mean()) / 30.0 if speed.size else 0.0,
        float(speed.std()) / 30.0 if speed.size else 0.0,
        float(speed.max()) / 30.0 if speed.size else 0.0,
        float(snapshot.n_nodes) / 30.0,
        float(snapshot.radius) / 1000.0,
        float(snapshot.radius_adaptive - snapshot.radius_fixed) / 1000.0,
        float(snapshot.graph_policy == "adaptive"),
        float(snapshot.radio_scenario == "degraded"),
        float(snapshot.mobility == "rwp"),
        float(snapshot.mobility == "gm"),
        float(snapshot.mobility == "mission"),
    ]
    return _safe_array(features), kinetic_risk


def kinetic_topoguard_feature_vector(snapshot: Snapshot, previous: Snapshot | None, horizon_steps: int, dt: float) -> tuple[np.ndarray, float]:
    kinetic, kinetic_risk = _kinetic_projection_features(snapshot, horizon_steps, dt)
    hybrid = snapshot_feature_vector(snapshot, "hybrid")
    shallow = shallow_feature_vector(snapshot)
    if previous is None:
        lag = [
            float(snapshot.beta_current),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    else:
        lag = [
            float(previous.beta_current),
            float(snapshot.beta_current - previous.beta_current),
            _current_edge_count(snapshot) - _current_edge_count(previous),
            float(snapshot.stats[0] - previous.stats[0]),
            float(snapshot.stats[1] - previous.stats[1]),
            float(snapshot.stats[7] - previous.stats[7]),
        ]
    return _safe_array(np.concatenate([hybrid, shallow, kinetic, _safe_array(lag)])), kinetic_risk


def _build_kinetic_topoguard_matrix(
    snapshots: list[Snapshot],
    horizon_steps: int,
    dt: float,
) -> tuple[np.ndarray, list[Snapshot], np.ndarray]:
    grouped: dict[str, list[Snapshot]] = {}
    for snap in snapshots:
        grouped.setdefault(snap.run_id, []).append(snap)
    rows = []
    aligned: list[Snapshot] = []
    kinetic_scores = []
    for _, seq in sorted(grouped.items()):
        previous = None
        for snap in sorted(seq, key=lambda item: item.time_index):
            features, kinetic_risk = kinetic_topoguard_feature_vector(snap, previous, horizon_steps, dt)
            rows.append(features)
            aligned.append(snap)
            kinetic_scores.append(float(kinetic_risk))
            previous = snap
    return np.vstack(rows), aligned, np.asarray(kinetic_scores, dtype=float)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _positive_class_probability(classifier, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
    classes = np.asarray(classifier.classes_)
    positive_index = np.flatnonzero(classes == 1)
    if positive_index.size:
        return probabilities[:, int(positive_index[0])]
    return np.zeros(len(features), dtype=float)


def _lead_median_for_threshold(
    snapshots: list[Snapshot],
    risk_scores: np.ndarray,
    dt: float,
    horizon_steps: int,
    threshold: float,
) -> float:
    grouped: dict[str, list[tuple[Snapshot, float]]] = {}
    for snap, score in zip(snapshots, risk_scores):
        grouped.setdefault(snap.run_id, []).append((snap, float(score)))
    leads = []
    for _, pairs in sorted(grouped.items()):
        pairs = sorted(pairs, key=lambda item: item[0].time_index)
        run_snaps = [snap for snap, _ in pairs]
        run_scores = np.asarray([score for _, score in pairs], dtype=float)
        for idx in range(1, len(run_snaps)):
            if run_snaps[idx - 1].adjacency.shape != run_snaps[idx].adjacency.shape:
                continue
            if not np.any(run_snaps[idx - 1].adjacency != run_snaps[idx].adjacency):
                continue
            warned_idx = None
            start = max(0, idx - horizon_steps)
            for look in range(start, idx):
                if run_snaps[look].beta_current <= 1 and run_scores[look] >= threshold and (look + horizon_steps) >= idx:
                    warned_idx = look
                    break
            leads.append(0.0 if warned_idx is None else (idx - warned_idx) * dt * 1000.0)
    if not leads:
        return 0.0
    return float(np.median(np.asarray(leads, dtype=float)))


class KineticTopoGuardPredictor:
    def __init__(self, horizon_steps: int, dt: float, seed: int):
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.seed = int(seed)
        self.risk_threshold = 0.5
        self.residual_scale = 0.0
        self.regressor = None
        self.classifiers: list[object] = []
        self.classifier_weights = np.asarray([], dtype=float)
        self.blend_weights = (0.75, 0.15, 0.10)

    def fit(self, train_data: list[Snapshot], val_data: list[Snapshot]) -> "KineticTopoGuardPredictor":
        train_x, train_aligned, train_kinetic = _build_kinetic_topoguard_matrix(train_data, self.horizon_steps, self.dt)
        val_x, val_aligned, val_kinetic = _build_kinetic_topoguard_matrix(val_data, self.horizon_steps, self.dt)
        y_train = np.asarray([snap.beta_target for snap in train_aligned], dtype=float)
        y_val = np.asarray([snap.beta_target for snap in val_aligned], dtype=float)
        base_train = np.asarray([snap.beta_current for snap in train_aligned], dtype=float)
        base_val = np.asarray([snap.beta_current for snap in val_aligned], dtype=float)
        residual_train = y_train - base_train
        risk_train = np.asarray([snap.frag_at_horizon for snap in train_aligned], dtype=int)
        risk_val = np.asarray([snap.frag_at_horizon for snap in val_aligned], dtype=int)

        if SKLEARN_AVAILABLE:
            candidates = [
                ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2, random_state=self.seed + 101, n_jobs=-1),
                GradientBoostingRegressor(n_estimators=160, learning_rate=0.04, max_depth=3, random_state=self.seed + 103),
                RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=self.seed + 107, n_jobs=-1),
            ]
        else:
            candidates = [
                NumpyPipeline(RidgeLeastSquares(alpha=0.5)),
                NumpyPipeline(RandomFeatureRegressor(hidden_dim=160, seed=self.seed + 101, alpha=1e-2)),
            ]

        best_mae = mean_absolute_error(y_val, base_val) if SKLEARN_AVAILABLE else mae_metric(y_val, base_val)
        self.regressor = candidates[0]
        self.residual_scale = 0.0
        for candidate in candidates:
            candidate.fit(train_x, residual_train)
            residual_val = np.asarray(candidate.predict(val_x), dtype=float)
            for scale in np.linspace(0.0, 1.0, 11):
                pred_val = np.clip(base_val + float(scale) * residual_val, 1.0, None)
                mae = mean_absolute_error(y_val, pred_val) if SKLEARN_AVAILABLE else mae_metric(y_val, pred_val)
                if mae < best_mae - 1e-5:
                    best_mae = mae
                    self.regressor = candidate
                    self.residual_scale = float(scale)

        val_pred = self._predict_from_matrix(val_x, val_aligned)
        reg_risk_val = np.clip((val_pred - 1.0) / 1.5, 0.0, 1.0)
        if SKLEARN_AVAILABLE and np.unique(risk_train).size >= 2:
            classifier_candidates = [
                ExtraTreesClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=self.seed + 211, n_jobs=-1),
                GradientBoostingClassifier(n_estimators=160, learning_rate=0.04, max_depth=2, random_state=self.seed + 223),
                RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=self.seed + 227, n_jobs=-1),
            ]
            probs = []
            f1s = []
            for classifier in classifier_candidates:
                classifier.fit(train_x, risk_train)
                prob = _positive_class_probability(classifier, val_x)
                probs.append(prob)
                f1s.append(_binary_f1(risk_val, prob >= 0.5))
            weights = np.square(np.asarray(f1s, dtype=float) + 1e-3)
            weights = weights / max(float(weights.sum()), 1e-8)
            self.classifiers = classifier_candidates
            self.classifier_weights = weights
            clf_risk_val = np.average(np.column_stack(probs), axis=1, weights=weights)
        else:
            self.classifiers = []
            self.classifier_weights = np.asarray([], dtype=float)
            clf_risk_val = (
                np.full(len(val_x), float(risk_train[0]), dtype=float)
                if risk_train.size and np.unique(risk_train).size == 1
                else reg_risk_val
            )

        blend_candidates = [
            (0.80, 0.10, 0.10),
            (0.70, 0.15, 0.15),
            (0.60, 0.20, 0.20),
            (0.55, 0.15, 0.30),
            (0.45, 0.20, 0.35),
        ]
        threshold_grid = np.linspace(0.05, 0.95, 181)
        target_lead = 0.5 * max(self.horizon_steps * self.dt * 1000.0, 1.0)
        best_any = (-1.0, 0.5, blend_candidates[0], 0.0, 0.0)
        best_eligible = (-1.0, 0.5, blend_candidates[0], 0.0, 0.0)
        for blend in blend_candidates:
            alpha, beta, gamma = blend
            score = np.clip(alpha * clf_risk_val + beta * reg_risk_val + gamma * val_kinetic, 0.0, 1.0)
            blend_best_f1 = 0.0
            candidates_for_blend = []
            for threshold in threshold_grid:
                pred = score >= float(threshold)
                f1 = _binary_f1(risk_val, pred)
                lead_median = _lead_median_for_threshold(val_aligned, score, self.dt, self.horizon_steps, float(threshold))
                blend_best_f1 = max(blend_best_f1, f1)
                candidates_for_blend.append((f1, float(threshold), blend, lead_median, float(np.mean(pred))))
                lead_norm = lead_median / max(self.horizon_steps * self.dt * 1000.0, 1.0)
                objective = f1 + 0.20 * min(lead_norm, 1.0)
                if objective > best_any[0]:
                    best_any = (objective, float(threshold), blend, f1, lead_median)
            for f1, threshold, blend, lead_median, _positive_rate in candidates_for_blend:
                if lead_median >= target_lead and f1 >= 0.50 * blend_best_f1 and f1 > best_eligible[0]:
                    best_eligible = (f1, threshold, blend, f1, lead_median)
        chosen = best_eligible if best_eligible[0] >= 0.0 else best_any
        self.risk_threshold = float(chosen[1])
        self.blend_weights = chosen[2]
        return self

    def _predict_from_matrix(self, x: np.ndarray, aligned: list[Snapshot]) -> np.ndarray:
        base = np.asarray([snap.beta_current for snap in aligned], dtype=float)
        if self.regressor is None or self.residual_scale <= 0.0:
            return np.clip(base, 1.0, None)
        residual = np.asarray(self.regressor.predict(x), dtype=float)
        return np.clip(base + self.residual_scale * residual, 1.0, None)

    def _risk_from_matrix(self, x: np.ndarray, aligned: list[Snapshot], kinetic_scores: np.ndarray, preds: np.ndarray) -> np.ndarray:
        reg_risk = np.clip((preds - 1.0) / 1.5, 0.0, 1.0)
        if self.classifiers:
            probs = [_positive_class_probability(classifier, x) for classifier in self.classifiers]
            clf_risk = np.average(np.column_stack(probs), axis=1, weights=self.classifier_weights)
        else:
            clf_risk = reg_risk
        alpha, beta, gamma = self.blend_weights
        score = alpha * clf_risk + beta * reg_risk + gamma * kinetic_scores
        return np.clip(score, 0.0, 1.0)

    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray, list[Snapshot]]:
        x, aligned, kinetic_scores = _build_kinetic_topoguard_matrix(snapshots, self.horizon_steps, self.dt)
        preds = self._predict_from_matrix(x, aligned)
        risk = self._risk_from_matrix(x, aligned, kinetic_scores, preds)
        return preds, risk, aligned


def fit_kinetic_topoguard(
    train_data: list[Snapshot],
    val_data: list[Snapshot],
    horizon_steps: int,
    dt: float,
    seed: int,
) -> TrainResult:
    model = KineticTopoGuardPredictor(horizon_steps=horizon_steps, dt=dt, seed=seed).fit(train_data, val_data)
    start = time.perf_counter()
    preds, _, _ = model.predict_snapshots(val_data[: min(len(val_data), 64)])
    inference_ms = ((time.perf_counter() - start) * 1000.0) / max(len(preds), 1)
    return TrainResult(model=model, model_name="Kinetic-TopoGuard", inference_ms=inference_ms)


class SnapshotRegressorWrapper:
    def __init__(self, name: str, estimator, mode: str):
        self.name = name
        self.estimator = estimator
        self.mode = mode

    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "shallow":
            x = np.stack([shallow_feature_vector(s) for s in snapshots])
        else:
            x = np.stack([snapshot_feature_vector(s, self.mode) for s in snapshots])
        preds = np.asarray(self.estimator.predict(x), dtype=float)
        risk = (preds > 1.0).astype(float)
        return preds, np.asarray(risk, dtype=float)


class TemporalRegressorWrapper:
    def __init__(self, name: str, estimator, mode: str, window: int):
        self.name = name
        self.estimator = estimator
        self.mode = mode
        self.window = window

    def _windows(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, list[Snapshot]]:
        grouped: dict[str, list[Snapshot]] = {}
        for snap in snapshots:
            grouped.setdefault(snap.run_id, []).append(snap)
        x_rows = []
        aligned = []
        for _, seq in sorted(grouped.items()):
            seq = sorted(seq, key=lambda item: item.time_index)
            features = [snapshot_feature_vector(s, self.mode) for s in seq]
            for idx in range(self.window - 1, len(seq)):
                x_rows.append(np.concatenate(features[idx - self.window + 1 : idx + 1]).astype(np.float32))
                aligned.append(seq[idx])
        return np.stack(x_rows), aligned

    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray, list[Snapshot]]:
        x, aligned = self._windows(snapshots)
        preds = np.asarray(self.estimator.predict(x), dtype=float)
        risk = (preds > 1.0).astype(float)
        return preds, np.asarray(risk, dtype=float), aligned


if TORCH_AVAILABLE:
    def collate_snapshots(batch: list[Snapshot]) -> dict[str, torch.Tensor]:
        max_nodes = max(item.n_nodes for item in batch)
        feat_dim = batch[0].node_features.shape[1]
        pi_dim = batch[0].pi.shape[0]
        xs, adjs, masks, pis, targets = [], [], [], [], []
        for item in batch:
            x = np.zeros((max_nodes, feat_dim), dtype=np.float32)
            adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
            mask = np.zeros(max_nodes, dtype=np.float32)
            x[: item.n_nodes] = item.node_features
            adj[: item.n_nodes, : item.n_nodes] = item.adjacency
            mask[: item.n_nodes] = 1.0
            xs.append(x)
            adjs.append(adj)
            masks.append(mask)
            pis.append(item.pi.reshape(pi_dim))
            targets.append(item.beta_target)
        return {
            "x": torch.tensor(np.stack(xs)),
            "adj": torch.tensor(np.stack(adjs)),
            "mask": torch.tensor(np.stack(masks)),
            "pi": torch.tensor(np.stack(pis)),
            "y": torch.tensor(np.asarray(targets, dtype=np.float32)),
        }


    class TemporalWindowDataset(Dataset):
        def __init__(self, snapshots: list[Snapshot], window: int):
            grouped: dict[str, list[Snapshot]] = {}
            for snap in snapshots:
                grouped.setdefault(snap.run_id, []).append(snap)
            self.items: list[list[Snapshot]] = []
            for _, seq in sorted(grouped.items()):
                seq = sorted(seq, key=lambda item: item.time_index)
                for idx in range(window - 1, len(seq)):
                    self.items.append(seq[idx - window + 1 : idx + 1])

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> list[Snapshot]:
            return self.items[idx]


    def collate_temporal(batch: list[list[Snapshot]]) -> dict[str, torch.Tensor]:
        max_nodes = max(item.n_nodes for seq in batch for item in seq)
        window = len(batch[0])
        feat_dim = batch[0][0].node_features.shape[1]
        pi_dim = batch[0][0].pi.shape[0]
        x = np.zeros((len(batch), window, max_nodes, feat_dim), dtype=np.float32)
        adj = np.zeros((len(batch), window, max_nodes, max_nodes), dtype=np.float32)
        mask = np.zeros((len(batch), window, max_nodes), dtype=np.float32)
        pi = np.zeros((len(batch), window, pi_dim), dtype=np.float32)
        y = np.zeros(len(batch), dtype=np.float32)
        for b_idx, seq in enumerate(batch):
            for t_idx, item in enumerate(seq):
                x[b_idx, t_idx, : item.n_nodes] = item.node_features
                adj[b_idx, t_idx, : item.n_nodes, : item.n_nodes] = item.adjacency
                mask[b_idx, t_idx, : item.n_nodes] = 1.0
                pi[b_idx, t_idx] = item.pi
            y[b_idx] = seq[-1].beta_target
        return {"x": torch.tensor(x), "adj": torch.tensor(adj), "mask": torch.tensor(mask), "pi": torch.tensor(pi), "y": torch.tensor(y)}
else:
    SnapshotDataset = None
    TemporalWindowDataset = None
    collate_snapshots = None
    collate_temporal = None


@dataclass
class TrainResult:
    model: object
    model_name: str
    inference_ms: float
    risk_inference_ms: float = 0.0


def _epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    loss_fn = nn.MSELoss()
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["x"], batch["adj"], batch["pi"], batch["mask"])
        loss = loss_fn(pred, batch["y"])
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(batch["y"])
        count += len(batch["y"])
    return total / max(count, 1)


def profile_torch_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    seen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch["x"], batch["adj"], batch["pi"], batch["mask"])
            seen += len(batch["y"])
            if seen >= 128:
                break
    return ((time.perf_counter() - start) * 1000.0) / max(seen, 1)


def train_torch_model(model_name: str, train_data: list[Snapshot], val_data: list[Snapshot], train_cfg: dict, pi_dim: int, seed: int) -> TrainResult:
    if not TORCH_AVAILABLE:
        return train_surrogate_model(model_name, train_data, val_data)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = train_cfg["hidden_dim"]
    dropout = train_cfg["dropout"]
    input_dim = train_data[0].node_features.shape[1]
    if model_name == "GCN":
        model = GraphRegressor(GCNEncoder(input_dim, hidden_dim, dropout), hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    elif model_name == "GAT":
        model = GraphRegressor(GATEncoder(input_dim, hidden_dim, dropout), hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    elif model_name == "GraphSAGE":
        model = GraphRegressor(SAGEEncoder(input_dim, hidden_dim, dropout), hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    elif model_name == "PI+MLP":
        model = PIRegressor(pi_dim, hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    elif model_name == "FANET-TopoGNN":
        model = FANETTopoGNN(input_dim, pi_dim, hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    elif model_name == "FANET-TopoGNN (concat)":
        model = FANETTopoGNNConcat(input_dim, pi_dim, hidden_dim, dropout)
        train_loader = DataLoader(SnapshotDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_snapshots)
        val_loader = DataLoader(SnapshotDataset(val_data), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_snapshots)
    else:
        kind, window_str = model_name.split(":")
        window = int(window_str)
        model = TemporalRegressor(GCNEncoder(input_dim, hidden_dim, dropout), hidden_dim, kind, dropout)
        train_loader = DataLoader(TemporalWindowDataset(train_data, window), batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_temporal)
        val_loader = DataLoader(TemporalWindowDataset(val_data, window), batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_temporal)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    for _ in range(train_cfg["epochs"]):
        _epoch(model, train_loader, optimizer, device)
        val_loss = _epoch(model, val_loader, None, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return TrainResult(model=model, model_name=model_name, inference_ms=profile_torch_model(model, val_loader, device))


class HeuristicPredictor:
    def fit(self, train_data: list[Snapshot]) -> "HeuristicPredictor":
        xs = np.stack([shallow_feature_vector(snap) for snap in train_data])
        y = np.asarray([snap.beta_target for snap in train_data], dtype=float)
        density_grid = np.quantile(xs[:, 0], [0.25, 0.5, 0.75])
        min_dist_grid = np.quantile(xs[:, 1], [0.25, 0.5, 0.75])
        mean_degree_grid = np.quantile(xs[:, 2], [0.25, 0.5, 0.75])
        best_mae = float("inf")
        self.thresholds = (float(density_grid[1]), float(min_dist_grid[1]), float(mean_degree_grid[1]))
        for density_thr in density_grid:
            for min_dist_thr in min_dist_grid:
                for mean_degree_thr in mean_degree_grid:
                    pred = self._predict_features(xs, float(density_thr), float(min_dist_thr), float(mean_degree_thr))
                    mae = mae_metric(y, pred)
                    if mae < best_mae:
                        best_mae = mae
                        self.thresholds = (float(density_thr), float(min_dist_thr), float(mean_degree_thr))
        return self

    def predict(self, stats: np.ndarray) -> np.ndarray:
        density_thr, min_dist_thr, mean_degree_thr = self.thresholds
        features = np.column_stack([stats[:, 0], stats[:, 1], stats[:, 4], stats[:, 5], stats[:, 7], stats[:, 8]])
        return self._predict_features(features, density_thr, min_dist_thr, mean_degree_thr)

    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([shallow_feature_vector(s) for s in snapshots])
        density_thr, min_dist_thr, mean_degree_thr = self.thresholds
        pred = self._predict_features(x, density_thr, min_dist_thr, mean_degree_thr)
        return pred, (pred > 1.0).astype(float)

    @staticmethod
    def _predict_features(features: np.ndarray, density_thr: float, min_dist_thr: float, mean_degree_thr: float) -> np.ndarray:
        sparse_flag = features[:, 0] < density_thr
        separated_flag = features[:, 1] > min_dist_thr
        low_degree_flag = features[:, 2] < mean_degree_thr
        pred = 1.0 + sparse_flag.astype(float) + separated_flag.astype(float) + low_degree_flag.astype(float)
        return np.clip(pred, 1.0, None)


def fit_heuristic(train_data: list[Snapshot]) -> TrainResult:
    return TrainResult(model=HeuristicPredictor().fit(train_data), model_name="Density + min-distance heuristics", inference_ms=0.05)


class UnionFindDetectionOracle:
    def predict_snapshots(self, snapshots: list[Snapshot]) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray([betti_zero(snap.adjacency) for snap in snapshots], dtype=float)
        return beta, (beta > 1.0).astype(float)


def fit_union_find_oracle() -> TrainResult:
    return TrainResult(model=UnionFindDetectionOracle(), model_name="Union-Find detection oracle", inference_ms=0.02)


def fit_shallow_models(train_data: list[Snapshot]) -> dict[str, object]:
    x = np.stack([shallow_feature_vector(snap) for snap in train_data])
    y = np.asarray([snap.beta_target for snap in train_data], dtype=float)
    if SKLEARN_AVAILABLE:
        candidates = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RF": RandomForestRegressor(n_estimators=150, random_state=42),
            "GBR": GradientBoostingRegressor(random_state=42),
            "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        }
    else:
        candidates = {
            "Linear": NumpyPipeline(LinearLeastSquares()),
            "Ridge": NumpyPipeline(RidgeLeastSquares(alpha=1.0)),
            "RF": NumpyPipeline(KNNRegressor(k=9)),
            "GBR": NumpyPipeline(RandomFeatureRegressor(hidden_dim=48, seed=11)),
            "MLP": NumpyPipeline(RandomFeatureRegressor(hidden_dim=96, seed=23)),
        }
    fitted = {}
    for name, reg in candidates.items():
        model = NumpyPipeline(reg) if SKLEARN_AVAILABLE else reg
        model.fit(x, y)
        fitted[name] = model
    return fitted


def select_best_shallow(train_data: list[Snapshot], val_data: list[Snapshot]) -> TrainResult:
    models = fit_shallow_models(train_data)
    x_val = np.stack([shallow_feature_vector(snap) for snap in val_data])
    y_val = np.asarray([snap.beta_target for snap in val_data], dtype=float)
    best_name, best_model, best_mae = None, None, float("inf")
    for name, model in models.items():
        mae = mean_absolute_error(y_val, model.predict(x_val)) if SKLEARN_AVAILABLE else mae_metric(y_val, model.predict(x_val))
        if mae < best_mae:
            best_name, best_model, best_mae = name, model, mae
    model_name = f"Shallow ML ({best_name})"
    return TrainResult(model=SnapshotRegressorWrapper(model_name, best_model, "shallow"), model_name=model_name, inference_ms=0.1)


def _fit_snapshot_surrogate(train_data: list[Snapshot], val_data: list[Snapshot], model_name: str, mode: str, estimator) -> TrainResult:
    x_train = np.stack([snapshot_feature_vector(s, mode) for s in train_data])
    y_train = np.asarray([s.beta_target for s in train_data], dtype=float)
    x_val = np.stack([snapshot_feature_vector(s, mode) for s in val_data])
    estimator.fit(x_train, y_train)
    start = time.perf_counter()
    _ = estimator.predict(x_val[: min(len(x_val), 64)])
    inference_ms = ((time.perf_counter() - start) * 1000.0) / max(min(len(x_val), 64), 1)
    return TrainResult(
        model=SnapshotRegressorWrapper(model_name, estimator, mode),
        model_name=model_name,
        inference_ms=inference_ms,
        risk_inference_ms=0.0,
    )


def _fit_temporal_surrogate(train_data: list[Snapshot], val_data: list[Snapshot], model_name: str, mode: str, estimator, window: int) -> TrainResult:
    wrapper = TemporalRegressorWrapper(model_name, estimator, mode, window)
    x_train, aligned_train = wrapper._windows(train_data)
    y_train = np.asarray([s.beta_target for s in aligned_train], dtype=float)
    x_val, aligned_val = wrapper._windows(val_data)
    estimator.fit(x_train, y_train)
    start = time.perf_counter()
    _ = estimator.predict(x_val[: min(len(x_val), 64)])
    inference_ms = ((time.perf_counter() - start) * 1000.0) / max(min(len(x_val), 64), 1)
    return TrainResult(
        model=TemporalRegressorWrapper(model_name, estimator, mode, window),
        model_name=model_name,
        inference_ms=inference_ms,
        risk_inference_ms=0.0,
    )


def train_surrogate_model(model_name: str, train_data: list[Snapshot], val_data: list[Snapshot]) -> TrainResult:
    if SKLEARN_AVAILABLE:
        ridge_est = lambda alpha=1.0: NumpyPipeline(Ridge(alpha=alpha))
        rf_est = lambda: NumpyPipeline(RandomForestRegressor(n_estimators=180, random_state=42))
        et_est = lambda: NumpyPipeline(ExtraTreesRegressor(n_estimators=220, random_state=52))
        gbr_est = lambda: NumpyPipeline(GradientBoostingRegressor(random_state=42))
        mlp_small = lambda: NumpyPipeline(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42))
        mlp_big = lambda: NumpyPipeline(MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
        hybrid_est = lambda: EnsembleRegressor([ridge_est(alpha=0.35), rf_est(), et_est(), gbr_est(), mlp_big()])
    else:
        ridge_est = lambda alpha=1.0: NumpyPipeline(RidgeLeastSquares(alpha=alpha))
        rf_est = lambda: NumpyPipeline(KNNRegressor(k=9))
        gbr_est = lambda: NumpyPipeline(RandomFeatureRegressor(hidden_dim=64, seed=19))
        mlp_small = lambda: NumpyPipeline(RandomFeatureRegressor(hidden_dim=96, seed=31))
        mlp_big = lambda: NumpyPipeline(RandomFeatureRegressor(hidden_dim=160, seed=41))
        hybrid_est = lambda: EnsembleRegressor([ridge_est(alpha=0.2), rf_est(), mlp_big()])
    if model_name == "GCN":
        estimator = ridge_est(alpha=1.0)
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "gcn", estimator)
    if model_name == "GAT":
        estimator = gbr_est()
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "gat", estimator)
    if model_name == "GraphSAGE":
        estimator = rf_est()
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "graphsage", estimator)
    if model_name == "PI+MLP":
        estimator = mlp_small()
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "pi_only", estimator)
    if model_name == "FANET-TopoGNN":
        estimator = hybrid_est()
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "hybrid", estimator)
    if model_name == "FANET-TopoGNN (concat)":
        estimator = mlp_big()
        return _fit_snapshot_surrogate(train_data, val_data, model_name, "hybrid", estimator)
    kind, window_str = model_name.split(":")
    window = int(window_str)
    estimator = gbr_est()
    return _fit_temporal_surrogate(train_data, val_data, model_name, "hybrid", estimator, window)
