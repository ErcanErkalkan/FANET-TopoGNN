from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from .dataset import Snapshot, split_by_run
from .geometry import pairwise_distances
from .graph_utils import adjacency_from_radius, betti_zero, connected_components, shortest_path_matrix
from .radio import build_link_adjacency
from .training import TORCH_AVAILABLE, SnapshotDataset, TemporalWindowDataset, collate_snapshots, collate_temporal

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader
else:
    torch = None
    DataLoader = None


def bootstrap_ci(values: np.ndarray, rounds: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(1234)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(max(rounds, 1))]
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    rounds: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(1234)
    n = len(y_true)
    metrics = []
    for _ in range(max(rounds, 1)):
        idx = rng.integers(0, n, size=n)
        metrics.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.quantile(metrics, alpha / 2.0)), float(np.quantile(metrics, 1.0 - alpha / 2.0))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "Risk_Precision": precision,
        "Risk_Recall": recall,
        "Risk_F1": f1,
        "Risk_Specificity": specificity,
        "Risk_Accuracy": accuracy,
    }


def risk_probability_metrics(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Return ranking and calibration metrics from observed binary labels and scores."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    if positives and negatives:
        order = np.argsort(-probabilities, kind="mergesort")
        sorted_labels = labels[order]
        true_positive = np.cumsum(sorted_labels == 1)
        false_positive = np.cumsum(sorted_labels == 0)
        recall = true_positive / positives
        precision = true_positive / np.maximum(true_positive + false_positive, 1)
        pr_auc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

        thresholds = np.r_[np.inf, np.unique(probabilities)[::-1], -np.inf]
        tpr = []
        fpr = []
        for threshold in thresholds:
            predicted = probabilities >= threshold
            tpr.append(float(np.sum(predicted & (labels == 1))) / positives)
            fpr.append(float(np.sum(predicted & (labels == 0))) / negatives)
        tpr_array = np.asarray(tpr, dtype=float)
        fpr_array = np.asarray(fpr, dtype=float)
        roc_auc = float(np.sum(np.diff(fpr_array) * (tpr_array[1:] + tpr_array[:-1]) * 0.5))
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else float("nan")
    ece = 0.0
    if len(labels):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for index in range(n_bins):
            if index == n_bins - 1:
                mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
            else:
                mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(float(np.mean(probabilities[mask])) - float(np.mean(labels[mask])))
    return {
        "Risk_PR_AUC": pr_auc,
        "Risk_ROC_AUC": roc_auc,
        "Risk_Brier": brier,
        "Risk_ECE": float(ece),
    }


def threshold_sensitivity_rows(
    y_true: np.ndarray,
    scores: np.ndarray,
    dt: float,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict[str, float]]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    duration_minutes = max(len(labels) * float(dt) / 60.0, 1e-12)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(labels, predicted)
        false_alarms = int(np.sum((labels == 0) & (predicted == 1)))
        rows.append(
            {
                "Threshold": float(threshold),
                **metrics,
                "False_Alarms": float(false_alarms),
                "False_Alarms_per_minute": float(false_alarms / duration_minutes),
            }
        )
    return rows


def predict_torch(model: torch.nn.Module, data: list[Snapshot], temporal_window: int | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    if temporal_window is None:
        loader = DataLoader(SnapshotDataset(data), batch_size=64, shuffle=False, collate_fn=collate_snapshots)
    else:
        loader = DataLoader(TemporalWindowDataset(data, temporal_window), batch_size=64, shuffle=False, collate_fn=collate_temporal)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(batch["x"], batch["adj"], batch["pi"], batch["mask"]).detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)


def predict_generic(model_result, data: list[Snapshot]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Snapshot], float]:
    if hasattr(model_result.model, "predict_snapshots"):
        start = time.perf_counter()
        out = model_result.model.predict_snapshots(data)
        elapsed = (time.perf_counter() - start) * 1000.0
        if isinstance(out, tuple) and len(out) == 3:
            preds, risk_scores, aligned = out
        elif isinstance(out, tuple) and len(out) == 2:
            preds, risk_scores = out
            aligned = data
        else:
            preds, risk_scores, aligned = out, (np.asarray(out, dtype=float) > 1.0).astype(float), data
        per_item = elapsed / max(len(preds), 1)
        threshold = float(getattr(model_result.model, "risk_threshold", 0.5))
        return np.asarray(preds, dtype=float), np.asarray(risk_scores, dtype=float), np.full(len(preds), per_item), aligned, threshold
    if hasattr(model_result.model, "predict"):
        stats = np.stack([snap.stats for snap in data])
        start = time.perf_counter()
        preds = model_result.model.predict(stats)
        elapsed = (time.perf_counter() - start) * 1000.0 / max(len(stats), 1)
        risk_scores = (np.asarray(preds, dtype=float) > 1.0).astype(float)
        return np.asarray(preds, dtype=float), risk_scores, np.full(len(preds), elapsed), data, 0.5
    if ":" in model_result.model_name:
        window = int(model_result.model_name.split(":")[1])
        preds = predict_torch(model_result.model, data, temporal_window=window)
        grouped = split_by_run(data)
        aligned = []
        for _, seq in sorted(grouped.items()):
            seq = sorted(seq, key=lambda item: item.time_index)
            aligned.extend(seq[window - 1 :])
        risk_scores = (np.asarray(preds, dtype=float) > 1.0).astype(float)
        return preds, risk_scores, np.full(len(preds), model_result.inference_ms), aligned, 0.5
    preds = predict_torch(model_result.model, data, temporal_window=None)
    risk_scores = (np.asarray(preds, dtype=float) > 1.0).astype(float)
    return preds, risk_scores, np.full(len(preds), model_result.inference_ms), data, 0.5


def event_warning_leads(run_snaps: list[Snapshot], risk_scores: np.ndarray, dt: float, horizon_steps: int, risk_threshold: float) -> tuple[list[float], list[float]]:
    """Compute event-based warning lead before physical-adjacency changes.

    The fragmentation-risk target is evaluated separately as the exact-horizon event beta0(t+h) > 1.
    This statistic measures how early a risk score crosses threshold before any
    edge-set change in the current physical graph.
    """
    flip_indices = [
        idx
        for idx in range(1, len(run_snaps))
        if run_snaps[idx - 1].adjacency.shape == run_snaps[idx].adjacency.shape
        and np.any(run_snaps[idx - 1].adjacency != run_snaps[idx].adjacency)
    ]
    if not flip_indices:
        return [], []
    if len(flip_indices) > 1:
        mean_inter_flip_ms = float(np.mean(np.diff(flip_indices)) * dt * 1000.0)
    else:
        mean_inter_flip_ms = float(max(horizon_steps, 1) * dt * 1000.0)
    leads = []
    normalised = []
    for idx in flip_indices:
        warned_idx = None
        start = max(0, idx - horizon_steps)
        for look in range(start, idx):
            if run_snaps[look].beta_current <= 1 and risk_scores[look] >= risk_threshold and (look + horizon_steps) >= idx:
                warned_idx = look
                break
        lead_ms = 0.0 if warned_idx is None else (idx - warned_idx) * dt * 1000.0
        leads.append(lead_ms)
        normalised.append(lead_ms / max(mean_inter_flip_ms, 1e-8))
    return leads, normalised


def evaluate_predictions(model_name: str, test_data: list[Snapshot], preds: np.ndarray, risk_scores: np.ndarray, inference_ms: np.ndarray, dt: float, bootstrap_rounds: int, horizon_steps: int, risk_threshold: float) -> tuple[dict, list[float], list[float]]:
    aligned_test = test_data
    y_true = np.asarray([snap.beta_target for snap in aligned_test], dtype=float)
    y_risk_true = np.asarray([snap.frag_at_horizon for snap in aligned_test], dtype=int)
    y_risk_pred = (risk_scores >= risk_threshold).astype(int)
    summary = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, preds),
        "MSE": mean_squared_error(y_true, preds),
        "R2": r2_score(y_true, preds),
        "MAE_CI_low": bootstrap_metric_ci(y_true, preds, mean_absolute_error, rounds=bootstrap_rounds)[0],
        "MAE_CI_high": bootstrap_metric_ci(y_true, preds, mean_absolute_error, rounds=bootstrap_rounds)[1],
        "R2_CI_low": bootstrap_metric_ci(y_true, preds, r2_score, rounds=bootstrap_rounds)[0],
        "R2_CI_high": bootstrap_metric_ci(y_true, preds, r2_score, rounds=bootstrap_rounds)[1],
        "Inference_ms": float(np.mean(inference_ms)),
    }
    summary.update(classification_metrics(y_risk_true, y_risk_pred))
    summary.update(risk_probability_metrics(y_risk_true, risk_scores))
    false_alarms = int(np.sum((y_risk_true == 0) & (y_risk_pred == 1)))
    duration_minutes = max(len(y_risk_true) * float(dt) / 60.0, 1e-12)
    summary["False_Alarms"] = float(false_alarms)
    summary["False_Alarms_per_minute"] = float(false_alarms / duration_minutes)
    run_pairs: dict[str, list[tuple[Snapshot, float]]] = {}
    for snap, risk in zip(aligned_test, risk_scores):
        run_pairs.setdefault(snap.run_id, []).append((snap, float(risk)))
    lead_times = []
    normalised_leads = []
    for _, pairs in sorted(run_pairs.items()):
        pairs = sorted(pairs, key=lambda item: item[0].time_index)
        items = [snap for snap, _ in pairs]
        run_risks = np.asarray([risk for _, risk in pairs], dtype=float)
        leads, normalised = event_warning_leads(items, run_risks, dt, horizon_steps, risk_threshold)
        lead_times.extend(leads)
        normalised_leads.extend(normalised)
    return summary, lead_times, normalised_leads


def summarise_leads(model_name: str, lead_times: list[float], normalised_leads: list[float] | None = None) -> dict:
    arr = np.asarray(lead_times if lead_times else [0.0], dtype=float)
    norm = np.asarray(normalised_leads if normalised_leads else [0.0], dtype=float)
    return {
        "Model": model_name,
        "Lead_5th_ms": float(np.quantile(arr, 0.05)),
        "Lead_median_ms": float(np.quantile(arr, 0.5)),
        "Lead_IQR_ms": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "Lead_95th_ms": float(np.quantile(arr, 0.95)),
        "Lead_mean_ms": float(arr.mean()),
        "Lead_norm_mean": float(norm.mean()),
    }


def _betweenness_scores(adj: np.ndarray) -> np.ndarray:
    n_nodes = adj.shape[0]
    neighbours = [np.flatnonzero(adj[node] > 0).astype(int).tolist() for node in range(n_nodes)]
    scores = np.zeros(n_nodes, dtype=float)
    for source in range(n_nodes):
        stack: list[int] = []
        predecessors: list[list[int]] = [[] for _ in range(n_nodes)]
        sigma = np.zeros(n_nodes, dtype=float)
        distance = np.full(n_nodes, -1, dtype=int)
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            stack.append(vertex)
            for neighbour in neighbours[vertex]:
                if distance[neighbour] < 0:
                    queue.append(neighbour)
                    distance[neighbour] = distance[vertex] + 1
                if distance[neighbour] == distance[vertex] + 1:
                    sigma[neighbour] += sigma[vertex]
                    predecessors[neighbour].append(vertex)

        dependency = np.zeros(n_nodes, dtype=float)
        for vertex in reversed(stack):
            for predecessor in predecessors[vertex]:
                if sigma[vertex] > 0:
                    dependency[predecessor] += (sigma[predecessor] / sigma[vertex]) * (1.0 + dependency[vertex])
            if vertex != source:
                scores[vertex] += dependency[vertex]
    return scores / 2.0


def _select_relays(adj: np.ndarray, components: list[list[int]], max_relays: int) -> list[int]:
    degree = adj.sum(axis=1).astype(float)
    betweenness = _betweenness_scores(adj)
    degree_norm = degree / max(float(degree.max()), 1.0)
    between_norm = betweenness / max(float(betweenness.max()), 1.0)
    score = 0.55 * degree_norm + 0.45 * between_norm
    relays: list[int] = []
    for comp in components[: max(max_relays, 1)]:
        if not comp:
            continue
        best = max(comp, key=lambda node: score[node])
        if best not in relays:
            relays.append(int(best))
        if len(relays) >= max_relays:
            break
    return relays


def _steer_relays_towards_component_midpoints(
    positions: np.ndarray,
    components: list[list[int]],
    relays: list[int],
    step_fraction: float = 0.35,
    min_separation: float = 10.0,
) -> np.ndarray:
    if len(components) < 2 or not relays:
        return positions
    adjusted = positions.copy()
    centroids = [positions[comp].mean(axis=0) for comp in components]
    for relay in relays:
        own_idx = next((idx for idx, comp in enumerate(components) if relay in comp), 0)
        other_indices = [idx for idx in range(len(components)) if idx != own_idx]
        if not other_indices:
            continue
        target_idx = min(other_indices, key=lambda idx: float(np.linalg.norm(centroids[idx] - centroids[own_idx])))
        midpoint = 0.5 * (centroids[own_idx] + centroids[target_idx])
        adjusted[relay] = positions[relay] + step_fraction * (midpoint - positions[relay])
        for _ in range(4):
            deltas = adjusted[relay] - adjusted
            distances = np.linalg.norm(deltas, axis=1)
            distances[relay] = np.inf
            if float(distances.min()) >= min_separation:
                break
            adjusted[relay] = 0.5 * (adjusted[relay] + positions[relay])
    return adjusted


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _scenario_sim_config(sim_cfg: dict | None, snap: Snapshot) -> dict | None:
    if sim_cfg is None:
        return None
    scenario_cfg = copy.deepcopy(sim_cfg)
    scenarios = sim_cfg.get("radio_scenarios")
    if not scenarios:
        return scenario_cfg

    base_physical = dict(sim_cfg.get("physical_layer", {}))
    wanted = _normalise_label(snap.radio_scenario)
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        name = _normalise_label(scenario.get("name", "default"))
        if name != wanted:
            continue
        if "physical_layer" in scenario and isinstance(scenario["physical_layer"], dict):
            overrides = dict(scenario["physical_layer"])
        else:
            overrides = {key: value for key, value in scenario.items() if key != "name"}
        physical = dict(base_physical)
        physical.update(overrides)
        scenario_cfg.pop("radio_scenarios", None)
        scenario_cfg["physical_layer"] = physical
        scenario_cfg["radio_scenario"] = name
        return scenario_cfg
    scenario_cfg.pop("radio_scenarios", None)
    return scenario_cfg


def _controller_link_seed(snap: Snapshot) -> int:
    marker = "_seed"
    if marker in snap.run_id:
        try:
            run_seed = int(snap.run_id.rsplit(marker, 1)[1])
            return int((run_seed * 1_000_003 + snap.time_index) % (2**63 - 1))
        except ValueError:
            pass
    digest = hashlib.sha256(f"{snap.run_id}:{snap.time_index}:controller".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _relay_incident_mask(n_nodes: int, relays: list[int]) -> np.ndarray:
    mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    for relay in relays:
        mask[relay, :] = True
        mask[:, relay] = True
    np.fill_diagonal(mask, False)
    return mask


def _controlled_physical_adjacency(
    snap: Snapshot,
    positions: np.ndarray,
    boost: float,
    relays: list[int],
    sim_cfg: dict | None,
) -> np.ndarray:
    base_adj = (snap.adjacency > 0).astype(np.float32)
    if not relays:
        return base_adj

    scenario_cfg = _scenario_sim_config(sim_cfg, snap)
    distances = pairwise_distances(positions)
    boosted_radius = float(snap.radius) * float(boost)
    if scenario_cfg is not None:
        candidate = build_link_adjacency(
            distances,
            boosted_radius,
            np.random.default_rng(_controller_link_seed(snap)),
            scenario_cfg,
        )
    else:
        candidate = adjacency_from_radius(distances, boosted_radius)

    relay_mask = _relay_incident_mask(base_adj.shape[0], relays)
    adj = base_adj.copy()
    adj[relay_mask] = 0.0
    adj = np.maximum(adj, candidate.astype(np.float32) * relay_mask.astype(np.float32))
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def _relay_shortest_path_pairs(sp: np.ndarray, relays: list[int]) -> int:
    if not relays:
        return 0
    finite = np.isfinite(sp) & (sp > 0)
    routed = np.zeros_like(finite, dtype=bool)
    for relay in relays:
        via_relay = sp[:, [relay]] + sp[[relay], :]
        routed |= finite & np.isclose(via_relay, sp)
        routed[relay, :] = False
        routed[:, relay] = False
    return int(routed.sum())


def run_network_controller(
    snaps: list[Snapshot],
    preds: np.ndarray,
    risk_scores: np.ndarray,
    boost: float,
    risk_threshold: float,
    dtn_delivery_fraction: float = 0.65,
    sim_config: dict | None = None,
) -> dict:
    connected_ticks = 0
    delivered = 0
    generated = 0
    rerouted_pairs = 0
    buffered_pairs = 0
    relay_actions = 0
    delays = []
    for snap, _pred, risk in zip(snaps[-len(preds) :], preds, risk_scores):
        positions = snap.positions.copy()
        base_adj = (snap.adjacency > 0).astype(np.float32)
        relays: list[int] = []
        proactive = risk >= risk_threshold
        if risk >= risk_threshold:
            components = connected_components(base_adj)
            components = sorted(components, key=len, reverse=True)
            relays = _select_relays(base_adj, components, max_relays=2)
            relay_actions += len(relays)
            if len(components) > 1 and relays:
                positions = _steer_relays_towards_component_midpoints(positions, components, relays)
            adj = _controlled_physical_adjacency(snap, positions, boost, relays, sim_config)
        else:
            adj = base_adj
        beta = betti_zero(adj)
        connected_ticks += int(beta == 1)
        sp = shortest_path_matrix(adj)
        finite_mask = np.isfinite(sp) & (sp > 0)
        finite = sp[finite_mask]
        generated_tick = snap.n_nodes * (snap.n_nodes - 1)
        generated += generated_tick
        immediate_delivered = int(finite.size)
        disconnected_pairs = max(generated_tick - immediate_delivered, 0)
        dtn_delivered = int(round(dtn_delivery_fraction * disconnected_pairs)) if proactive else 0
        delivered += immediate_delivered + dtn_delivered
        if relays:
            rerouted_pairs += _relay_shortest_path_pairs(sp, relays)
        buffered_pairs += dtn_delivered
        if finite.size:
            delays.extend((finite * 10.0).astype(float).tolist())
        if dtn_delivered:
            delays.extend([float(120.0 + max(beta - 1, 0) * 25.0)] * dtn_delivered)
    return {
        "Connectivity ratio": connected_ticks / max(len(preds), 1),
        "PDR (%)": 100.0 * delivered / max(generated, 1),
        "Avg. end-to-end delay (ms)": float(np.mean(delays) if delays else 0.0),
        "Proactive reroute (%)": 100.0 * rerouted_pairs / max(generated, 1),
        "DTN buffered (%)": 100.0 * buffered_pairs / max(generated, 1),
        "Relay actions": float(relay_actions),
    }


def paired_statistics(reference_name: str, reference_errors: np.ndarray, candidate_name: str, candidate_errors: np.ndarray) -> dict:
    n = min(len(reference_errors), len(candidate_errors))
    reference_errors = reference_errors[:n]
    candidate_errors = candidate_errors[:n]
    if n < 2 or np.allclose(reference_errors, candidate_errors):
        return {
            "reference": reference_name,
            "candidate": candidate_name,
            "paired_t_pvalue": 1.0,
            "wilcoxon_pvalue": 1.0,
            "cohens_d": 0.0,
        }
    delta = candidate_errors - reference_errors
    try:
        wilcoxon_p = float(wilcoxon(candidate_errors, reference_errors, zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    paired_t_p = float(ttest_rel(candidate_errors, reference_errors).pvalue)
    if not np.isfinite(paired_t_p):
        paired_t_p = 1.0
    delta_std = float(delta.std(ddof=1))
    cohens_d = 0.0 if delta_std < 1e-8 else float(delta.mean() / delta_std)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "paired_t_pvalue": paired_t_p,
        "wilcoxon_pvalue": wilcoxon_p,
        "cohens_d": cohens_d,
    }


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    clean = [float(p) if np.isfinite(float(p)) else 1.0 for p in pvalues]
    n = len(clean)
    if n == 0:
        return []
    indexed = sorted(enumerate(clean), key=lambda item: item[1])
    adjusted = [0.0] * n
    running = 1.0
    for rank, (idx, pval) in reversed(list(enumerate(indexed, start=1))):
        corrected = min(running, pval * n / max(rank, 1))
        running = corrected
        adjusted[idx] = corrected
    return adjusted


def save_summary_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
