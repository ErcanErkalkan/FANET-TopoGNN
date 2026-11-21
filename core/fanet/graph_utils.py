from __future__ import annotations

import numpy as np


def adjacency_from_radius(distances: np.ndarray, radius: float) -> np.ndarray:
    adj = (distances <= radius).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def connected_components(adj: np.ndarray) -> list[list[int]]:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neighbors = np.flatnonzero(adj[node] > 0)
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(int(nb))
        components.append(comp)
    return components


def betti_zero(adj: np.ndarray) -> int:
    return len(connected_components(adj))


def degree_features(adj: np.ndarray) -> tuple[float, float, float]:
    deg = adj.sum(axis=1)
    return float(deg.mean()), float(deg.max()), float(deg.min())


def largest_component_ratio(adj: np.ndarray) -> float:
    comps = connected_components(adj)
    largest = max(len(comp) for comp in comps)
    return largest / adj.shape[0]


def avg_clustering_coefficient(adj: np.ndarray) -> float:
    coeffs = []
    for i in range(adj.shape[0]):
        neighbors = np.flatnonzero(adj[i] > 0)
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        sub = adj[np.ix_(neighbors, neighbors)]
        links = sub.sum() / 2.0
        coeffs.append((2.0 * links) / (k * (k - 1)))
    return float(np.mean(coeffs))


def shortest_path_matrix(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist[adj > 0] = 1.0
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist
