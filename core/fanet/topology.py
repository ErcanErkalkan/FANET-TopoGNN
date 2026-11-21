from __future__ import annotations

import numpy as np

from .geometry import pairwise_distances


def mst_h0_deaths(points: np.ndarray, max_radius: float) -> np.ndarray:
    n = len(points)
    if n <= 1:
        return np.zeros(1, dtype=float)
    dist = pairwise_distances(points)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    min_cost = dist[0].copy()
    deaths = []
    for _ in range(n - 1):
        candidates = np.where(~visited, min_cost, np.inf)
        nxt = int(np.argmin(candidates))
        cost = float(candidates[nxt])
        if not np.isfinite(cost):
            cost = max_radius
        deaths.append(min(cost, max_radius))
        visited[nxt] = True
        min_cost = np.minimum(min_cost, dist[nxt])
    return np.asarray(deaths if deaths else [0.0], dtype=float)


def persistence_image(points: np.ndarray, resolution: int, sigma: float, max_radius: float) -> np.ndarray:
    deaths = mst_h0_deaths(points, max_radius=max_radius)
    grid = np.linspace(0.0, max_radius, resolution)
    xx, yy = np.meshgrid(grid, grid)
    image = np.zeros_like(xx)
    for death in deaths:
        persistence = max(death, 0.0)
        weight = persistence
        image += weight * np.exp(-((xx ** 2) + ((yy - persistence) ** 2)) / (2.0 * sigma * sigma))
    if np.max(image) > 0:
        image = image / np.max(image)
    return image.astype(np.float32)
