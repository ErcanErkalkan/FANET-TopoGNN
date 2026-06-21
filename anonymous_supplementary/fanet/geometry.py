from __future__ import annotations

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=-1))


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        oa = a - o
        ob = b - o
        return oa[0] * ob[1] - oa[1] * ob[0]

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 1.0
    x = points[:, 0]
    y = points[:, 1]
    return max(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))), 1.0)


def density(points: np.ndarray) -> float:
    return len(points) / polygon_area(convex_hull(points))


def normalize_positions(points: np.ndarray) -> np.ndarray:
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (points - mean) / std


def clip_positions(points: np.ndarray, area_size: float) -> np.ndarray:
    return np.clip(points, 0.0, area_size)


def heading_to_vector(heading: np.ndarray, speed: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(heading) * speed, np.sin(heading) * speed], axis=1)


def random_waypoints(rng: np.random.Generator, n_nodes: int, area_size: float) -> np.ndarray:
    return rng.uniform(0.0, area_size, size=(n_nodes, 2))
