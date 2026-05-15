from __future__ import annotations

import numpy as np

from .dataset import Snapshot
from .geometry import pairwise_distances

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional dependency
    Data = None


def torch_geometric_available() -> bool:
    return torch is not None and Data is not None


def _require_pyg() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyG export requires torch. Install the 'deep' optional dependencies first.")
    if Data is None:
        raise ModuleNotFoundError("PyG export requires torch-geometric. Install the 'pyg' optional dependency.")


def _edge_index_from_adjacency(adj: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(adj > 0)
    if rows.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack([rows, cols]).astype(np.int64)


def snapshot_to_pyg_data(snapshot: Snapshot, include_edge_attr: bool = True):
    _require_pyg()
    edge_index_np = _edge_index_from_adjacency(snapshot.adjacency)
    data = Data(
        x=torch.as_tensor(snapshot.node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index_np, dtype=torch.long),
        pos=torch.as_tensor(snapshot.positions, dtype=torch.float32),
        y=torch.as_tensor([snapshot.beta_target], dtype=torch.float32),
    )
    data.pi = torch.as_tensor(snapshot.pi, dtype=torch.float32)
    data.stats = torch.as_tensor(snapshot.stats, dtype=torch.float32)
    data.beta_current = torch.as_tensor([snapshot.beta_current], dtype=torch.float32)
    data.frag_within_horizon = torch.as_tensor([snapshot.frag_within_horizon], dtype=torch.long)
    data.run_id = snapshot.run_id
    data.time_index = int(snapshot.time_index)
    data.mobility = snapshot.mobility
    data.n_nodes = int(snapshot.n_nodes)
    data.link_model = snapshot.link_model
    data.graph_policy = snapshot.graph_policy
    data.radio_scenario = snapshot.radio_scenario
    if include_edge_attr and edge_index_np.shape[1] > 0:
        distances = pairwise_distances(snapshot.positions)
        data.edge_attr = torch.as_tensor(distances[edge_index_np[0], edge_index_np[1]][:, None], dtype=torch.float32)
    return data


def snapshots_to_pyg_data(snapshots: list[Snapshot], include_edge_attr: bool = True):
    return [snapshot_to_pyg_data(snapshot, include_edge_attr=include_edge_attr) for snapshot in snapshots]


__all__ = ["snapshot_to_pyg_data", "snapshots_to_pyg_data", "torch_geometric_available"]
