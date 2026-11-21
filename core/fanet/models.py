from __future__ import annotations

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        a_hat = adj + eye
        deg = a_hat.sum(dim=-1)
        inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
        norm = inv_sqrt.unsqueeze(-1) * a_hat * inv_sqrt.unsqueeze(-2)
        return self.linear(norm @ x) * mask.unsqueeze(-1)


class SageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = (adj @ x) / deg
        return (self.self_linear(x) + self.neigh_linear(neigh)) * mask.unsqueeze(-1)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1):
        super().__init__()
        self.heads = max(int(heads), 1)
        self.head_dim = max(out_dim // self.heads, 1)
        self.inner_dim = self.heads * self.head_dim
        self.proj = nn.Linear(in_dim, self.inner_dim)
        self.attn = nn.Linear(2 * self.head_dim, 1)
        self.out_proj = nn.Identity() if self.inner_dim == out_dim else nn.Linear(self.inner_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        h = self.proj(x).reshape(bsz, n_nodes, self.heads, self.head_dim).transpose(1, 2)
        h_i = h.unsqueeze(3).expand(bsz, self.heads, n_nodes, n_nodes, self.head_dim)
        h_j = h.unsqueeze(2).expand(bsz, self.heads, n_nodes, n_nodes, self.head_dim)
        scores = self.act(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        eye = torch.eye(n_nodes, device=adj.device, dtype=torch.bool).unsqueeze(0)
        valid_nodes = mask > 0
        valid_edges = ((adj > 0) | eye) & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        scores = scores.masked_fill(~valid_edges.unsqueeze(1), -1e9)
        alpha = torch.softmax(scores, dim=-1)
        out = alpha @ h
        out = out.transpose(1, 2).reshape(bsz, n_nodes, self.inner_dim)
        return self.out_proj(out) * mask.unsqueeze(-1)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(in_dim, hidden_dim), GCNLayer(hidden_dim, hidden_dim)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x, adj, mask)))
        return masked_mean(x, mask)


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([SageLayer(in_dim, hidden_dim), SageLayer(hidden_dim, hidden_dim)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x, adj, mask)))
        return masked_mean(x, mask)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([GATLayer(in_dim, hidden_dim, heads=4), GATLayer(hidden_dim, hidden_dim, heads=1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x, adj, mask)))
        return masked_mean(x, mask)


class GraphRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, adj, mask)).squeeze(-1)


class PIRegressor(nn.Module):
    def __init__(self, pi_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(pi).squeeze(-1)


class FANETTopoGNN(nn.Module):
    def __init__(self, in_dim: int, pi_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gcn = GCNEncoder(in_dim, hidden_dim, dropout)
        self.pi_net = nn.Sequential(nn.Linear(pi_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        g = self.gcn(x, adj, mask)
        p = self.pi_net(pi)
        gate = self.gate(torch.cat([g, p], dim=-1))
        fused = gate * g + (1.0 - gate) * p
        return self.head(fused).squeeze(-1)


class FANETTopoGNNConcat(nn.Module):
    def __init__(self, in_dim: int, pi_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gcn = GCNEncoder(in_dim, hidden_dim, dropout)
        self.pi_net = nn.Sequential(nn.Linear(pi_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        g = self.gcn(x, adj, mask)
        p = self.pi_net(pi)
        fused = torch.cat([g, p], dim=-1)
        return self.head(fused).squeeze(-1)


class TemporalRegressor(nn.Module):
    def __init__(self, snapshot_encoder: nn.Module, hidden_dim: int, model_type: str, dropout: float):
        super().__init__()
        self.snapshot_encoder = snapshot_encoder
        self.model_type = model_type
        if model_type == "stgcn":
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.memory = nn.Parameter(torch.zeros(hidden_dim)) if model_type == "tgn" else None
        self.message = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) if model_type == "tgn" else None
        self.memory_gate = nn.GRUCell(hidden_dim, hidden_dim) if model_type == "tgn" else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, steps, nodes, feats = x.shape
        flat_x = x.reshape(bsz * steps, nodes, feats)
        flat_adj = adj.reshape(bsz * steps, nodes, nodes)
        flat_mask = mask.reshape(bsz * steps, nodes)
        embeds = self.snapshot_encoder(flat_x, flat_adj, flat_mask).reshape(bsz, steps, -1)
        if self.model_type == "stgcn":
            last = self.temporal(embeds.transpose(1, 2)).transpose(1, 2)[:, -1, :]
        else:
            out, _ = self.temporal(embeds)
            last = out[:, -1, :]
            if self.memory is not None:
                memory = self.memory.unsqueeze(0).expand(last.size(0), -1)
                last = self.memory_gate(self.message(last), memory)
        return self.head(self.dropout(torch.relu(last))).squeeze(-1)
