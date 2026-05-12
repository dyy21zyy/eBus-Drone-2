from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        if dueling:
            self.v = nn.Linear(128, 1)
            self.a = nn.Linear(128, action_dim)
        else:
            self.q = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.body(x)
        if not self.dueling:
            return self.q(h)
        v = self.v(h)
        a = self.a(h)
        if action_mask is None:
            mean_a = a.mean(dim=1, keepdim=True)
        else:
            denom = torch.clamp(action_mask.sum(dim=1, keepdim=True), min=1.0)
            mean_a = (a * action_mask).sum(dim=1, keepdim=True) / denom
        return v + a - mean_a
