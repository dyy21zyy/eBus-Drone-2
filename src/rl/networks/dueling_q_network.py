from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.feature = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.adv = nn.Linear(prev, output_dim)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.feature(obs)
        v = self.value(h)
        a = self.adv(h)
        if action_mask is None:
            mean_a = a.mean(dim=1, keepdim=True)
        else:
            mask = action_mask.float()
            denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            mean_a = (a * mask).sum(dim=1, keepdim=True) / denom
        return v + a - mean_a
