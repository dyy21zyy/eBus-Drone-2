from __future__ import annotations

from src.rl.networks.dueling_q_network import DuelingQNetwork
from src.rl.networks.q_network import QNetwork


def build_network(obs_dim: int, action_dim: int, dueling: bool = False, hidden_dims: list[int] | None = None):
    if dueling:
        return DuelingQNetwork(obs_dim, action_dim, hidden_dims=hidden_dims)
    return QNetwork(obs_dim, action_dim, hidden_dims=hidden_dims)
