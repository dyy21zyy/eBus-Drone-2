from __future__ import annotations

import numpy as np
from src.env.action_space import A_FULL
from .base_policy import BasePolicy


class UniformPolicy(BasePolicy):
    def __init__(self, duration: int = 60):
        self.duration = duration
        self.index = A_FULL.index(duration) if duration in A_FULL else 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        _ = observation, info
        if action_mask[self.index] == 1:
            return self.index
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        return feasible[0] if feasible else 0
