from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class MaxFeasiblePolicy(BasePolicy):
    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        _ = observation
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        return max(feasible) if feasible else 0
