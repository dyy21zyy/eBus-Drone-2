from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class MaxFeasiblePolicy(BasePolicy):
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        _ = observation, info
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        proposal = max(feasible) if feasible else 0
        return self.clip_to_feasible(proposal, action_mask)
