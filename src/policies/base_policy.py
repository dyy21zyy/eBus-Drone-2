from __future__ import annotations

import numpy as np


class BasePolicy:
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        return feasible[0] if feasible else 0

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return self.select_action(observation, action_mask)
