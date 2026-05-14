from __future__ import annotations

import numpy as np


class BasePolicy:
    def clip_to_feasible(self, action_index: int, action_mask: np.ndarray) -> int:
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        if not feasible:
            return 0
        if action_index in feasible:
            return int(action_index)
        lower = [i for i in feasible if i <= int(action_index)]
        return int(max(lower) if lower else feasible[0])

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        return self.clip_to_feasible(0, action_mask)

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return self.select_action(observation, action_mask)
