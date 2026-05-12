from __future__ import annotations

import numpy as np


class BasePolicy:
    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        return feasible[0] if feasible else 0
