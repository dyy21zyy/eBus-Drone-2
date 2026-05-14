from __future__ import annotations

import numpy as np
from src.env.action_space import A_FULL
from .base_policy import BasePolicy


class BatteryThresholdPolicy(BasePolicy):
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        info = info or {}
        e = float(info.get("E_current", 0.0)); e_min = float(info.get("E_min", 0.0)); e_max = float(info.get("E_max", 1.0))
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        if not feasible:
            return 0
        if e <= e_min + 10:
            return self.clip_to_feasible(max(feasible), action_mask)
        if e >= 0.8 * e_max:
            return self.clip_to_feasible(0, action_mask)
        target = 60
        return self.clip_to_feasible(min(feasible, key=lambda i: abs(A_FULL[i] - target)), action_mask)
