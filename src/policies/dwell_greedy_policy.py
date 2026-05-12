from __future__ import annotations

import numpy as np
from src.env.action_space import A_FULL
from .base_policy import BasePolicy


class DwellGreedyPolicy(BasePolicy):
    def __init__(self, dwell_scale: float = 1.0):
        self.dwell_scale = dwell_scale

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        info = info or {}
        dwell_est = max(float(info.get("T_P_est", 0.0)), float(info.get("T_F", 0.0)))
        target = self.dwell_scale * dwell_est
        feasible = [i for i, v in enumerate(action_mask.tolist()) if v == 1]
        candidates = [i for i in feasible if A_FULL[i] <= target]
        return (max(candidates) if candidates else (max(feasible) if feasible else 0))
