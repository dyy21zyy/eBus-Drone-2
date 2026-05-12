from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class NoChargingPolicy(BasePolicy):
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        _ = observation, action_mask, info
        return 0
