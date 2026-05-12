from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class NoChargingPolicy(BasePolicy):
    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        _ = observation, action_mask
        return 0
