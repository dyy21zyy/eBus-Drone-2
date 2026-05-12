from __future__ import annotations

from typing import List

import numpy as np

A_FULL: List[int] = [0, 15, 30, 45, 60, 75, 90, 105, 120]


def action_index_to_duration(action_index: int) -> int:
    if action_index < 0 or action_index >= len(A_FULL):
        raise IndexError(f"Invalid action index: {action_index}")
    return A_FULL[action_index]


def feasible_action_mask(available_chargers: int, battery: float, battery_max: float, charge_rate: float) -> np.ndarray:
    mask = np.zeros(len(A_FULL), dtype=np.int8)
    if available_chargers <= 0:
        mask[0] = 1
        return mask
    max_duration = max(0.0, (battery_max - battery) / max(charge_rate, 1e-9))
    for i, u in enumerate(A_FULL):
        if u <= max_duration + 1e-9:
            mask[i] = 1
    mask[0] = 1
    return mask


def feasible_actions(mask: np.ndarray) -> list[int]:
    return [i for i, v in enumerate(mask.tolist()) if v == 1]


def repair_action(action_index: int, mask: np.ndarray) -> int:
    feasible = feasible_actions(mask)
    if not feasible:
        return 0
    if action_index in feasible:
        return action_index
    lower_or_equal = [i for i in feasible if i <= action_index]
    return max(lower_or_equal) if lower_or_equal else 0
