from __future__ import annotations

from typing import List

import numpy as np

A_FULL: List[int] = [0, 15, 30, 45, 60, 75, 90, 105, 120]


def action_index_to_duration(action_index: int) -> int:
    if action_index < 0 or action_index >= len(A_FULL):
        raise IndexError(f"Invalid action index: {action_index}")
    return A_FULL[action_index]


def max_charge_duration_sec(current_battery_kwh: float, capacity_kwh: float, power_kw: float, eta: float, u_max_sec: float) -> float:
    if power_kw <= 0 or eta <= 0:
        return 0.0
    remaining_kwh = max(0.0, capacity_kwh - current_battery_kwh)
    return min(max(0.0, 3600.0 * remaining_kwh / (eta * power_kw)), max(0.0, u_max_sec))


def feasible_action_mask(available_chargers: int, current_battery_kwh: float, capacity_kwh: float, power_kw: float, eta: float, atol: float = 1e-9) -> np.ndarray:
    mask = np.zeros(len(A_FULL), dtype=np.int8)
    mask[0] = 1
    if available_chargers <= 0:
        return mask
    max_feasible_duration_sec = max_charge_duration_sec(current_battery_kwh, capacity_kwh, power_kw, eta, max(A_FULL))
    for i, u_sec in enumerate(A_FULL):
        if u_sec <= max_feasible_duration_sec + atol:
            mask[i] = 1
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
