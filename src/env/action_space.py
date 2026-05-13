from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_ACTION_SET_SECONDS: List[int] = [0, 15, 30, 45, 60, 75, 90, 105, 120]
A_FULL: List[int] = DEFAULT_ACTION_SET_SECONDS


def get_action_set(config: dict | None = None) -> List[int]:
    if config is None:
        return list(DEFAULT_ACTION_SET_SECONDS)
    vals = config.get("charging", {}).get("action_set_seconds", DEFAULT_ACTION_SET_SECONDS)
    return [int(v) for v in vals]


def action_index_to_duration(action_index: int, action_set: list[int] | None = None) -> int:
    actions = action_set or DEFAULT_ACTION_SET_SECONDS
    if action_index < 0 or action_index >= len(actions):
        raise IndexError(f"Invalid action index: {action_index}")
    return actions[action_index]


def max_charge_duration_sec(current_battery_kwh: float, capacity_kwh: float, power_kw: float, eta: float, u_max_sec: float) -> float:
    if power_kw <= 0 or eta <= 0:
        return 0.0
    remaining_kwh = max(0.0, capacity_kwh - current_battery_kwh)
    return min(max(0.0, 3600.0 * remaining_kwh / (eta * power_kw)), max(0.0, u_max_sec))


def feasible_action_mask(available_chargers: int, current_battery_kwh: float, capacity_kwh: float, power_kw: float, eta: float, action_set: list[int] | None = None, max_single_stop_seconds: float | None = None, atol: float = 1e-9) -> np.ndarray:
    actions = action_set or DEFAULT_ACTION_SET_SECONDS
    mask = np.zeros(len(actions), dtype=np.int8)
    mask[0] = 1
    if available_chargers <= 0:
        return mask
    u_max = max(actions) if max_single_stop_seconds is None else max(0.0, float(max_single_stop_seconds))
    max_feasible_duration_sec = max_charge_duration_sec(current_battery_kwh, capacity_kwh, power_kw, eta, u_max)
    for i, u_sec in enumerate(actions):
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
