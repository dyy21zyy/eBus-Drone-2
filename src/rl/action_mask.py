from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionMaskInputs:
    available_chargers: int
    current_battery_kwh: float
    capacity_kwh: float
    power_kw: float
    eta: float
    max_single_stop_seconds: float
    atol: float = 1e-9


def build_feasible_action_mask(action_set_seconds: list[int], inputs: ActionMaskInputs) -> np.ndarray:
    """Build paper-aligned feasibility mask.

    Hard constraints in mask:
    - no charger available => only action 0 feasible
    - remaining bus battery capacity (prevent overcharge)
    - no waiting for charger

    Soft constraints (e.g., station power overload) are intentionally excluded.
    """
    actions = [int(a) for a in action_set_seconds]
    mask = np.zeros(len(actions), dtype=np.int8)
    if len(mask) == 0:
        return mask
    mask[0] = 1  # always keep no-charge feasible
    if int(inputs.available_chargers) <= 0:
        return mask
    if float(inputs.power_kw) <= 0.0 or float(inputs.eta) <= 0.0:
        return mask

    remaining_kwh = max(0.0, float(inputs.capacity_kwh) - float(inputs.current_battery_kwh))
    max_from_battery_sec = 3600.0 * remaining_kwh / (float(inputs.power_kw) * float(inputs.eta))
    max_feasible_sec = min(max(0.0, float(inputs.max_single_stop_seconds)), max(0.0, max_from_battery_sec))
    for i, dur in enumerate(actions):
        if float(dur) <= max_feasible_sec + float(inputs.atol):
            mask[i] = 1
    return mask


def feasible_actions(mask: np.ndarray) -> list[int]:
    return [i for i, v in enumerate(np.asarray(mask).tolist()) if int(v) == 1]


def repair_to_nearest_not_exceeding(action_index: int, mask: np.ndarray) -> int:
    feasible = feasible_actions(mask)
    if not feasible:
        return 0
    if int(action_index) in feasible:
        return int(action_index)
    candidates = [i for i in feasible if i <= int(action_index)]
    return max(candidates) if candidates else 0
